# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded HTTP fetching and format classification for ingest URL sources."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from email.message import Message
import logging
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Sequence
from urllib.parse import unquote, urlparse

import httpx

from nemo_retriever.common.input_files import AUTO_INPUT_EXTENSIONS, MIME_TYPE_TO_EXTENSION, input_type_for_path
from nemo_retriever.common.params import UrlFetchParams


logger = logging.getLogger(__name__)

_GENERIC_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


@dataclass(frozen=True)
class FetchedUrl:
    """Disk-backed response ready for format-specific ingestion."""

    url: str
    local_path: Path
    content_type: str
    classification_filename: str
    input_type: str
    transport_path: str


@dataclass(frozen=True)
class UrlFetchFailure:
    """Expected per-source failure returned through the ingest failure contract."""

    url: str
    error_type: str
    message: str

    def as_record(self) -> dict[str, object]:
        return {
            "row_index": None,
            "source_identifier": self.url,
            "column": "url_fetch",
            "path": "error",
            "error": {
                "stage": "url_fetch",
                "type": self.error_type,
                "message": self.message,
            },
        }


class UrlFetchError(ValueError):
    """Expected response validation error for one URL source."""


class UrlResponseTooLargeError(UrlFetchError):
    """Raised when one response exceeds its configured byte limit."""


class UrlUnsupportedFormatError(UrlFetchError):
    """Raised when response metadata cannot identify a supported format."""


def normalize_urls(urls: str | Sequence[str]) -> list[str]:
    """Validate and normalize URL inputs.

    Parameters
    ----------
    urls
        One absolute HTTP(S) URL or a sequence of absolute HTTP(S) URLs.

    Returns
    -------
    list[str]
        Whitespace-trimmed URLs in caller order.

    Raises
    ------
    ValueError
        If an entry is empty, non-string, relative, or uses a non-HTTP(S)
        scheme.
    """
    values = [urls] if isinstance(urls, str) else list(urls)
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("urls() entries must be nonempty strings")
        url = value.strip()
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"urls() requires an absolute HTTP(S) URL; got {value!r}")
        normalized.append(url)
    return normalized


def _content_disposition_filename(header: str) -> str:
    if not header:
        return ""
    message = Message()
    message["content-disposition"] = header
    return message.get_filename() or ""


def _supported_suffix(name: str) -> str:
    try:
        parsed = urlparse(name)
    except ValueError:
        return ""
    suffix = PurePosixPath(unquote(parsed.path)).suffix.lower()
    return suffix if suffix in AUTO_INPUT_EXTENSIONS else ""


def _classify_response(url: str, response: httpx.Response, position: int) -> tuple[str, str, str, str]:
    content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    disposition_name = _content_disposition_filename(response.headers.get("content-disposition", ""))
    hinted_suffix = (
        _supported_suffix(disposition_name) or _supported_suffix(str(response.url)) or _supported_suffix(url)
    )
    mime_suffix = MIME_TYPE_TO_EXTENSION.get(content_type, "")

    if mime_suffix:
        suffix = hinted_suffix
        if not suffix or input_type_for_path(f"source{suffix}") != input_type_for_path(f"source{mime_suffix}"):
            suffix = mime_suffix
    elif content_type in _GENERIC_MIME_TYPES and hinted_suffix:
        suffix = hinted_suffix
    elif hinted_suffix:
        suffix = hinted_suffix
    else:
        displayed = content_type or "missing Content-Type"
        raise UrlUnsupportedFormatError(f"unsupported response format ({displayed})")

    input_type = input_type_for_path(f"source{suffix}")
    if input_type is None:
        raise UrlUnsupportedFormatError(f"unsupported response format ({content_type or suffix})")
    filename = f"url-{position:08d}{suffix}"
    transport_path = f"url-source://{position:08d}/{filename}"
    return content_type or "application/octet-stream", filename, input_type, transport_path


def _fetch_one(
    client: httpx.Client,
    url: str,
    position: int,
    params: UrlFetchParams,
    spool_dir: Path,
) -> FetchedUrl | UrlFetchFailure:
    local_path: Path | None = None
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            try:
                parsed_content_length = int(content_length) if content_length is not None else None
            except ValueError:
                parsed_content_length = None
            if parsed_content_length is not None and parsed_content_length > params.max_response_bytes:
                raise UrlResponseTooLargeError(
                    f"response exceeds max_response_bytes={params.max_response_bytes} "
                    f"(Content-Length={content_length})"
                )
            content_type, filename, input_type, transport_path = _classify_response(url, response, position)
            suffix = PurePosixPath(filename).suffix
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f"url-{position:08d}-",
                suffix=suffix,
                dir=spool_dir,
                delete=False,
            ) as spool:
                # Converters resolve source paths; use the same spelling for
                # ingestion and provenance, including macOS /var symlinks.
                local_path = Path(spool.name).resolve()
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > params.max_response_bytes:
                        raise UrlResponseTooLargeError(
                            f"response exceeds max_response_bytes={params.max_response_bytes}"
                        )
                    spool.write(chunk)
        if local_path is None:
            raise RuntimeError("URL fetch completed without creating a spool file")
        return FetchedUrl(url, local_path, content_type, filename, input_type, transport_path)
    except (httpx.HTTPError, UrlFetchError, OSError) as exc:
        if local_path is not None:
            with suppress(OSError):
                local_path.unlink()
        return UrlFetchFailure(url=url, error_type=type(exc).__name__, message=str(exc))
    except Exception:
        if local_path is not None:
            with suppress(OSError):
                local_path.unlink()
        logger.exception("Unexpected URL fetch failure at input position %d", position)
        raise


def cleanup_fetched_urls(fetched: Sequence[FetchedUrl]) -> None:
    """Remove managed spool files created for fetched URL responses."""

    parents: set[Path] = set()
    for item in fetched:
        parents.add(item.local_path.parent)
        with suppress(OSError):
            item.local_path.unlink()
    for parent in parents:
        with suppress(OSError):
            parent.rmdir()


def restore_url_source_value(value: object, source_map: dict[str, str]) -> object:
    """Restore URL provenance recursively while preserving derived suffixes.

    Exact transport identifiers become their submitted URL. Derived identifiers
    with an underscore suffix retain that suffix so page-level identities
    remain distinct.
    """

    if isinstance(value, str):
        for transport, source_url in source_map.items():
            if value == transport:
                return source_url
            if value.startswith(f"{transport}_"):
                return f"{source_url}{value[len(transport):]}"
        return value
    if isinstance(value, dict):
        return {key: restore_url_source_value(item, source_map) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_url_source_value(item, source_map) for item in value]
    return value


def fetch_urls(urls: Sequence[str], params: UrlFetchParams) -> tuple[list[FetchedUrl], list[UrlFetchFailure]]:
    """Fetch URL sources concurrently into managed disk spool files.

    Parameters
    ----------
    urls
        Validated HTTP(S) URLs in caller order.
    params
        Shared request settings, response-size bound, and concurrency limit.

    Returns
    -------
    tuple[list[FetchedUrl], list[UrlFetchFailure]]
        Successful disk-backed responses and expected per-source failures,
        each preserving caller order. Consumers must call cleanup_fetched_urls
        after reading successful responses.

    Raises
    ------
    Exception
        Unexpected implementation failures propagate after managed spool
        cleanup rather than being downgraded to source failures. Per-source
        failures are logged with traceback by the fetch worker.
    """

    if not urls:
        return [], []
    spool_dir = Path(tempfile.mkdtemp(prefix="nrl-url-fetch-"))
    with ExitStack() as cleanup:
        cleanup.callback(shutil.rmtree, spool_dir, ignore_errors=True)
        timeout = httpx.Timeout(params.request_timeout_s)
        limits = httpx.Limits(
            max_connections=params.max_concurrency,
            max_keepalive_connections=params.max_concurrency,
        )
        with httpx.Client(
            headers=params.headers,
            timeout=timeout,
            follow_redirects=params.follow_redirects,
            limits=limits,
        ) as client:
            with ThreadPoolExecutor(max_workers=params.max_concurrency, thread_name_prefix="nrl-url-fetch") as executor:
                outcomes = list(
                    executor.map(
                        lambda item: _fetch_one(client, item[1], item[0], params, spool_dir),
                        enumerate(urls),
                    )
                )

        fetched = [outcome for outcome in outcomes if isinstance(outcome, FetchedUrl)]
        failures = [outcome for outcome in outcomes if isinstance(outcome, UrlFetchFailure)]
        if fetched:
            cleanup.pop_all()
        return fetched, failures
