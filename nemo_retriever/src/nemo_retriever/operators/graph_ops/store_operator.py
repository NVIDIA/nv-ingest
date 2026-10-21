# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator for persisting post-embedding row images to storage."""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import fsspec
import numpy as np
import pandas as pd

from nemo_retriever.common.modality.ocr.shared import _crop_b64_image_by_norm_bbox
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator

logger = logging.getLogger(__name__)

_FORMAT_ALIASES = {"jpg": "jpeg"}
_SUPPORTED_FORMATS = {"png", "jpeg"}


def _normalize_image_format(image_format: str) -> str:
    fmt = str(image_format or "png").strip().lower()
    fmt = _FORMAT_ALIASES.get(fmt, fmt)
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image_format: {image_format!r}. Supported formats: png, jpeg")
    return fmt


def _sniff_image_format(raw: bytes) -> str | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


def _decode_image_b64(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value.strip():
        return None

    payload = value.strip()
    if payload.lower().startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        return base64.b64decode(payload)
    except binascii.Error as exc:
        logger.warning("Skipping store row with invalid _image_b64 payload: %s", exc)
        return None


def _filename_token(value: Any, *, fallback: str) -> str:
    if value is None or not pd.api.types.is_scalar(value) or pd.isna(value):
        return fallback

    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return token[:48] or fallback


def _source_stem(source_path: Any) -> str:
    if not isinstance(source_path, str) or not source_path.strip():
        return "image"

    parsed = urlparse(source_path.strip())
    path = unquote(parsed.path if parsed.scheme else source_path.strip()).replace("\\", "/")
    return _filename_token(Path(path).stem[:64], fallback="image")


def _row_source_path(row: pd.Series) -> Any:
    for column in ("path", "source", "source_id"):
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _build_object_key(
    *,
    raw: bytes,
    extension: str,
    source_path: Any = None,
    page_number: Any = None,
    content_type: Any = None,
    chunk_number: Any = None,
) -> str:
    image_hash = hashlib.sha1(raw).hexdigest()
    source = _source_stem(source_path)
    page = _filename_token(page_number, fallback="unknown")
    kind = _filename_token(content_type, fallback="image")
    position = f"-c{_filename_token(chunk_number, fallback='1')}" if chunk_number is not None else ""
    return f"{source}-p{page}-{kind}{position}-{image_hash}.{extension}"


def _join_storage_uri(storage_uri: str, object_key: str) -> str:
    return f"{str(storage_uri).rstrip('/')}/{object_key.lstrip('/')}"


def _stored_uri(dest_uri: str) -> str:
    parsed = urlparse(dest_uri)
    if parsed.scheme:
        return dest_uri
    return Path(dest_uri).resolve().as_uri()


def _write_image_b64(
    value: Any,
    *,
    storage_uri: str,
    storage_options: dict[str, Any],
    fallback_format: str,
    source_path: Any = None,
    page_number: Any = None,
    content_type: Any = None,
    chunk_number: Any = None,
) -> str | None:
    raw = _decode_image_b64(value)
    if raw is None:
        return None

    extension = _sniff_image_format(raw) or fallback_format
    object_key = _build_object_key(
        raw=raw,
        extension=extension,
        source_path=source_path,
        page_number=page_number,
        content_type=content_type,
        chunk_number=chunk_number,
    )
    dest_uri = _join_storage_uri(storage_uri, object_key)

    try:
        with fsspec.open(dest_uri, mode="wb", **storage_options) as f:
            f.write(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to store image to {dest_uri!r}: {exc}") from exc

    return _stored_uri(dest_uri)


def _store_nested_image_payloads(
    value: Any,
    *,
    storage_uri: str,
    storage_options: dict[str, Any],
    fallback_format: str,
    strip_base64: bool,
    source_path: Any = None,
    page_number: Any = None,
    content_type: Any = None,
    chunk_prefix: tuple[int, ...] = (),
) -> Any:
    """Persist nested ``image_b64`` values and replace them with URIs."""
    if isinstance(value, np.ndarray) and value.ndim == 1:
        # Batch element cells arrive as NumPy arrays; ``DataFrame.at`` collapses
        # those to a 0-d array, which breaks Arrow conversion of the result.
        value = value.tolist()

    if isinstance(value, list):
        return [
            _store_nested_image_payloads(
                item,
                storage_uri=storage_uri,
                storage_options=storage_options,
                fallback_format=fallback_format,
                strip_base64=strip_base64,
                source_path=source_path,
                page_number=page_number,
                content_type=content_type,
                chunk_prefix=(*chunk_prefix, index + 1),
            )
            for index, item in enumerate(value)
        ]

    if not isinstance(value, dict):
        return value

    out = dict(value)
    image_b64 = out.get("image_b64")
    stored_uri = out.get("stored_image_uri")
    if isinstance(image_b64, str) and image_b64.strip():
        if not stored_uri:
            stored_uri = _write_image_b64(
                image_b64,
                storage_uri=storage_uri,
                storage_options=storage_options,
                fallback_format=fallback_format,
                source_path=source_path,
                page_number=page_number,
                content_type=content_type,
                chunk_number="_".join(map(str, chunk_prefix)) or 1,
            )
            if stored_uri:
                out["stored_image_uri"] = stored_uri
        if strip_base64:
            out["image_b64"] = None

    for key, child in list(out.items()):
        if key in {"image_b64", "stored_image_uri"}:
            continue
        out[key] = _store_nested_image_payloads(
            child,
            storage_uri=storage_uri,
            storage_options=storage_options,
            fallback_format=fallback_format,
            strip_base64=strip_base64,
            source_path=source_path,
            page_number=page_number,
            content_type=content_type,
            chunk_prefix=chunk_prefix,
        )

    return out


def _structured_page_crop_b64(row: pd.Series, page_image_b64: Any) -> str | None:
    content_type = row.get("_content_type")
    if not isinstance(content_type, str) or not content_type.strip() or content_type == "text":
        return None

    bbox = row.get("_bbox_xyxy_norm")
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    cropped_b64, _ = _crop_b64_image_by_norm_bbox(page_image_b64, bbox_xyxy_norm=bbox)
    return cropped_b64


def _row_image_b64_with_source(row: pd.Series) -> tuple[Any, str | None]:
    value = row.get("_image_b64")
    if isinstance(value, str) and value.strip():
        return value, "_image_b64"

    value = row.get("image_b64")
    if isinstance(value, str) and value.strip():
        return value, "image_b64"

    page_image = row.get("page_image")
    if isinstance(page_image, dict):
        page_image_b64 = page_image.get("image_b64")
        cropped_b64 = _structured_page_crop_b64(row, page_image_b64)
        if cropped_b64:
            return cropped_b64, "page_image_crop"
        return page_image_b64, "page_image"

    return None, None


def _row_image_represents_page(row: pd.Series, *, image_source: str | None) -> bool:
    if image_source == "page_image":
        return True

    content_type = row.get("_content_type")
    if not isinstance(content_type, str) or not content_type.strip():
        return True
    return content_type == "text"


def _ensure_object_column(df: pd.DataFrame, column: str) -> None:
    """Convert Arrow-backed columns so Python payloads can be assigned.

    Pandas ArrowExtensionArray rejects ``DataFrame.at`` assignment of updated
    Python dicts into struct/list columns, and of a URI string into a
    null-typed column that upstream stages left empty. Object dtype accepts
    both.
    """
    if column not in df.columns:
        return
    dtype = df[column].dtype
    if isinstance(dtype, pd.ArrowDtype):
        df[column] = df[column].astype(object)


def _with_uploaded_image_uri(metadata: Any, stored_uri: str) -> Any:
    """Return metadata with the public structured-image URI synchronized."""
    if not isinstance(metadata, dict):
        return metadata

    out = dict(metadata)
    for key in ("content_metadata", "image_metadata", "table_metadata", "chart_metadata"):
        structured_metadata = out.get(key)
        if isinstance(structured_metadata, dict):
            updated = dict(structured_metadata)
            updated["uploaded_image_uri"] = stored_uri
            out[key] = updated
    return out


def _is_inherited_page_uri(row: pd.Series, stored_uri: Any, *, image_source: str | None, raw: bytes | None) -> bool:
    """Return whether a structured row only carries its page image URI."""
    content_type = row.get("_content_type")
    if not isinstance(content_type, str) or not content_type.strip() or content_type == "text":
        return False

    # A valid row-level payload will be stored as the structured asset below,
    # even when the row initially inherited the page URI during explosion.
    if raw is not None and image_source in {"_image_b64", "image_b64"}:
        return False

    page_image = row.get("page_image")
    page_uri = page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
    return (
        isinstance(stored_uri, str)
        and stored_uri.strip()
        and isinstance(page_uri, str)
        and page_uri.strip()
        and stored_uri == page_uri
    )


def _store_row_images(
    df: pd.DataFrame,
    *,
    storage_uri: str,
    storage_options: dict[str, Any] | None = None,
    image_format: str = "png",
    strip_base64: bool = True,
) -> pd.DataFrame:
    """Return a copy of *df* with ``_stored_image_uri`` set for stored rows."""
    image_columns = ("page_image", "images", "tables", "charts", "infographics", "table", "chart", "infographic")
    row_image_columns = ("_image_b64", "image_b64")
    if df.empty or (
        not any(column in df.columns for column in row_image_columns)
        and not any(column in df.columns for column in image_columns)
    ):
        return df

    out = df.copy()
    for column in (*image_columns, *row_image_columns, "_stored_image_uri", "metadata"):
        _ensure_object_column(out, column)
    fallback_format = _normalize_image_format(image_format)
    fsspec_options = dict(storage_options or {})

    for idx, row in out.iterrows():
        image_b64, image_source = _row_image_b64_with_source(row)
        stored_uri = row.get("_stored_image_uri")
        raw = _decode_image_b64(image_b64)
        inherited_page_uri = _is_inherited_page_uri(row, stored_uri, image_source=image_source, raw=raw)
        if raw is not None:
            source_path = _row_source_path(row)
            page_number = row.get("page_number")
            content_type = row.get("_content_type")
            if not isinstance(content_type, str) or not content_type.strip():
                content_type = image_source
            stored_uri = _write_image_b64(
                image_b64,
                storage_uri=storage_uri,
                storage_options=fsspec_options,
                fallback_format=fallback_format,
                source_path=source_path,
                page_number=page_number,
                content_type=content_type,
            )
            if stored_uri is not None:
                out.at[idx, "_stored_image_uri"] = stored_uri

                page_image = row.get("page_image")
                if isinstance(page_image, dict) and _row_image_represents_page(row, image_source=image_source):
                    updated_page_image = dict(page_image)
                    updated_page_image["stored_image_uri"] = stored_uri
                    if strip_base64:
                        updated_page_image["image_b64"] = None
                    out.at[idx, "page_image"] = updated_page_image

                if strip_base64 and image_source in row_image_columns and image_source in out.columns:
                    out.at[idx, image_source] = None

        if isinstance(stored_uri, str) and stored_uri.strip() and not inherited_page_uri and "metadata" in out.columns:
            out.at[idx, "metadata"] = _with_uploaded_image_uri(row.get("metadata"), stored_uri)

        for column in image_columns:
            if column not in out.columns:
                continue
            nested_type = {
                "page_image": "page",
                "images": "image",
                "tables": "table",
                "table": "table",
                "charts": "chart",
                "chart": "chart",
                "infographics": "infographic",
                "infographic": "infographic",
            }[column]
            out.at[idx, column] = _store_nested_image_payloads(
                out.at[idx, column],
                storage_uri=storage_uri,
                storage_options=fsspec_options,
                fallback_format=fallback_format,
                strip_base64=strip_base64,
                source_path=_row_source_path(row),
                page_number=row.get("page_number"),
                content_type=nested_type,
            )

    return out


class StoreOperator(AbstractOperator, CPUOperator):
    """Persist row-level image payloads to local or object storage.

    The operator consumes row-level ``_image_b64`` / ``image_b64`` payloads
    and writes ``_stored_image_uri`` for downstream vector DB upload. By
    default it clears inline base64 after successful writes to avoid carrying
    page-sized payloads into VDB upload.
    """

    def __init__(self, *, params: Any = None) -> None:
        super().__init__()
        self._params = params

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame):
            return data

        if hasattr(self._params, "model_dump"):
            store_kwargs = self._params.model_dump(mode="python")
        elif isinstance(self._params, dict):
            store_kwargs = self._params
        else:
            store_kwargs = {}
        return _store_row_images(
            data,
            storage_uri=store_kwargs.get("storage_uri", "stored_images"),
            storage_options=store_kwargs.get("storage_options") or {},
            image_format=store_kwargs.get("image_format", "png"),
            strip_base64=bool(store_kwargs.get("strip_base64", True)),
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
