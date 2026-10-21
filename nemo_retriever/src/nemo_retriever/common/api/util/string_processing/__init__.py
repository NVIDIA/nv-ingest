# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

DEPLOT_MAX_TOKENS = 128
DEPLOT_TEMPERATURE = 1.0
DEPLOT_TOP_P = 1.0

_V1_PATH_SEGMENT = re.compile(r"(?<!/)/v1(?=[/?#]|$)")


def remove_url_endpoints(url) -> str:
    """Some configurations provide the full endpoint in the URL.
    Ex: http://deplot:8000/v1/chat/completions. For hitting the
    health endpoint we need to get just the hostname:port combo
    that we can append the health/ready endpoint to so we attempt
    to parse that information here.

    Args:
        url str: Incoming URL

    Returns:
        str: URL with just the hostname:port portion remaining
    """
    # Match "/v1" only as a whole path segment, never inside the authority
    # (e.g. "http://v1-nim:8000"), where it would truncate the hostname.
    match = _V1_PATH_SEGMENT.search(url)
    if match:
        url = url[: match.start()]

    return url


def ensure_openai_embeddings_http_url(endpoint_url: str) -> str:
    """
    Return a concrete OpenAI-style embeddings POST URL (path ending in ``/embeddings``).

    Idempotent: if the path already ends with ``/embeddings`` (case-insensitive), the URL is
    returned unchanged aside from stripping a trailing slash on the path. This avoids
    ``.../embeddings`` + ``/embeddings`` doubling when callers pass a full integrate/NIM URL.
    """
    raw = (endpoint_url or "").strip()
    if not raw:
        raise ValueError("endpoint_url is empty")
    parts = urlsplit(raw)
    path = (parts.path or "").rstrip("/")
    if path.lower().endswith("/embeddings"):
        return raw.rstrip("/")
    new_path = f"{path}/embeddings" if path else "/embeddings"
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))


def prepend_model_provider_prefix(model_name: str | None, model_provider_prefix: str | None) -> str | None:
    """Prepend a LiteLLM provider prefix to a model identifier when configured."""
    if model_name is None:
        return None
    model = str(model_name).strip()
    prefix = str(model_provider_prefix or "").strip().strip("/")
    if not model or not prefix:
        return model
    return f"{prefix}/{model.lstrip('/')}"


def generate_url(url) -> str:
    """Examines the user defined URL for http*://. If that
    pattern is detected the URL is used as provided by the user.
    If that pattern does not exist then the assumption is made that
    the endpoint is simply `http://` and that is prepended
    to the user supplied endpoint.

    Args:
        url str: Endpoint where the Rest service is running

    Returns:
        str: Fully validated URL
    """
    if not re.match(r"^https?://", url):
        # Add the default `http://` if it's not already present in the URL
        url = f"http://{url}"

    return url
