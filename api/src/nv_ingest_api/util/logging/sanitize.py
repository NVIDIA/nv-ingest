# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence, Set

try:
    # Pydantic is optional at runtime for this helper; import if available
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover - pydantic always present in this repo
    BaseModel = None  # type: ignore


_DEFAULT_SENSITIVE_KEYS: Set[str] = {
    "api_key",
    "auth_token",
    "password",
    "secret",
    "client_secret",
    "refresh_token",
    "access_token",
    "authorization",
    "x-api-key",
    "ssl_cert",
}

_REDACTION = "***REDACTED***"


def _is_mapping(obj: Any) -> bool:
    try:
        return isinstance(obj, Mapping)
    except Exception:
        return False


def _is_sequence(obj: Any) -> bool:
    # Exclude strings/bytes from sequences we want to traverse
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def sanitize_for_logging(
    data: Any,
    sensitive_keys: Set[str] | None = None,
    redaction: str = _REDACTION,
) -> Any:
    """
    Recursively sanitize common secret fields from dicts, lists, tuples, and Pydantic models.

    - Key comparison is case-insensitive and matches exact keys only.
    - Does not mutate input; returns a sanitized deep copy.
    - For Pydantic BaseModel instances, uses model_dump() before redaction.
    """
    keys = {k.lower() for k in (sensitive_keys or _DEFAULT_SENSITIVE_KEYS)}

    # Handle Pydantic models without importing pydantic at module import time
    if BaseModel is not None and isinstance(data, BaseModel):  # type: ignore[arg-type]
        try:
            return sanitize_for_logging(data.model_dump(), keys, redaction)
        except Exception:
            # Fall through and try generic handling below
            pass

    # Dict-like
    if _is_mapping(data):
        out: MutableMapping[str, Any] = type(data)()  # preserve mapping type where possible
        for k, v in data.items():  # type: ignore[assignment]
            key_lower = str(k).lower()
            if key_lower in keys:
                out[k] = redaction
            else:
                out[k] = sanitize_for_logging(v, keys, redaction)
        return out

    # List/Tuple/Sequence
    if _is_sequence(data):
        return type(data)(sanitize_for_logging(v, keys, redaction) for v in data)

    # Fallback: return as-is
    return data
