from __future__ import annotations

from typing import Dict, Optional


def normalize_endpoint(endpoint: Optional[str]) -> str:
    """
    Best-effort endpoint normalization for optional remote invocation.

    - Adds `http://` if scheme is missing
    - Strips trailing slashes
    """
    ep = str(endpoint or "").strip()
    if not ep:
        return ep
    if "://" not in ep:
        ep = "http://" + ep
    return ep.rstrip("/")


def default_headers(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Default headers for JSON HTTP endpoints, with optional overrides.
    """
    out: Dict[str, str] = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if headers:
        out.update({str(k): str(v) for k, v in dict(headers).items()})
    return out

