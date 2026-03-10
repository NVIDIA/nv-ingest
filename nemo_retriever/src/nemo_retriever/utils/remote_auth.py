from __future__ import annotations

import os
from typing import Optional


def resolve_remote_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    """Resolve bearer token for hosted NIM endpoints."""
    token = explicit_api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY")
    token = (token or "").strip()
    return token or None
