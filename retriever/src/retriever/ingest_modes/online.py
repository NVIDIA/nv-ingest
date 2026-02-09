"""
Online runmode.

Intended for low-latency request/response serving with concurrent requests.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..ingest import Ingestor


class OnlineIngestor(Ingestor):
    RUN_MODE = "online"

    def __init__(self, documents: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(documents=documents, **kwargs)

