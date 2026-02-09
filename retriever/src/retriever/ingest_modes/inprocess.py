"""
In-process runmode.

Intended to run locally in a plain Python process with minimal assumptions
about surrounding framework/runtime.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..ingest import Ingestor


class InProcessIngestor(Ingestor):
    RUN_MODE = "inprocess"

    def __init__(self, documents: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(documents=documents, **kwargs)

