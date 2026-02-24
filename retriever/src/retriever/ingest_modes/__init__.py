"""
Ingestion runmodes.

Each runmode provides a concrete `Ingestor` implementation with the same public
method surface defined by `retriever.ingest.Ingestor`.
"""

from __future__ import annotations

from .batch import BatchIngestor
from .fused import FusedIngestor
from .inprocess import InProcessIngestor
from .online import OnlineIngestor

__all__ = [
    "BatchIngestor",
    "FusedIngestor",
    "InProcessIngestor",
    "OnlineIngestor",
]
