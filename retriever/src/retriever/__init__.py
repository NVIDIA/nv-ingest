"""Retriever application package."""

from .ingest import create_ingestor
from .version import __version__, get_version, get_version_info

__all__ = ["__version__", "create_ingestor", "get_version", "get_version_info"]
