"""
Recall evaluation utilities and CLI.
"""

from .__main__ import app
from .core import RecallConfig, evaluate_recall

__all__ = [
    "app",
    "RecallConfig",
    "evaluate_recall",
]
