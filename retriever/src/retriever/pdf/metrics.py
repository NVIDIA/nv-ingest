"""Backwards-compatible shim for metrics imports.

New stages should import from `retriever.metrics`. This module remains so existing
imports like `retriever.pdf.metrics` keep working.
"""

from retriever.metrics import LoggingMetrics, Metrics, NullMetrics, safe_metrics

