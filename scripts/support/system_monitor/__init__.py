"""System Monitor package.

Provides the canonical entry point for the dashboard and tracer.

Usage:
    python -m system_monitor --datafile system_monitor.parquet
"""

from .system_monitor import run_dashboard

__all__ = ["run_dashboard"]
