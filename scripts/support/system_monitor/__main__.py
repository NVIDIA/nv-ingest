# flake8: noqa
# noqa
"""
Module entry point for the System Monitor dashboard.

Run with:
    python -m system_monitor --datafile system_monitor.parquet
"""
import os

_HERE = os.path.abspath(os.path.dirname(__file__))
_PKG_ASSETS = os.path.join(_HERE, "assets")
if os.path.isdir(_PKG_ASSETS) and not os.environ.get("SYSTEM_MONITOR_ASSETS"):
    os.environ["SYSTEM_MONITOR_ASSETS"] = _PKG_ASSETS

from .system_monitor import run_dashboard

if __name__ == "__main__":
    run_dashboard()
