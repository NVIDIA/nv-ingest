"""Stable import surface for tracer utilities.

Usage:
    from system_monitor.tracer import (
        get_process_tree_summary,
    )
"""

# flake8: noqa

if __name__ == "__main__":
    import sys

    # Delegate to the full CLI in system_tracer
    from .system_tracer import main as tracer_main

    sys.exit(tracer_main())

"""Lightweight wrapper to expose tracer CLI under a short module path."""

from .system_tracer import (
    get_process_tree_summary,
)

__all__ = ["get_process_tree_summary"]
