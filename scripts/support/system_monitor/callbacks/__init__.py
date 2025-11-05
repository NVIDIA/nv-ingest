from typing import Any


def register_callbacks(app: Any, *, cy_available: bool) -> None:
    """Register all dashboard callbacks grouped by domain.

    This function aggregates domain callback registration to keep the main
    system_monitor.py body declarative.

    Parameters
    ----------
    app : Any
        Dash app instance.
    cy_available : bool
        Whether dash_cytoscape is available (for Process Tree graph callbacks).
    """
    from .overview import register_overview_callbacks
    from .proctree_impl import register_proctree_callbacks
    from .events import register_events_callbacks
    from .containers import register_containers_callbacks
    from .theme import register_theme_callbacks

    # Order can matter for readability; functional independence is maintained.
    register_overview_callbacks(app)
    register_proctree_callbacks(app, cy_available=cy_available)
    register_events_callbacks(app)
    register_containers_callbacks(app)
    register_theme_callbacks(app)
