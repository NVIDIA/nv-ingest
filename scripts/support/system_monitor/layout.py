# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: skip-file
# flake8: noqa

from typing import Any
from dash import dcc, html

try:
    import dash_cytoscape as cy  # type: ignore
except Exception:  # pragma: no cover
    cy = None


def build_layout(*, datafile: str, interval: int, cy_available: bool, cy_module: Any = None):
    """Build the System Monitor dashboard layout (sidebar + tabs + stores + interval).

    Parameters
    ----------
    datafile : str
        Default datafile path to display in the Output Parquet Path field and initial store.
    interval : int
        Refresh interval in seconds for the dcc.Interval component.
    cy_available : bool
        If True, render the Cytoscape process tree graph; otherwise show helper text.
    cy_module : Any, optional
        The dash_cytoscape module to use if available; if None, falls back to local import if present.
    """
    _cy = cy_module if cy_module is not None else cy

    return html.Div(
        [
            # Stores
            dcc.Store(id="datafile-store", data=datafile, storage_type="local"),
            dcc.Store(id="event-store", data=[], storage_type="local"),
            dcc.Store(id="event-auto-store", data=[], storage_type="local"),
            dcc.Store(id="watch-state-store", data={}, storage_type="session"),
            dcc.Store(id="proctree-last-summary", data={}, storage_type="memory"),
            dcc.Store(id="proctree-snapshot", data=None, storage_type="local"),
            dcc.Store(id="theme-sink", data=None, storage_type="memory"),
            # body-bg-sync is provided as a hidden Div below for clientside callback output
            html.Div(
                [
                    # Sidebar controls
                    html.Div(
                        [
                            html.H2("System Monitor", className="section-title"),
                            html.Div([html.Span(id="data-source", className="muted")]),
                            html.Div(
                                [
                                    html.Span("Last updated: ", className="muted"),
                                    html.Span("Never", id="last-updated", className="muted"),
                                ],
                                style={"marginBottom": "6px"},
                            ),
                            html.Details(
                                [
                                    html.Summary("Time & Display", className="section-title"),
                                    html.Div(
                                        [
                                            html.Div("Time range", className="label"),
                                            dcc.RadioItems(
                                                id="time-range",
                                                options=[
                                                    {"label": "10m", "value": 10},
                                                    {"label": "30m", "value": 30},
                                                    {"label": "1h", "value": 60},
                                                    {"label": "3h", "value": 180},
                                                    {"label": "All", "value": 0},
                                                ],
                                                value=30,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                                labelStyle={"display": "inline-block", "marginRight": "8px"},
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Display timezone", className="label"),
                                            dcc.Dropdown(
                                                id="display-tz",
                                                options=[
                                                    {"label": "Local", "value": "local"},
                                                    {"label": "UTC", "value": "utc"},
                                                    {"label": "Custom…", "value": "custom"},
                                                ],
                                                value="local",
                                                clearable=False,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="display-tz-custom",
                                                type="text",
                                                placeholder="e.g., America/Denver",
                                                debounce=True,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                            )
                                        ],
                                        id="display-tz-custom-wrap",
                                        className="control",
                                        style={"display": "none"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Data timezone (source)", className="label"),
                                            dcc.RadioItems(
                                                id="data-tz",
                                                options=[
                                                    {"label": "Local", "value": "local"},
                                                    {"label": "UTC", "value": "utc"},
                                                ],
                                                value="local",
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                                labelStyle={"display": "inline-block", "marginRight": "8px"},
                                            ),
                                            html.Div(
                                                "Tip: Set to UTC if you started the tracer with --utc.",
                                                className="muted",
                                                style={"marginTop": "4px"},
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Smoothing (samples)", className="label"),
                                            dcc.Slider(
                                                id="smoothing-window",
                                                min=1,
                                                max=10,
                                                step=1,
                                                value=3,
                                                marks={1: "1", 5: "5", 10: "10"},
                                                tooltip={"placement": "bottom", "always_visible": False},
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="pause-refresh",
                                                options=[{"label": "Pause auto-refresh", "value": "pause"}],
                                                value=[],
                                                inline=True,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                            )
                                        ],
                                        className="control",
                                    ),
                                ],
                                open=True,
                            ),
                            html.Details(
                                [
                                    html.Summary("Live Tracing", className="section-title"),
                                    html.Div(
                                        "Configure and control in-process tracing. For offline files, "
                                        "leave tracing stopped.",
                                        className="muted",
                                        style={"marginTop": "-6px", "marginBottom": "6px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Data source mode", className="label"),
                                            dcc.RadioItems(
                                                id="data-source-mode",
                                                options=[
                                                    {"label": "Auto (prefer live if running)", "value": "auto"},
                                                    {"label": "Live tracer", "value": "live"},
                                                    {"label": "File (Parquet/CSV)", "value": "file"},
                                                ],
                                                value="auto",
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                                labelStyle={"display": "block", "marginRight": "8px"},
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Output Parquet Path", className="label"),
                                            dcc.Input(
                                                id="tracer-output-path",
                                                type="text",
                                                value=datafile,
                                                debounce=True,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                                style={"width": "100%"},
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Sampling (s)", className="label"),
                                            dcc.Input(
                                                id="tracer-sample-interval",
                                                type="number",
                                                min=0.1,
                                                step=0.1,
                                                value=5.0,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Write Interval (s)", className="label"),
                                            dcc.Input(
                                                id="tracer-write-interval",
                                                type="number",
                                                min=1,
                                                step=1,
                                                value=10.0,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="tracer-options",
                                                options=[
                                                    {"label": "Enable GPU", "value": "gpu"},
                                                    {"label": "Enable Docker", "value": "docker"},
                                                    {"label": "UTC timestamps", "value": "utc"},
                                                ],
                                                value=["gpu", "docker"],
                                                inline=True,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                            )
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Start", id="tracer-start-btn", n_clicks=0, className="button primary"
                                            ),
                                            html.Button(
                                                "Stop",
                                                id="tracer-stop-btn",
                                                n_clicks=0,
                                                className="button",
                                                style={"marginLeft": "6px"},
                                            ),
                                            html.Button(
                                                "Reset Buffer",
                                                id="tracer-reset-btn",
                                                n_clicks=0,
                                                className="button",
                                                style={"marginLeft": "6px"},
                                            ),
                                            html.Button(
                                                "Snapshot Now",
                                                id="tracer-snapshot-btn",
                                                n_clicks=0,
                                                className="button primary",
                                                style={"marginLeft": "6px"},
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(id="tracer-status", className="muted", style={"marginTop": "6px"}),
                                ],
                                open=False,
                            ),
                            html.Hr(style={"opacity": 0.2}),
                            html.Details(
                                [
                                    html.Summary("Events & Annotations", className="section-title"),
                                    html.Div(
                                        "Add named events to appear as vertical markers on all charts.",
                                        className="muted",
                                        style={"marginTop": "-6px", "marginBottom": "6px"},
                                    ),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="event-name",
                                                type="text",
                                                placeholder="Event name",
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                                style={"width": "100%"},
                                            )
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Event time", className="label"),
                                            html.Div(
                                                [
                                                    dcc.DatePickerSingle(
                                                        id="event-date",
                                                        display_format="YYYY-MM-DD",
                                                        persistence=True,
                                                        persistence_type="local",
                                                    ),
                                                    dcc.Input(
                                                        id="event-time",
                                                        type="text",
                                                        placeholder="HH:MM[:SS]",
                                                        persistence=True,
                                                        persisted_props=["value"],
                                                        persistence_type="local",
                                                        style={"width": "120px", "marginLeft": "8px"},
                                                    ),
                                                ],
                                                style={"display": "flex", "alignItems": "center"},
                                            ),
                                            html.Div(
                                                "Date/time interpreted in selected Display Timezone; stored as UTC.",
                                                className="muted",
                                                style={"marginTop": "4px"},
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Add", id="add-event-btn", n_clicks=0, className="inline button primary"
                                            ),
                                            html.Button(
                                                "Add (Now)",
                                                id="add-event-now-btn",
                                                n_clicks=0,
                                                className="inline button primary",
                                            ),
                                            html.Button("Clear", id="clear-events-btn", n_clicks=0, className="inline"),
                                            dcc.Upload(
                                                id="event-upload",
                                                children=html.Div(["Import CSV"]),
                                                className="dccUpload inline",
                                            ),
                                        ],
                                        className="control",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="event-display-options",
                                                options=[{"label": "Show event markers", "value": "markers"}],
                                                value=["markers"],
                                                inline=True,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                            )
                                        ],
                                        className="control",
                                    ),
                                    html.Div(id="event-list", className="control"),
                                ],
                                open=False,
                            ),
                            html.Details(
                                [
                                    html.Summary(
                                        "Watch Points (auto-create events when thresholds are exceeded)",
                                        className="section-title",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="watch-enable",
                                                options=[
                                                    {"label": "CPU %", "value": "cpu"},
                                                    {"label": "Memory %", "value": "mem"},
                                                    {"label": "Threads", "value": "threads"},
                                                    {"label": "Processes", "value": "procs"},
                                                ],
                                                value=[],
                                                inline=True,
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("CPU % >"),
                                                    dcc.Input(
                                                        id="watch-cpu",
                                                        type="number",
                                                        min=0,
                                                        max=100,
                                                        step=1,
                                                        style={"width": "70px", "marginRight": "12px"},
                                                        persistence=True,
                                                        persisted_props=["value"],
                                                        persistence_type="local",
                                                    ),
                                                    html.Span("Mem % >"),
                                                    dcc.Input(
                                                        id="watch-mem",
                                                        type="number",
                                                        min=0,
                                                        max=100,
                                                        step=1,
                                                        style={"width": "70px", "marginRight": "12px"},
                                                        persistence=True,
                                                        persisted_props=["value"],
                                                        persistence_type="local",
                                                    ),
                                                    html.Span("Threads >"),
                                                    dcc.Input(
                                                        id="watch-threads",
                                                        type="number",
                                                        min=0,
                                                        step=1,
                                                        style={"width": "90px", "marginRight": "12px"},
                                                        persistence=True,
                                                        persisted_props=["value"],
                                                        persistence_type="local",
                                                    ),
                                                    html.Span("Processes >"),
                                                    dcc.Input(
                                                        id="watch-procs",
                                                        type="number",
                                                        min=0,
                                                        step=1,
                                                        style={"width": "90px", "marginRight": "12px"},
                                                        persistence=True,
                                                        persisted_props=["value"],
                                                        persistence_type="local",
                                                    ),
                                                ],
                                                style={"marginTop": "6px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "Clear Auto Events",
                                                        id="clear-auto-events-btn",
                                                        n_clicks=0,
                                                        className="inline",
                                                    )
                                                ],
                                                className="control",
                                            ),
                                        ]
                                    ),
                                ],
                                open=False,
                            ),
                            html.Hr(style={"opacity": 0.2}),
                            html.Details(
                                [
                                    html.Summary("Appearance", className="section-title"),
                                    html.Div(
                                        [
                                            html.Div("Theme", className="label"),
                                            dcc.RadioItems(
                                                id="theme-toggle",
                                                options=[
                                                    {"label": "Light", "value": "light"},
                                                    {"label": "Dark", "value": "dark"},
                                                ],
                                                value="dark",
                                                persistence=True,
                                                persisted_props=["value"],
                                                persistence_type="local",
                                                labelStyle={"display": "inline-block", "marginRight": "8px"},
                                            ),
                                        ],
                                        className="control",
                                    ),
                                ],
                                open=False,
                            ),
                        ],
                        className="sidebar",
                    ),
                    # Main content
                    html.Div(
                        [
                            # Contextual notice banner (filled by callback)
                            html.Div(
                                id="notice-banner",
                                style={
                                    "marginBottom": "10px",
                                    "border": "1px solid var(--border)",
                                    "padding": "8px",
                                    "display": "block",
                                },
                            ),
                            # Tabs
                            dcc.Tabs(
                                id="main-tabs",
                                value="tab-overview",
                                colors={
                                    "border": "var(--border)",
                                    "primary": "var(--text)",
                                    "background": "var(--surface)",
                                },
                                children=[
                                    dcc.Tab(
                                        label="Overview",
                                        value="tab-overview",
                                        children=[
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.H2("System Overview", className="section-title"),
                                                            html.Div(id="kpi-row", className="kpis"),
                                                            dcc.Graph(
                                                                id="system-overview-graph",
                                                                className="graph",
                                                                style={"height": "340px"},
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={"marginTop": "4px"},
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Process Tree",
                                        value="tab-proctree",
                                        children=[
                                            html.Div(
                                                [
                                                    html.H2("Process Tree Inspector", className="section-title"),
                                                    dcc.Loading(
                                                        id="proctree-loading-controls",
                                                        type="dot",
                                                        children=html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span("Root PID:"),
                                                                        dcc.Input(
                                                                            id="proctree-pid",
                                                                            type="number",
                                                                            placeholder="Enter PID",
                                                                            debounce=True,
                                                                            style={
                                                                                "width": "140px",
                                                                                "marginLeft": "6px",
                                                                                "marginRight": "12px",
                                                                            },
                                                                            persistence=True,
                                                                            persisted_props=["value"],
                                                                            persistence_type="local",
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="proctree-verbose",
                                                                            options=[
                                                                                {"label": "Verbose", "value": "verbose"}
                                                                            ],
                                                                            value=[],
                                                                            inline=True,
                                                                            persistence=True,
                                                                            persisted_props=["value"],
                                                                            persistence_type="local",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "marginRight": "12px",
                                                                            },
                                                                        ),
                                                                        html.Span("Find PID:"),
                                                                        dcc.Input(
                                                                            id="proctree-search",
                                                                            type="text",
                                                                            placeholder="e.g. "
                                                                            "microservice_entrypoint.py",
                                                                            debounce=False,
                                                                            style={
                                                                                "width": "320px",
                                                                                "marginLeft": "6px",
                                                                                "marginRight": "8px",
                                                                            },
                                                                            persistence=True,
                                                                            persisted_props=["value"],
                                                                            persistence_type="local",
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="proctree-suggestions",
                                                                            options=[],
                                                                            placeholder="Select a matching PID",
                                                                            style={
                                                                                "width": "420px",
                                                                                "display": "inline-block",
                                                                            },
                                                                            clearable=True,
                                                                        ),
                                                                        html.Button(
                                                                            "Inspect",
                                                                            id="proctree-run",
                                                                            n_clicks=0,
                                                                            className="button primary",
                                                                            style={"marginLeft": "12px"},
                                                                        ),
                                                                        html.Div(
                                                                            "Type to search; "
                                                                            "select a row to populate Root PID.",
                                                                            className="muted",
                                                                            style={"marginTop": "6px"},
                                                                        ),
                                                                        html.Div(
                                                                            [
                                                                                html.Span(
                                                                                    "Tree view:",
                                                                                    style={"marginRight": "8px"},
                                                                                ),
                                                                                dcc.RadioItems(
                                                                                    id="proctree-view-mode",
                                                                                    options=[
                                                                                        {
                                                                                            "label": "Text",
                                                                                            "value": "text",
                                                                                        },
                                                                                        {
                                                                                            "label": "Graph",
                                                                                            "value": "graph",
                                                                                        },
                                                                                    ],
                                                                                    value="text",
                                                                                    labelStyle={
                                                                                        "display": "inline-block",
                                                                                        "marginRight": "8px",
                                                                                    },
                                                                                    persistence=True,
                                                                                    persisted_props=["value"],
                                                                                    persistence_type="local",
                                                                                ),
                                                                                html.Button(
                                                                                    "Take Snapshot",
                                                                                    id="proctree-snapshot-btn",
                                                                                    n_clicks=0,
                                                                                    className="button",
                                                                                    style={"marginLeft": "12px"},
                                                                                ),
                                                                                html.Button(
                                                                                    "Compare to Snapshot",
                                                                                    id="proctree-diff-btn",
                                                                                    n_clicks=0,
                                                                                    className="button",
                                                                                    style={"marginLeft": "8px"},
                                                                                ),
                                                                            ],
                                                                            style={"marginTop": "8px"},
                                                                        ),
                                                                    ],
                                                                    className="control",
                                                                ),
                                                            ],
                                                            className="control",
                                                        ),
                                                    ),
                                                    dcc.Loading(
                                                        id="proctree-loading-totals",
                                                        type="dot",
                                                        children=html.Div(
                                                            id="proctree-totals",
                                                            className="muted",
                                                            style={"marginBottom": "8px"},
                                                        ),
                                                    ),
                                                    dcc.Loading(
                                                        id="proctree-loading-graphs",
                                                        type="default",
                                                        style={"width": "100%"},
                                                        children=html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        dcc.Graph(
                                                                            id="proctree-procs-by-cmd",
                                                                            className="graph",
                                                                            style={"height": "260px", "width": "100%"},
                                                                            config={"responsive": True},
                                                                        )
                                                                    ],
                                                                    style={
                                                                        "flex": "1 1 0",
                                                                        "minWidth": "0",
                                                                        "boxSizing": "border-box",
                                                                        "overflow": "hidden",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dcc.Graph(
                                                                            id="proctree-threads-by-cmd",
                                                                            className="graph",
                                                                            style={"height": "260px", "width": "100%"},
                                                                            config={"responsive": True},
                                                                        )
                                                                    ],
                                                                    style={
                                                                        "flex": "1 1 0",
                                                                        "minWidth": "0",
                                                                        "boxSizing": "border-box",
                                                                        "overflow": "hidden",
                                                                    },
                                                                ),
                                                            ],
                                                            style={
                                                                "display": "flex",
                                                                "gap": "12px",
                                                                "alignItems": "stretch",
                                                                "width": "100%",
                                                            },
                                                        ),
                                                    ),
                                                    dcc.Loading(
                                                        id="proctree-loading-tree",
                                                        type="cube",
                                                        style={"width": "100%"},
                                                        children=html.Div(
                                                            [
                                                                html.H3("Tree", className="section-title"),
                                                                html.Div(
                                                                    [
                                                                        (
                                                                            _cy.Cytoscape(
                                                                                id="proctree-graph",
                                                                                elements=[],
                                                                                layout={
                                                                                    "name": "breadthfirst",
                                                                                    "directed": True,
                                                                                },
                                                                                style={
                                                                                    "width": "100%",
                                                                                    "height": "520px",
                                                                                    "border": "1px solid #444",
                                                                                    "boxSizing": "border-box",
                                                                                    "maxWidth": "100%",
                                                                                    "overflow": "hidden",
                                                                                },
                                                                                stylesheet=[],
                                                                            )
                                                                            if cy_available and _cy is not None
                                                                            else html.Div(
                                                                                "Graph view requires "
                                                                                "dash-cytoscape. Install with: "
                                                                                "pip install dash-cytoscape",
                                                                                className="muted",
                                                                                style={"padding": "8px"},
                                                                            )
                                                                        )
                                                                    ],
                                                                    id="proctree-cyto-container",
                                                                    style={
                                                                        "display": "none",
                                                                        "width": "100%",
                                                                        "maxWidth": "100%",
                                                                        "overflowX": "hidden",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    id="proctree-graph-status",
                                                                    className="muted",
                                                                    style={"marginTop": "6px"},
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dcc.Markdown(
                                                                            id="proctree-tree-md",
                                                                            style={"whiteSpace": "pre-wrap"},
                                                                        )
                                                                    ],
                                                                    id="proctree-tree-text-container",
                                                                    style={"display": "block"},
                                                                ),
                                                                html.Div(
                                                                    id="proctree-node-details",
                                                                    className="muted",
                                                                    style={"marginTop": "8px"},
                                                                ),
                                                                html.Div(
                                                                    id="proctree-snapshot-status",
                                                                    className="muted",
                                                                    style={"marginTop": "8px"},
                                                                ),
                                                                html.Div(
                                                                    id="proctree-diff-result",
                                                                    className="muted",
                                                                    style={"marginTop": "8px"},
                                                                ),
                                                            ]
                                                        ),
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="CPU & Memory",
                                        value="tab-cpu-mem",
                                        children=[
                                            html.Div(
                                                [
                                                    html.H2("CPU Utilization", className="section-title"),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="cpu-aggregated-utilization-graph",
                                                                        className="graph",
                                                                        style={"height": "320px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="cpu-individual-utilization-graph",
                                                                        className="graph",
                                                                        style={"height": "320px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H2("Memory Usage", className="section-title"),
                                                    dcc.Graph(
                                                        id="memory-graph", className="graph", style={"height": "320px"}
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H2("File Descriptor Usage", className="section-title"),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="file-count-graph",
                                                                        className="graph",
                                                                        style={"height": "280px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="fd-usage-graph",
                                                                        className="graph",
                                                                        style={"height": "280px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H2("Processes and Threads", className="section-title"),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="process-count-graph",
                                                                        className="graph",
                                                                        style={"height": "260px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="thread-count-graph",
                                                                        className="graph",
                                                                        style={"height": "260px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="I/O & Network",
                                        value="tab-io-net",
                                        children=[
                                            html.Div(
                                                [
                                                    html.H2("Network Activity", className="section-title"),
                                                    dcc.Graph(
                                                        id="network-graph", className="graph", style={"height": "320px"}
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H2("Disk I/O", className="section-title"),
                                                    dcc.Graph(
                                                        id="disk-io-graph", className="graph", style={"height": "320px"}
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="GPU",
                                        value="tab-gpu",
                                        children=[
                                            html.Div(
                                                [
                                                    html.H2("GPU Usage", className="section-title"),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="gpu-utilization-graph",
                                                                        className="graph",
                                                                        style={"height": "300px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="gpu-memory-graph",
                                                                        className="graph",
                                                                        style={"height": "300px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Containers",
                                        value="tab-containers",
                                        children=[
                                            html.Div(
                                                [
                                                    html.Details(
                                                        [
                                                            html.Summary("Container Focus"),
                                                            html.Div(
                                                                [
                                                                    dcc.Checklist(
                                                                        id="container-auto-top",
                                                                        options=[
                                                                            {
                                                                                "label": "Auto select top by CPU",
                                                                                "value": "auto",
                                                                            }
                                                                        ],
                                                                        value=["auto"],
                                                                        persistence=True,
                                                                        persisted_props=["value"],
                                                                        persistence_type="local",
                                                                        labelStyle={
                                                                            "display": "inline-block",
                                                                            "marginRight": "10px",
                                                                        },
                                                                    ),
                                                                    html.Label("Top N (when auto):"),
                                                                    dcc.Slider(
                                                                        id="container-top-n",
                                                                        min=1,
                                                                        max=10,
                                                                        step=1,
                                                                        value=5,
                                                                        marks={1: "1", 5: "5", 10: "10"},
                                                                        tooltip={
                                                                            "placement": "bottom",
                                                                            "always_visible": False,
                                                                        },
                                                                    ),
                                                                    html.Label("Or select containers:"),
                                                                    dcc.Dropdown(
                                                                        id="container-select",
                                                                        options=[],
                                                                        value=[],
                                                                        multi=True,
                                                                        placeholder="Select containers...",
                                                                        persistence=True,
                                                                        persisted_props=["value"],
                                                                        persistence_type="local",
                                                                    ),
                                                                ],
                                                                style={"margin": "8px 0"},
                                                            ),
                                                        ],
                                                        open=False,
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.H2("Container Metrics", className="section-title"),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="container-cpu-utilization-graph",
                                                                        className="graph",
                                                                        style={"height": "280px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "33%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="container-memory-utilization-graph",
                                                                        className="graph",
                                                                        style={"height": "280px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "33%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="container-files-graph",
                                                                        className="graph",
                                                                        style={"height": "280px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "33%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H2(
                                                        "Container Network Throughput (MB/s)", className="section-title"
                                                    ),
                                                    dcc.Graph(
                                                        id="container-net-graph",
                                                        className="graph",
                                                        style={"height": "300px"},
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H2("Container Disk I/O (MB/s)", className="section-title"),
                                                    dcc.Graph(
                                                        id="container-io-graph",
                                                        className="graph",
                                                        style={"height": "300px"},
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H2(
                                                        "Top Containers (Latest Sample)", className="section-title"
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="container-top-cpu-graph",
                                                                        className="graph",
                                                                        style={"height": "280px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="container-top-mem-graph",
                                                                        className="graph",
                                                                        style={"height": "280px"},
                                                                    )
                                                                ],
                                                                style={
                                                                    "width": "50%",
                                                                    "display": "inline-block",
                                                                    "verticalAlign": "top",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={"width": "100%", "marginBottom": "16px"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ]
                    ),
                ],
                className="grid",
            ),
            # Refresh interval
            dcc.Interval(
                id="interval-component", interval=interval * 1000, n_intervals=0  # convert seconds to milliseconds
            ),
            # Hidden div to sync body background with theme (already declared store above)
            html.Div(id="body-bg-sync", style={"display": "none"}),
        ],
        id="page-container",
        style={"padding": "16px"},
    )
