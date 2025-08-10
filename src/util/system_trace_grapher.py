import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import click
import base64
import io
import csv
from datetime import datetime

try:
    from dateutil.tz import tzlocal, gettz
except Exception:
    tzlocal = None
    gettz = None

# Ensure pyarrow is available for parquet file reading
try:
    import pyarrow.parquet as pq  # noqa

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Warning: pyarrow is not available. Please install pyarrow for Parquet file support.")


@click.command()
@click.option("--datafile", "-d", default="system_monitor.parquet", help="Path to the parquet data file")
@click.option("--port", "-p", default=8050, help="Port to run the dashboard server on")
@click.option("--host", "-h", default="0.0.0.0", help="Host to run the dashboard server on")
@click.option("--interval", "-i", default=10, help="Refresh interval in seconds")
@click.option("--debug/--no-debug", default=True, help="Run in debug mode")
def run_dashboard(datafile, port, host, interval, debug):
    """Run the system monitoring dashboard with the specified parameters."""

    # Validate the data file
    if not os.path.exists(datafile):
        print(f"Error: Data file '{datafile}' not found.")
        print("Dashboard will start but won't display data until the file exists.")
    elif datafile.endswith(".parquet") and not PARQUET_AVAILABLE:
        print("Error: Parquet file specified but pyarrow is not available.")
        print("Please install pyarrow to read parquet files.")
        sys.exit(1)
    elif not datafile.endswith(".parquet") and not datafile.endswith(".csv"):
        print(f"Warning: Data file '{datafile}' is not a .parquet or .csv file.")
        print("Attempting to load it anyway, but this may cause errors.")

    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Make the full window background dark by default and remove body margins.
    # (The page container still adapts to theme toggle dynamically.)
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                :root {
                    --bg-dark: #111111;
                    --bg-light: #ffffff;
                    --fg-dark: #e5e5e5;
                    --fg-light: #222222;
                    --muted: #888888;
                }
                html, body { height: 100%; margin: 0; }
                body { background-color: var(--bg-dark); color: var(--fg-dark); font-family: system-ui, -apple-system,
                Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.45; }
                .grid { display: grid; grid-template-columns: 280px 1fr; gap: 16px; align-items: start; }
                .sidebar { position: sticky; top: 12px; align-self: start; padding: 12px; border-radius: 6px;
                border: 1px solid rgba(127,127,127,0.2); }
                .section-title { font-size: 18px; margin: 8px 0; font-weight: 600; }
                .muted { color: var(--muted); font-size: 13px; }
                .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                 gap: 8px; margin: 8px 0 12px; }
                .graph { margin-bottom: 12px; }
                .control { margin: 6px 0; }
                .inline { display: inline-block; margin-right: 8px; vertical-align: middle; }
                details > summary { cursor: pointer; }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """

    # Set the app title
    app.title = "System Resource Monitor"

    # Define the layout with additional graphs for new metrics
    app.layout = html.Div(
        [
            # Stores
            dcc.Store(id="event-store", data=[], storage_type="local"),
            dcc.Store(id="event-auto-store", data=[], storage_type="local"),
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
                            html.Div(
                                [
                                    html.Div("Time range", className="muted"),
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
                                    html.Div("Theme", className="muted"),
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
                            html.Div(
                                [
                                    html.Div("Smoothing (samples)", className="muted"),
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
                            html.Hr(style={"opacity": 0.2}),
                            html.Div(
                                [
                                    html.Div("Display timezone", className="muted"),
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
                                className="control",
                            ),
                            html.Hr(style={"opacity": 0.2}),
                            html.Div([html.Div("Events", className="section-title")]),
                            html.Div(
                                [
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
                                            dcc.Input(
                                                id="event-datetime-text",
                                                type="text",
                                                placeholder="YYYY-MM-DD HH:MM[:SS]",
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
                                            html.Button("Add", id="add-event-btn", n_clicks=0, className="inline"),
                                            html.Button(
                                                "Add (Now)", id="add-event-now-btn", n_clicks=0, className="inline"
                                            ),
                                            html.Button("Clear", id="clear-events-btn", n_clicks=0, className="inline"),
                                            dcc.Upload(
                                                id="event-upload",
                                                children=html.Div(["Import CSV"]),
                                                style={
                                                    "display": "inline-block",
                                                    "padding": "6px 10px",
                                                    "border": "1px dashed #999",
                                                    "marginLeft": "8px",
                                                },
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
                                ]
                            ),
                        ],
                        className="sidebar",
                    ),
                    # Main content
                    html.Div(
                        [
                            dcc.Tabs(
                                id="main-tabs",
                                value="tab-overview",
                                colors={"border": "#333333", "primary": "#333333", "background": "#111111"},
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
                                                    html.Details(
                                                        open=False,
                                                        children=[
                                                            html.Summary(
                                                                "Watch Points (auto-create events when "
                                                                "thresholds are exceeded)"
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
                                                                                style={
                                                                                    "width": "70px",
                                                                                    "marginRight": "12px",
                                                                                },
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
                                                                                style={
                                                                                    "width": "70px",
                                                                                    "marginRight": "12px",
                                                                                },
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
                                                                                style={
                                                                                    "width": "90px",
                                                                                    "marginRight": "12px",
                                                                                },
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
                                                                                style={
                                                                                    "width": "90px",
                                                                                    "marginRight": "12px",
                                                                                },
                                                                                persistence=True,
                                                                                persisted_props=["value"],
                                                                                persistence_type="local",
                                                                            ),
                                                                        ],
                                                                        style={"marginTop": "6px"},
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                                style={"marginTop": "4px"},
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
                                style={"width": "100%"},
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

    # Helper function to load and filter data
    def load_data(time_range_minutes):
        try:
            if os.path.exists(datafile):
                if datafile.endswith(".parquet") and PARQUET_AVAILABLE:
                    # Read parquet file
                    df = pd.read_parquet(datafile)
                elif datafile.endswith(".csv"):
                    # Read CSV file
                    df = pd.read_csv(datafile, parse_dates=["timestamp"])
                else:
                    # Try to guess the format
                    try:
                        if PARQUET_AVAILABLE:
                            df = pd.read_parquet(datafile)
                        else:
                            df = pd.read_csv(datafile, parse_dates=["timestamp"])
                    except Exception as e:
                        print(f"Error reading data file: {e}")
                        return pd.DataFrame()

                # Filter by time range if specified
                if time_range_minutes > 0 and not df.empty:
                    latest_time = df["timestamp"].max()
                    time_threshold = latest_time - pd.Timedelta(minutes=time_range_minutes)
                    df = df[df["timestamp"] >= time_threshold]

                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    # ----------------------------
    # Helper utilities (theme, events, small figure helpers)
    # ----------------------------
    def normalize_ts(ts_any):
        try:
            ts = pd.to_datetime(ts_any)
            # drop tz if present
            if getattr(ts, "tzinfo", None) is not None:
                try:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                except Exception:
                    try:
                        ts = ts.tz_localize("UTC").tz_localize(None)
                    except Exception:
                        pass
            return ts
        except Exception:
            return pd.Timestamp.utcnow()

    def apply_theme(fig, theme_value):
        try:
            if theme_value == "dark":
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="#111111", plot_bgcolor="#111111", font=dict(color="#e5e5e5")
                )
            else:
                fig.update_layout(
                    template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#222222")
                )
        except Exception:
            pass
        return fig

    def style_minimal_figure(fig, theme_value):
        """Apply a Tufte-inspired minimal style to reduce chartjunk and emphasize data.

        - Remove heavy titles, rely on surrounding headings
        - Thin margins and light gridlines
        - Hide zero lines and reduce axis clutter
        - Use unified hover for easier comparison across series
        - Do not override legend visibility here; caller decides.
        """
        try:
            # Remove figure title; surrounding layout already provides section headers
            fig.update_layout(title=None)

            # Compact margins and legend
            fig.update_layout(
                margin=dict(l=40, r=10, t=10, b=28),
                hovermode="x unified",
            )

            # Light, subtle grids; no axis zero lines
            grid_color = "rgba(127,127,127,0.15)" if theme_value == "dark" else "rgba(0,0,0,0.08)"
            axis_color = "rgba(127,127,127,0.4)" if theme_value == "dark" else "rgba(0,0,0,0.35)"
            fig.update_xaxes(
                showgrid=True,
                gridcolor=grid_color,
                zeroline=False,
                linecolor=axis_color,
                ticks="outside",
                tickcolor=axis_color,
                ticklen=4,
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor=grid_color,
                zeroline=False,
                linecolor=axis_color,
                ticks="outside",
                tickcolor=axis_color,
                ticklen=4,
            )

            # Thin lines for traces to keep focus on shape; avoid overly thick strokes
            try:
                for tr in fig.data:
                    if hasattr(tr, "line") and getattr(tr, "line", None) is not None:
                        # keep bar width as-is; only adjust scatter-like traces
                        if getattr(tr, "type", "").lower() in ("scatter", "scattergl", "lines"):
                            lw = getattr(tr.line, "width", None)
                            new_w = 1.25 if lw is None else min(lw, 1.5)
                            tr.line.width = new_w
            except Exception:
                pass

        except Exception:
            # Never let styling break figure rendering
            pass
        return fig

    def make_sparkline(ts, series, theme_value):
        """Create a tiny sparkline figure for KPI cards.

        Expects ts (Datetime-like Series) and series (numeric Series) of same length.
        """
        fig = go.Figure()
        try:
            fig.add_trace(go.Scatter(x=ts, y=series, mode="lines", name="", hoverinfo="skip"))
        except Exception:
            # fallback empty
            pass
        # Minimal styling
        apply_theme(fig, theme_value)
        style_minimal_figure(fig, theme_value)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    # ----------------------------
    # Strategy pattern: Graph components and registry
    # ----------------------------
    class GraphContext:
        def __init__(self, smoothing_window, apply_theme_fn, smooth_series_fn):
            self.smoothing_window = smoothing_window
            self.apply_theme = apply_theme_fn
            self.smooth_series = smooth_series_fn

    class GraphComponent:
        component_id = ""
        title = None
        is_time_series = True  # whether event markers should be applied

        def build(self, df: pd.DataFrame, ts: pd.Series, ctx: GraphContext, params: dict) -> go.Figure:
            raise NotImplementedError

    class CPUIndividualGraph(GraphComponent):
        component_id = "cpu-individual-utilization-graph"
        title = "CPU Utilization (per core)"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            cpu_cols = [c for c in df.columns if c.startswith("cpu_") and c.endswith("_utilization")]
            for c in sorted(cpu_cols):
                fig.add_trace(
                    go.Scatter(x=ts, y=ctx.smooth_series(df[c]), mode="lines", name=c.replace("_utilization", ""))
                )
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="%")
            return fig

    class CPUAggregateGraph(GraphComponent):
        component_id = "cpu-aggregated-utilization-graph"
        title = "CPU Utilization (aggregate)"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            cpu_cols = [c for c in df.columns if c.startswith("cpu_") and c.endswith("_utilization")]
            if cpu_cols:
                cpu_mean = ctx.smooth_series(df[cpu_cols].mean(axis=1))
                fig.add_trace(go.Scatter(x=ts, y=cpu_mean, mode="lines", name="CPU %"))
                fig.update_layout(title=self.title, yaxis_title="%")
            return fig

    class MemoryGraph(GraphComponent):
        component_id = "memory-graph"
        title = "Memory Utilization"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            if {"sys_used", "sys_total"}.issubset(df.columns):
                mem_pct = (df["sys_used"] / df["sys_total"] * 100.0).clip(lower=0, upper=100)
                fig.add_trace(go.Scatter(x=ts, y=ctx.smooth_series(mem_pct), mode="lines", name="Mem %"))
                fig.update_layout(title=self.title, yaxis_title="%")
            return fig

    class FileCountGraph(GraphComponent):
        component_id = "file-count-graph"
        title = "Total Open Files"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            if "total_open_files" in df.columns:
                fig.add_trace(
                    go.Scatter(x=ts, y=ctx.smooth_series(df["total_open_files"]), mode="lines", name="Open Files")
                )
                fig.update_layout(title=self.title)
            return fig

    class FDUsageGraph(GraphComponent):
        component_id = "fd-usage-graph"
        title = "FD Usage %"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            if "fd_usage_percent" in df.columns:
                fig.add_trace(go.Scatter(x=ts, y=ctx.smooth_series(df["fd_usage_percent"]), mode="lines", name="FD %"))
                fig.update_layout(title=self.title, yaxis_title="%")
            return fig

    class NetworkGraph(GraphComponent):
        component_id = "network-graph"
        title = "Network Throughput"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            recv_col = "net_bytes_recv_per_sec" if "net_bytes_recv_per_sec" in df.columns else None
            sent_col = "net_bytes_sent_per_sec" if "net_bytes_sent_per_sec" in df.columns else None
            if recv_col:
                fig.add_trace(
                    go.Scatter(x=ts, y=ctx.smooth_series(df[recv_col]) / (1024**2), mode="lines", name="Down MB/s")
                )
            if sent_col:
                fig.add_trace(
                    go.Scatter(x=ts, y=ctx.smooth_series(df[sent_col]) / (1024**2), mode="lines", name="Up MB/s")
                )
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="MB/s")
            return fig

    class DiskIOGraph(GraphComponent):
        component_id = "disk-io-graph"
        title = "Disk I/O"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            r_col = "disk_read_bytes_per_sec" if "disk_read_bytes_per_sec" in df.columns else None
            w_col = "disk_write_bytes_per_sec" if "disk_write_bytes_per_sec" in df.columns else None
            if r_col:
                fig.add_trace(
                    go.Scatter(x=ts, y=ctx.smooth_series(df[r_col]) / (1024**2), mode="lines", name="Read MB/s")
                )
            if w_col:
                fig.add_trace(
                    go.Scatter(x=ts, y=ctx.smooth_series(df[w_col]) / (1024**2), mode="lines", name="Write MB/s")
                )
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="MB/s")
            return fig

    class GPUUtilGraph(GraphComponent):
        component_id = "gpu-utilization-graph"
        title = "GPU Utilization %"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            util_cols = [c for c in df.columns if c.endswith("_utilization") and c.startswith("gpu_")]
            for c in sorted(util_cols):
                fig.add_trace(
                    go.Scatter(x=ts, y=ctx.smooth_series(df[c]), mode="lines", name=c.replace("_utilization", " util"))
                )
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="%")
            return fig

    class GPUMemoryGraph(GraphComponent):
        component_id = "gpu-memory-graph"
        title = "GPU Memory %"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            gpu_mem_used = [c for c in df.columns if c.startswith("gpu_") and c.endswith("_used")]
            for c in sorted(gpu_mem_used):
                idx = c.split("_")[1]
                tot_col = f"gpu_{idx}_total"
                if tot_col in df.columns:
                    pct = (df[c] / df[tot_col] * 100.0).clip(lower=0, upper=100)
                    fig.add_trace(go.Scatter(x=ts, y=ctx.smooth_series(pct), mode="lines", name=f"GPU {idx} Mem %"))
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="%")
            return fig

    # Helpers for Docker naming (support old and new) — module scope so callbacks can access
    def docker_container_names(df):
        try:
            names = set()
            for col in df.columns:
                if col.endswith("_container_cpu_percent"):
                    names.add(col[: -len("_container_cpu_percent")])
                elif col.startswith("docker_") and col.endswith("_cpu_percent"):
                    names.add(col[len("docker_") : -len("_cpu_percent")])
            return sorted(names)
        except Exception:
            return []

    def docker_pick_col(df, name, new_suffix, old_suffix):
        for cand in (f"docker_{name}_{new_suffix}", f"{name}_{old_suffix}"):
            if cand in df.columns:
                return cand
        return None

    class ContainerCPUGraph(GraphComponent):
        component_id = "container-cpu-utilization-graph"
        title = "Container CPU %"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            for name in params.get("selected_containers", []):
                col = docker_pick_col(df, name, "cpu_percent", "container_cpu_percent")
                if col:
                    fig.add_trace(go.Scatter(x=ts, y=ctx.smooth_series(df[col]), mode="lines", name=f"{name} CPU%"))
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="%")
            return fig

    class ContainerMemGraph(GraphComponent):
        component_id = "container-memory-utilization-graph"
        title = "Container Memory %"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            for name in params.get("selected_containers", []):
                col = docker_pick_col(df, name, "mem_percent", "container_mem_percent")
                if col:
                    fig.add_trace(go.Scatter(x=ts, y=ctx.smooth_series(df[col]), mode="lines", name=f"{name} Mem%"))
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="%")
            return fig

    class ContainerFilesGraph(GraphComponent):
        component_id = "container-files-graph"
        title = "Container Open Files"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            for name in params.get("selected_containers", []):
                col = docker_pick_col(df, name, "open_files", "container_open_files")
                if col:
                    fig.add_trace(go.Scatter(x=ts, y=ctx.smooth_series(df[col]), mode="lines", name=f"{name} Files"))
            if len(fig.data) > 0:
                fig.update_layout(title=self.title)
            return fig

    class ContainerNetGraph(GraphComponent):
        component_id = "container-net-graph"
        title = "Container Network (selected sum)"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            rx_cols = []
            tx_cols = []
            for n in params.get("selected_containers", []):
                col_rx = docker_pick_col(df, n, "net_rx_bytes_per_sec", "container_net_rx_bytes_per_sec")
                col_tx = docker_pick_col(df, n, "net_tx_bytes_per_sec", "container_net_tx_bytes_per_sec")
                if col_rx:
                    rx_cols.append(col_rx)
                if col_tx:
                    tx_cols.append(col_tx)
            if rx_cols:
                fig.add_trace(
                    go.Scatter(
                        x=ts, y=ctx.smooth_series(df[rx_cols].sum(axis=1)) / (1024**2), mode="lines", name="RX MB/s"
                    )
                )
            if tx_cols:
                fig.add_trace(
                    go.Scatter(
                        x=ts, y=ctx.smooth_series(df[tx_cols].sum(axis=1)) / (1024**2), mode="lines", name="TX MB/s"
                    )
                )
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="MB/s")
            return fig

    class ContainerIOGraph(GraphComponent):
        component_id = "container-io-graph"
        title = "Container Disk I/O (selected sum)"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            r_cols = []
            w_cols = []
            for n in params.get("selected_containers", []):
                col_r = docker_pick_col(df, n, "blkio_read_bytes_per_sec", "container_blkio_read_bytes_per_sec")
                col_w = docker_pick_col(df, n, "blkio_write_bytes_per_sec", "container_blkio_write_bytes_per_sec")
                if col_r:
                    r_cols.append(col_r)
                if col_w:
                    w_cols.append(col_w)
            if r_cols:
                fig.add_trace(
                    go.Scatter(
                        x=ts, y=ctx.smooth_series(df[r_cols].sum(axis=1)) / (1024**2), mode="lines", name="Read MB/s"
                    )
                )
            if w_cols:
                fig.add_trace(
                    go.Scatter(
                        x=ts, y=ctx.smooth_series(df[w_cols].sum(axis=1)) / (1024**2), mode="lines", name="Write MB/s"
                    )
                )
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="MB/s")
            return fig

    class TopContainersCPUBar(GraphComponent):
        component_id = "container-top-cpu-graph"
        title = "Top Containers by CPU (latest)"
        is_time_series = False

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            all_containers = docker_container_names(df)
            if not all_containers or df.empty:
                return fig
            latest = df.iloc[-1]
            pairs = []
            for name in all_containers:
                col = docker_pick_col(df, name, "cpu_percent", "container_cpu_percent")
                if col:
                    pairs.append((name, latest[col]))
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
            if pairs:
                fig.add_trace(go.Bar(x=[n for n, _ in pairs], y=[v for _, v in pairs], name="CPU %"))
                fig.update_layout(title=self.title)
            return fig

    class TopContainersMemBar(GraphComponent):
        component_id = "container-top-mem-graph"
        title = "Top Containers by Mem (latest)"
        is_time_series = False

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            all_containers = docker_container_names(df)
            if not all_containers or df.empty:
                return fig
            latest = df.iloc[-1]
            pairs = []
            for name in all_containers:
                col = docker_pick_col(df, name, "mem_percent", "container_mem_percent")
                if col:
                    pairs.append((name, latest[col]))
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
            if pairs:
                fig.add_trace(go.Bar(x=[n for n, _ in pairs], y=[v for _, v in pairs], name="Mem %"))
                fig.update_layout(title=self.title)
            return fig

    class ProcessCountGraph(GraphComponent):
        component_id = "process-count-graph"
        title = "System Process Count"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            if "system_process_count" in df.columns:
                fig.add_trace(
                    go.Scatter(x=ts, y=ctx.smooth_series(df["system_process_count"]), mode="lines", name="Processes")
                )
                fig.update_layout(title=self.title)
            return fig

    class ThreadCountGraph(GraphComponent):
        component_id = "thread-count-graph"
        title = "System Thread Count"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            if "system_thread_count" in df.columns:
                fig.add_trace(
                    go.Scatter(x=ts, y=ctx.smooth_series(df["system_thread_count"]), mode="lines", name="Threads")
                )
                fig.update_layout(title=self.title)
            return fig

    class OverviewGraph(GraphComponent):
        component_id = "system-overview-graph"
        title = "System Overview"

        def build(self, df, ts, ctx, params):
            fig = go.Figure()
            # include CPU aggregate and Memory % if available
            cpu_cols = [c for c in df.columns if c.startswith("cpu_") and c.endswith("_utilization")]
            if cpu_cols:
                cpu_mean = ctx.smooth_series(df[cpu_cols].mean(axis=1))
                fig.add_trace(go.Scatter(x=ts, y=cpu_mean, mode="lines", name="CPU %"))
            if {"sys_used", "sys_total"}.issubset(df.columns):
                mem_pct = (df["sys_used"] / df["sys_total"] * 100.0).clip(lower=0, upper=100)
                fig.add_trace(go.Scatter(x=ts, y=ctx.smooth_series(mem_pct), mode="lines", name="Mem %"))
            if len(fig.data) > 0:
                fig.update_layout(title=self.title, yaxis_title="%")
            return fig

    def add_event_markers(fig, events_list, display_tz, display_tz_custom):
        try:
            events_list = events_list or []
            for evt in events_list:
                ts_raw = evt.get("timestamp")
                name = evt.get("name", "event")
                if not ts_raw:
                    continue
                ts = pd.to_datetime(ts_raw)
                # Events are stored UTC-naive internally. Convert to selected display tz (naive) for rendering.
                try:
                    if display_tz == "local" and tzlocal is not None:
                        ts = ts.tz_localize("UTC").tz_convert(tzlocal()).tz_localize(None)
                    elif display_tz == "custom" and display_tz_custom and gettz is not None:
                        tz = gettz(display_tz_custom)
                        if tz is not None:
                            ts = ts.tz_localize("UTC").tz_convert(tz).tz_localize(None)
                    else:
                        # utc: leave as UTC-naive
                        pass
                except Exception:
                    pass
                # Draw vertical line via shape for broader Plotly compatibility
                try:
                    fig.add_shape(
                        type="line",
                        xref="x",
                        x0=ts,
                        x1=ts,
                        yref="paper",
                        y0=0,
                        y1=1,
                        line=dict(color="#8888d8", width=1, dash="dash"),
                        layer="above",
                    )
                    fig.add_annotation(
                        x=ts,
                        y=1,
                        xref="x",
                        yref="paper",
                        text=name,
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=10),
                        bgcolor="rgba(136,132,216,0.15)",
                    )
                except Exception:
                    pass
        except Exception:
            pass
        return fig

    def render_event_list(events_list):
        items = [html.Div(f"{e.get('timestamp', '')} — {e.get('name', 'event')}") for e in (events_list or [])]
        if not items:
            return html.Div("No events", style={"opacity": 0.7})
        return items

    def make_empty_fig(title):
        fig = go.Figure()
        fig.add_annotation(text=title, showarrow=False, yref="paper", y=0.5, xref="paper", x=0.5)
        fig.update_layout(margin=dict(l=30, r=10, t=30, b=30))
        return fig

    def convert_ts_for_display(ts_series, display_tz, display_tz_custom):
        try:
            ts = pd.to_datetime(ts_series)
            # Our data timestamps are stored UTC-naive internally.
            if display_tz == "local" and tzlocal is not None:
                try:
                    ts = ts.dt.tz_localize("UTC").dt.tz_convert(tzlocal()).dt.tz_localize(None)
                except Exception:
                    pass
            elif display_tz == "custom" and display_tz_custom and gettz is not None:
                try:
                    tz = gettz(display_tz_custom)
                    if tz is not None:
                        ts = ts.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
                except Exception:
                    pass
            # If displaying UTC, keep UTC-naive as-is.
            return ts
        except Exception:
            return pd.to_datetime(ts_series, errors="coerce")

    def event_times_for_display(events_list, display_tz, display_tz_custom):
        try:
            if not events_list:
                return pd.Series([], dtype="datetime64[ns]")
            ets = pd.to_datetime([e.get("timestamp") for e in events_list if e.get("timestamp")])
            try:
                if display_tz == "local" and tzlocal is not None:
                    ets = pd.DatetimeIndex(ets).tz_localize("UTC").tz_convert(tzlocal()).tz_localize(None)
                elif display_tz == "custom" and display_tz_custom and gettz is not None:
                    tz = gettz(display_tz_custom)
                    if tz is not None:
                        ets = pd.DatetimeIndex(ets).tz_localize("UTC").tz_convert(tz).tz_localize(None)
            except Exception:
                pass
            return pd.Series(ets)
        except Exception:
            return pd.Series([], dtype="datetime64[ns]")

    # Callback to update all graphs periodically
    @app.callback(
        [
            Output("page-container", "style"),
            Output("data-source", "children"),
            Output("last-updated", "children"),
            Output("kpi-row", "children"),
            Output("system-overview-graph", "figure"),
            Output("cpu-individual-utilization-graph", "figure"),
            Output("cpu-aggregated-utilization-graph", "figure"),
            Output("memory-graph", "figure"),
            Output("file-count-graph", "figure"),
            Output("fd-usage-graph", "figure"),
            Output("network-graph", "figure"),
            Output("disk-io-graph", "figure"),
            Output("gpu-utilization-graph", "figure"),
            Output("gpu-memory-graph", "figure"),
            Output("container-cpu-utilization-graph", "figure"),
            Output("container-memory-utilization-graph", "figure"),
            Output("container-files-graph", "figure"),
            Output("container-net-graph", "figure"),
            Output("container-io-graph", "figure"),
            Output("container-top-cpu-graph", "figure"),
            Output("container-top-mem-graph", "figure"),
            Output("process-count-graph", "figure"),
            Output("thread-count-graph", "figure"),
        ],
        [
            Input("interval-component", "n_intervals"),
            Input("time-range", "value"),
            Input("theme-toggle", "value"),
            Input("smoothing-window", "value"),
            Input("container-auto-top", "value"),
            Input("container-top-n", "value"),
            Input("container-select", "value"),
            Input("event-store", "data"),
            Input("event-auto-store", "data"),
            Input("event-display-options", "value"),
            Input("display-tz", "value"),
            Input("display-tz-custom", "value"),
        ],
    )
    def update_graphs(
        n,
        time_range,
        theme_value,
        smoothing_window,
        auto_top,
        top_n,
        selected_manual,
        events_data,
        auto_events,
        event_display_options,
        display_tz,
        display_tz_custom,
    ):
        # Load data
        df = load_data(time_range)
        last_timestamp = "Never"
        if not df.empty and "timestamp" in df.columns:
            try:
                last_timestamp = pd.to_datetime(df["timestamp"].max()).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                last_timestamp = str(df["timestamp"].max())

        # Helpers
        def smooth_series(s):
            try:
                w = max(1, int(smoothing_window or 1))
                if w > 1:
                    return s.rolling(window=w, min_periods=1).mean()
            except Exception:
                pass
            return s

        ts = (
            convert_ts_for_display(df["timestamp"], display_tz, display_tz_custom)
            if (not df.empty and "timestamp" in df.columns)
            else pd.Series([])
        )

        # Build display timezone label and offset for UI
        def display_tz_info():
            try:
                label = "UTC"
                tzinfo = None
                if display_tz == "local" and tzlocal is not None:
                    tzinfo = tzlocal()
                    label = "Local"
                elif display_tz == "custom" and display_tz_custom and gettz is not None:
                    tzinfo = gettz(display_tz_custom)
                    label = f"Custom ({display_tz_custom})"
                else:
                    from dateutil.tz import tzutc

                    tzinfo = tzutc()
                    label = "UTC"
                now_dt = datetime.now(tzinfo) if tzinfo is not None else datetime.utcnow()
                offset = now_dt.utcoffset() or pd.Timedelta(0)
                total_minutes = int(offset.total_seconds() // 60)
                sign = "+" if total_minutes >= 0 else "-"
                hh = abs(total_minutes) // 60
                mm = abs(total_minutes) % 60
                offset_str = f"UTC{sign}{hh:02d}:{mm:02d}"
                # Try to get a friendly tz name
                tzname = now_dt.tzname() if tzinfo is not None else "UTC"
                return label, tzname, offset_str
            except Exception:
                return "UTC", "UTC", "UTC+00:00"

        disp_label, disp_name, disp_offset = display_tz_info()

        # KPI row (compact, minimal)
        def kpi_card(label, value, spark_fig=None):
            return html.Div(
                [
                    html.Div(label, style={"fontSize": "12px", "opacity": 0.7}),
                    html.Div(value, style={"fontSize": "18px", "fontWeight": "600"}),
                    (dcc.Graph(figure=spark_fig, style={"height": "36px"}) if spark_fig is not None else None),
                ],
                style={"border": "1px solid #333", "borderRadius": "6px", "padding": "8px", "marginRight": "8px"},
            )

        kpi_children = [kpi_card("Samples", f"{len(df)}")]

        # Add key latest metrics if present (Tufte-inspired: high information density, no chartjunk)
        try:
            if not df.empty:
                latest_row = df.iloc[-1]
                # CPU % (aggregate)
                cpu_cols = [c for c in df.columns if c.startswith("cpu_") and c.endswith("_utilization")]
                if cpu_cols:
                    cpu_latest = float(pd.to_numeric(latest_row[cpu_cols], errors="coerce").mean())
                    # sparkline for CPU mean
                    try:
                        cpu_mean_series = pd.to_numeric(df[cpu_cols], errors="coerce").mean(axis=1)
                        cpu_spark = make_sparkline(ts, cpu_mean_series, theme_value)
                    except Exception:
                        cpu_spark = None
                    kpi_children.append(kpi_card("CPU %", f"{cpu_latest:0.1f}", cpu_spark))
                # Mem %
                if {"sys_used", "sys_total"}.issubset(df.columns):
                    try:
                        mem_latest = float(latest_row["sys_used"]) / float(latest_row["sys_total"]) * 100.0
                        mem_latest = max(0.0, min(100.0, mem_latest))
                        try:
                            mem_pct_series = (df["sys_used"] / df["sys_total"] * 100.0).clip(lower=0, upper=100)
                            mem_spark = make_sparkline(ts, mem_pct_series, theme_value)
                        except Exception:
                            mem_spark = None
                        kpi_children.append(kpi_card("Mem %", f"{mem_latest:0.1f}", mem_spark))
                    except Exception:
                        pass
                # Processes / Threads
                if "system_process_count" in df.columns:
                    try:
                        kpi_children.append(kpi_card("Procs", f"{int(latest_row['system_process_count'])}"))
                    except Exception:
                        pass
                if "system_thread_count" in df.columns:
                    try:
                        kpi_children.append(kpi_card("Threads", f"{int(latest_row['system_thread_count'])}"))
                    except Exception:
                        pass
                # Net MB/s (sum up/down)
                rx_col = "net_bytes_recv_per_sec" if "net_bytes_recv_per_sec" in df.columns else None
                tx_col = "net_bytes_sent_per_sec" if "net_bytes_sent_per_sec" in df.columns else None
                if rx_col or tx_col:
                    try:
                        rx = float(latest_row.get(rx_col, 0.0) or 0.0)
                        tx = float(latest_row.get(tx_col, 0.0) or 0.0)
                        mbps = (rx + tx) / (1024**2)
                        try:
                            rx_series = df[rx_col] if rx_col in df.columns else 0.0
                            tx_series = df[tx_col] if tx_col in df.columns else 0.0
                            net_series = (
                                pd.to_numeric(rx_series, errors="coerce").fillna(0)
                                + pd.to_numeric(tx_series, errors="coerce").fillna(0)
                            ) / (1024**2)
                            net_spark = make_sparkline(ts, net_series, theme_value)
                        except Exception:
                            net_spark = None
                        kpi_children.append(kpi_card("Net MB/s", f"{mbps:0.2f}", net_spark))
                    except Exception:
                        pass
                # Disk MB/s (sum r+w)
                r_col = "disk_read_bytes_per_sec" if "disk_read_bytes_per_sec" in df.columns else None
                w_col = "disk_write_bytes_per_sec" if "disk_write_bytes_per_sec" in df.columns else None
                if r_col or w_col:
                    try:
                        r = float(latest_row.get(r_col, 0.0) or 0.0)
                        w = float(latest_row.get(w_col, 0.0) or 0.0)
                        mbps = (r + w) / (1024**2)
                        try:
                            r_series = df[r_col] if r_col in df.columns else 0.0
                            w_series = df[w_col] if w_col in df.columns else 0.0
                            io_series = (
                                pd.to_numeric(r_series, errors="coerce").fillna(0)
                                + pd.to_numeric(w_series, errors="coerce").fillna(0)
                            ) / (1024**2)
                            io_spark = make_sparkline(ts, io_series, theme_value)
                        except Exception:
                            io_spark = None
                        kpi_children.append(kpi_card("Disk MB/s", f"{mbps:0.2f}", io_spark))
                    except Exception:
                        pass
        except Exception:
            pass

        # Build component registry and figures using strategy pattern
        # Determine selected containers first (support old and new docker column names)
        all_containers = docker_container_names(df) if (not df.empty) else []
        if all_containers:
            if auto_top and "auto" in (auto_top or []):
                latest = df.iloc[-1]
                scored = []
                for name in all_containers:
                    col = docker_pick_col(df, name, "cpu_percent", "container_cpu_percent")
                    if col:
                        scored.append((name, latest[col]))
                selected_containers = [
                    n for n, _ in sorted(scored, key=lambda x: x[1], reverse=True)[: int(top_n or 5)]
                ]
            else:
                selected_containers = selected_manual or []
            if not selected_containers:
                selected_containers = all_containers[: min(5, len(all_containers))]
        else:
            selected_containers = []

        ctx_obj = GraphContext(smoothing_window, apply_theme, smooth_series)
        params = {"selected_containers": selected_containers}

        components = [
            OverviewGraph(),
            CPUIndividualGraph(),
            CPUAggregateGraph(),
            MemoryGraph(),
            FileCountGraph(),
            FDUsageGraph(),
            NetworkGraph(),
            DiskIOGraph(),
            GPUUtilGraph(),
            GPUMemoryGraph(),
            ContainerCPUGraph(),
            ContainerMemGraph(),
            ContainerFilesGraph(),
            ContainerNetGraph(),
            ContainerIOGraph(),
            TopContainersCPUBar(),
            TopContainersMemBar(),
            ProcessCountGraph(),
            ThreadCountGraph(),
        ]

        # Build figures map by id
        figures_by_id = {c.component_id: go.Figure() for c in components}
        if not df.empty and "timestamp" in df.columns:
            for comp in components:
                try:
                    figures_by_id[comp.component_id] = comp.build(df, ts, ctx_obj, params)
                except Exception:
                    figures_by_id[comp.component_id] = go.Figure()

        # Merge events and theme
        merged_events = (events_data or []) + (auto_events or [])
        # Add events KPI
        try:
            evt_count = len(merged_events)
        except Exception:
            evt_count = 0
        kpi_children.append(
            html.Div(
                [
                    html.Div("Events", style={"fontSize": "12px", "opacity": 0.7}),
                    html.Div(f"{evt_count}", style={"fontSize": "18px", "fontWeight": "600"}),
                ],
                style={"border": "1px solid #333", "borderRadius": "6px", "padding": "8px"},
            )
        )
        # Compute visible event time bounds to ensure markers are in-range
        evt_ts = event_times_for_display(merged_events, display_tz, display_tz_custom)
        # Removed unused figs_all variable (was redundant with time_series_figs and not referenced)
        # Time-series figs exclude categorical bar charts (top_*). Bar charts won't get event lines.
        # Post-process: apply x-axis type, include events and theme to time-series figs
        time_series_ids = [
            "system-overview-graph",
            "cpu-individual-utilization-graph",
            "cpu-aggregated-utilization-graph",
            "memory-graph",
            "file-count-graph",
            "fd-usage-graph",
            "network-graph",
            "disk-io-graph",
            "gpu-utilization-graph",
            "gpu-memory-graph",
            "container-cpu-utilization-graph",
            "container-memory-utilization-graph",
            "container-files-graph",
            "container-net-graph",
            "container-io-graph",
            "process-count-graph",
            "thread-count-graph",
        ]
        markers_enabled = True
        try:
            markers_enabled = (event_display_options is None) or ("markers" in (event_display_options or []))
        except Exception:
            markers_enabled = True

        for fig_id in time_series_ids:
            f = figures_by_id.get(fig_id, go.Figure())
            # Ensure date x-axis for proper placement of vertical lines
            f.update_xaxes(type="date")
            # If we have both data ts and event ts, expand range to include both
            try:
                if len(ts) > 0 and len(evt_ts) > 0:
                    xmin = min(pd.to_datetime(ts.min()), pd.to_datetime(evt_ts.min()))
                    xmax = max(pd.to_datetime(ts.max()), pd.to_datetime(evt_ts.max()))
                    # Add small padding
                    pad = pd.Timedelta(seconds=1)
                    f.update_xaxes(range=[xmin - pad, xmax + pad])
                elif len(ts) == 0 and len(evt_ts) > 0:
                    # No data, but we do have events: center axis around events so markers are visible
                    xmin = pd.to_datetime(evt_ts.min())
                    xmax = pd.to_datetime(evt_ts.max())
                    if xmin == xmax:
                        xmax = xmin + pd.Timedelta(minutes=1)
                    pad = pd.Timedelta(seconds=1)
                    f.update_xaxes(range=[xmin - pad, xmax + pad])
            except Exception:
                pass
            if markers_enabled:
                add_event_markers(f, merged_events, display_tz, display_tz_custom)
            apply_theme(f, theme_value)
            style_minimal_figure(f, theme_value)
            # Always show legends for clarity (after styling which doesn't override legend)
            f.update_layout(
                showlegend=True,
                legend=dict(orientation="h", x=0.0, y=1.0, yanchor="top"),
                margin=dict(l=40, r=10, t=40, b=28),
            )

        # Apply theme to bar charts as well
        for fig_id in ["container-top-cpu-graph", "container-top-mem-graph"]:
            f = figures_by_id.get(fig_id, go.Figure())
            apply_theme(f, theme_value)
            style_minimal_figure(f, theme_value)
            f.update_layout(showlegend=True)

        # Page style per theme
        page_style = {
            "padding": "20px",
            "backgroundColor": ("#111111" if theme_value == "dark" else "#ffffff"),
            "color": ("#e5e5e5" if theme_value == "dark" else "#222222"),
        }

        return (
            page_style,
            f"Data source: {datafile} | Data TZ: UTC-naive (stored); Displayed in:"
            f" {disp_label} ({disp_name}, {disp_offset})",
            f"Last updated: {last_timestamp} | Display TZ: {disp_label} ({disp_name}, {disp_offset})",
            kpi_children,
            figures_by_id["system-overview-graph"],
            figures_by_id["cpu-individual-utilization-graph"],
            figures_by_id["cpu-aggregated-utilization-graph"],
            figures_by_id["memory-graph"],
            figures_by_id["file-count-graph"],
            figures_by_id["fd-usage-graph"],
            figures_by_id["network-graph"],
            figures_by_id["disk-io-graph"],
            figures_by_id["gpu-utilization-graph"],
            figures_by_id["gpu-memory-graph"],
            figures_by_id["container-cpu-utilization-graph"],
            figures_by_id["container-memory-utilization-graph"],
            figures_by_id["container-files-graph"],
            figures_by_id["container-net-graph"],
            figures_by_id["container-io-graph"],
            figures_by_id["container-top-cpu-graph"],
            figures_by_id["container-top-mem-graph"],
            figures_by_id["process-count-graph"],
            figures_by_id["thread-count-graph"],
        )

    # Populate container dropdown options dynamically
    @app.callback(
        Output("container-select", "options"),
        [Input("interval-component", "n_intervals"), Input("time-range", "value")],
    )
    def update_container_options(n, time_range):
        try:
            df = load_data(time_range)
            if df.empty:
                return []
            names = sorted(
                {
                    col.replace("_container_cpu_percent", "")
                    for col in df.columns
                    if col.endswith("_container_cpu_percent")
                }
            )
            return [{"label": name, "value": name} for name in names]
        except Exception:
            return []

    # Clientside callback to sync body background with theme
    app.clientside_callback(
        """
        function(theme){
            var bg = (theme === 'dark') ? '#111111' : '#ffffff';
            document.body.style.backgroundColor = bg;
            document.documentElement.style.backgroundColor = bg;
            return '';
        }
        """,
        Output("body-bg-sync", "children"),
        Input("theme-toggle", "value"),
    )

    # Manage events: add, add-now, clear
    @app.callback(
        [Output("event-store", "data"), Output("event-list", "children")],
        [
            Input("add-event-btn", "n_clicks"),
            Input("add-event-now-btn", "n_clicks"),
            Input("clear-events-btn", "n_clicks"),
            Input("event-upload", "contents"),
        ],
        [
            State("event-name", "value"),
            State("event-datetime-text", "value"),
            State("event-store", "data"),
            State("event-auto-store", "data"),
            State("time-range", "value"),
            State("display-tz", "value"),
            State("display-tz-custom", "value"),
        ],
    )
    def manage_events(
        n_add,
        n_now,
        n_clear,
        upload_contents,
        name,
        dt_text,
        data,
        auto_data,
        time_range,
        display_tz,
        display_tz_custom,
    ):
        data = data or []
        triggered = getattr(dash, "callback_context", None)
        trig_id = ""
        if triggered and triggered.triggered:
            trig_id = triggered.triggered[0]["prop_id"].split(".")[0]

        # Helper to get latest timestamp from datafile
        def latest_data_ts():
            df = load_data(time_range)
            if not df.empty and "timestamp" in df.columns:
                try:
                    return df["timestamp"].max()
                except Exception:
                    pass
            # Fallback: now in selected display timezone (naive), will be normalized to UTC-naive below
            return pd.Timestamp(datetime.now())

        def selected_tzinfo():
            if display_tz == "utc":
                try:
                    from dateutil.tz import tzutc

                    return tzutc()
                except Exception:
                    return None
            if display_tz == "custom" and display_tz_custom and gettz is not None:
                tz = gettz(display_tz_custom)
                if tz is not None:
                    return tz
            return tzlocal() if tzlocal is not None else None

        def normalize_ts_to_utc_naive(ts_any):
            ts = pd.to_datetime(ts_any)
            # If tz-aware: convert to UTC then drop tz
            if getattr(ts, "tzinfo", None) is not None and ts.tzinfo is not None:
                try:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                except Exception:
                    try:
                        # If tz_convert fails, maybe it's offset-naive; localize first assuming UTC
                        ts = ts.tz_localize("UTC").tz_localize(None)
                    except Exception:
                        pass
                return ts
            # tz-naive: assume in the selected display tz, convert to UTC, drop tz
            try:
                tzinf = selected_tzinfo()
                if tzinf is not None:
                    ts_loc = ts.tz_localize(tzinf)
                    ts = ts_loc.tz_convert("UTC").tz_localize(None)
                    return ts
            except Exception:
                pass
            return ts

        def from_text_to_utc_naive(val):
            if not val:
                raise ValueError("no datetime provided")
            base = pd.to_datetime(val)
            # Treat input as wall time in selected display tz; convert to UTC, then drop tz
            tzinf = selected_tzinfo()
            if tzinf is not None:
                try:
                    base_loc = base.tz_localize(tzinf)
                    base = base_loc.tz_convert("UTC").tz_localize(None)
                except Exception:
                    pass
            return base

        if trig_id == "clear-events-btn":
            return [], html.Div("No events", style={"opacity": 0.7})

        if trig_id in ("add-event-btn", "add-event-now-btn"):
            evt_name = (name or "").strip() or "event"
            if trig_id == "add-event-now-btn":
                # Use current time in selected tz -> convert to UTC-naive for storage
                now_dt = pd.Timestamp(datetime.now())
                tzinf = selected_tzinfo()
                if tzinf is not None:
                    try:
                        now_dt = now_dt.tz_localize(tzinf).tz_convert("UTC").tz_localize(None)
                    except Exception:
                        pass
                ts = now_dt
            else:
                # Use date/time picker if provided, else latest data timestamp
                if dt_text:
                    try:
                        ts = from_text_to_utc_naive(dt_text)
                    except Exception:
                        ts = latest_data_ts()
                else:
                    ts = latest_data_ts()
            # Normalize to isoformat string
            try:
                ts_norm = normalize_ts(ts)
                ts_str = pd.to_datetime(ts_norm).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts_str = str(ts)
            new_data = data + [{"name": evt_name, "timestamp": ts_str}]
            items = [html.Div(f"{d['timestamp']} — {d['name']}") for d in new_data]
            return new_data, items

        if trig_id == "event-upload":
            try:
                content_type, content_string = upload_contents.split(",")
                decoded = base64.b64decode(content_string)
                text = decoded.decode("utf-8", errors="ignore")
                reader = csv.reader(io.StringIO(text))
                imported = []
                for row in reader:
                    if not row:
                        continue
                    if len(row) == 1:
                        # try split by comma manually
                        parts = row[0].split(",")
                        if len(parts) >= 2:
                            row = [parts[0], ",".join(parts[1:])]
                        else:
                            continue
                    evt_name = (row[0] or "").strip() or "event"
                    ts_text = (row[1] or "").strip()
                    if not ts_text:
                        continue
                    try:
                        ts = normalize_ts_to_utc_naive(ts_text)
                        ts_str = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        ts_str = str(ts_text)
                    imported.append({"name": evt_name, "timestamp": ts_str})
                new_data = (data or []) + imported
                items = [html.Div(f"{d['timestamp']} — {d['name']}") for d in new_data]
                return new_data, items
            except Exception:
                items = [html.Div(f"{d['timestamp']} — {d['name']}") for d in data]
                if not items:
                    items = html.Div("No events", style={"opacity": 0.7})
                return data, items

        # default: just render existing
        items = [html.Div(f"{d['timestamp']} — {d['name']}") for d in data]
        if not items:
            items = html.Div("No events", style={"opacity": 0.7})
        return data, items

    # Start the server
    print(f"Starting dashboard server on http://{host}:{port}/")
    print(f"Using data file: {datafile}")
    print(f"Refresh interval: {interval} seconds")
    print("Press Ctrl+C to stop the server")
    app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()  # This invokes the Click command
