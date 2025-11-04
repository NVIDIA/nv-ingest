import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State, ALL
import pandas as pd
import plotly.graph_objects as go
import os
import click
import base64
import io
import csv
from datetime import datetime
import json
import psutil

# noqa
# flake8: noqa

# Use absolute package import only (no relatives or fallbacks)
from system_tracer import (
    get_process_tree_summary,
    SystemTracer,
)
from layout import build_layout

try:
    from dateutil.tz import tzlocal, gettz
except Exception:
    tzlocal = None
    gettz = None

try:
    import dash_cytoscape as cy  # type: ignore

    CY_AVAILABLE = True
except Exception:
    CY_AVAILABLE = False
    cy = None

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

    # Validate the data file (be permissive; pandas can use either pyarrow or fastparquet)
    if not os.path.exists(datafile):
        print(f"Error: Data file '{datafile}' not found.")
        print("Dashboard will start but won't display data until the file exists.")
    elif datafile.endswith(".parquet") and not PARQUET_AVAILABLE:
        print("Warning: pyarrow is not installed; will attempt to read parquet via pandas (fastparquet if available).")
    elif not datafile.endswith(".parquet") and not datafile.endswith(".csv"):
        print(f"Warning: Data file '{datafile}' is not a .parquet or .csv file.")
        print("Attempting to load it anyway, but this may cause errors.")

    # Initialize the Dash app and ensure assets/ resolves to the packaged assets by default
    pkg_dir = os.path.abspath(os.path.dirname(__file__))
    default_assets = os.path.join(pkg_dir, "assets")
    assets_override = os.environ.get("SYSTEM_MONITOR_ASSETS")
    assets_path = assets_override if assets_override and os.path.isdir(assets_override) else default_assets
    app = dash.Dash(__name__, assets_folder=assets_path)

    # Global Plotly defaults: Inter font and transparent backgrounds (Tufte-style minimalism)
    try:
        import plotly.io as pio  # type: ignore

        tufted = go.layout.Template()
        tufted.layout.font.family = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
        tufted.layout.paper_bgcolor = "rgba(0,0,0,0)"
        tufted.layout.plot_bgcolor = "rgba(0,0,0,0)"
        pio.templates["tufted"] = tufted
        pio.templates.default = "tufted"
    except Exception:
        pass

    # Keep index minimal; styling handled via assets/style.css
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
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

    # Use extracted layout factory (overrides inline layout above)
    app.layout = build_layout(
        datafile=datafile,
        interval=interval,
        cy_available=CY_AVAILABLE,
        cy_module=cy,
    )

    # Helper function to load and filter data
    def load_data(data_path, time_range_minutes, source_mode="auto"):
        try:
            # Live buffer branch (explicit when mode == live, or auto + running)
            if source_mode in ("live", "auto"):
                try:
                    if _is_running():
                        with _tracer_lock:
                            tracer = _tracer_obj.get("tracer")
                            if tracer is not None and getattr(tracer, "data_buffer", None) is not None:
                                df = pd.DataFrame(list(tracer.data_buffer))
                                if time_range_minutes > 0 and not df.empty and "timestamp" in df.columns:
                                    latest_time = pd.to_datetime(df["timestamp"]).max()
                                    time_threshold = latest_time - pd.Timedelta(minutes=time_range_minutes)
                                    df = df[pd.to_datetime(df["timestamp"]) >= time_threshold]
                                return df
                except Exception as le:
                    print(f"Error reading live buffer: {le}")

            # File branch (explicit when mode == file, or auto + no live)
            if data_path and os.path.exists(data_path):
                # Prefer parquet if extension says so; let pandas pick available engine (pyarrow/fastparquet)
                if data_path.endswith(".parquet"):
                    try:
                        df = pd.read_parquet(data_path)
                    except Exception as pe:
                        print(f"Parquet read failed via pandas: {pe}. Trying CSV fallback (may fail)...")
                        df = pd.read_csv(data_path, parse_dates=["timestamp"])  # best-effort
                elif data_path.endswith(".csv"):
                    df = pd.read_csv(data_path, parse_dates=["timestamp"])
                else:
                    # Try parquet first, then CSV
                    try:
                        df = pd.read_parquet(data_path)
                    except Exception:
                        df = pd.read_csv(data_path, parse_dates=["timestamp"])  # best-effort

                # Filter by time range if specified
                if time_range_minutes > 0 and not df.empty and "timestamp" in df.columns:
                    latest_time = pd.to_datetime(df["timestamp"]).max()
                    time_threshold = latest_time - pd.Timedelta(minutes=time_range_minutes)
                    df = df[pd.to_datetime(df["timestamp"]) >= time_threshold]

                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    # Interval enable/disable based on pause toggle
    @app.callback(
        Output("interval-component", "disabled"),
        [Input("pause-refresh", "value")],
    )
    def _toggle_interval_disabled(pause_values):
        try:
            return isinstance(pause_values, list) and ("pause" in pause_values)
        except Exception:
            return False

    # Notice banner content (guides first-time usage)
    @app.callback(
        Output("notice-banner", "children"),
        [
            Input("interval-component", "n_intervals"),
            Input("datafile-store", "data"),
            Input("data-source-mode", "value"),
        ],
    )
    def _notice_banner(_n, data_path, source_mode):
        try:
            running = _is_running()
            has_file = bool(data_path) and os.path.exists(data_path)
            if source_mode == "live":
                if not running:
                    return html.Div(
                        [
                            html.Strong("No live data. "),
                            "Click Start in Live Tracing to begin collecting metrics. ",
                            "Or switch Data source mode to File and set a Parquet/CSV path.",
                        ]
                    )
            elif source_mode == "file":
                if not has_file:
                    return html.Div(
                        [
                            html.Strong("No file loaded. "),
                            "Set Output Parquet Path to an existing Parquet/CSV and Start the tracer, ",
                            "or switch Data source mode to Live to collect data in-memory.",
                        ]
                    )
            else:  # auto
                if not running and not has_file:
                    return html.Div(
                        [
                            html.Strong("No data available. "),
                            "Start live tracing (left) or set a Parquet/CSV in Output Parquet Path. ",
                            "Data source mode is Auto: it will prefer live data when the tracer is running.",
                        ]
                    )
        except Exception:
            pass
        return ""

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

    # PID search: suggest processes matching a search string (similar to `ps -AFl | grep <pattern>`)
    @app.callback(
        Output("proctree-suggestions", "options"),
        [Input("proctree-search", "value")],
        prevent_initial_call=False,
    )
    def update_proctree_suggestions(search_text):
        opts = []
        try:
            pattern = (search_text or "").strip()
            if not pattern:
                return []
            pattern_low = pattern.lower()
            # Collect processes with safe attribute access
            matches = []
            for p in psutil.process_iter(attrs=["pid", "name", "username", "cmdline", "num_threads"]):
                try:
                    info = p.info
                    pid = info.get("pid")
                    name = info.get("name") or ""
                    username = info.get("username") or ""
                    cmdline_list = info.get("cmdline") or []
                    cmdline = " ".join(cmdline_list)
                    haystack = f"{name} {cmdline}".lower()
                    if pattern_low in haystack:
                        threads = info.get("num_threads") or 0
                        label = f"PID {pid} • {username} • thr={threads} • {name} — {cmdline}".strip()
                        matches.append({"label": label[:300], "value": pid})
                except Exception:
                    continue
            # Limit to first 50
            opts = matches[:50]
        except Exception:
            opts = []
        return opts

    # When a suggestion is selected, set the PID input
    @app.callback(
        Output("proctree-pid", "value"),
        [Input("proctree-suggestions", "value")],
        prevent_initial_call=True,
    )
    def set_pid_from_selection(selected_pid):
        try:
            if selected_pid is None:
                return dash.no_update
            return int(selected_pid)
        except Exception:
            return dash.no_update

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
        events_list = events_list or []
        items = []
        for idx, e in enumerate(events_list):
            ts_txt = e.get("timestamp", "")
            name_txt = e.get("name", "event")
            items.append(
                html.Div(
                    [
                        html.Span(f"{ts_txt} — {name_txt}", className="event-text"),
                        html.Button(
                            "Delete",
                            id={"type": "event-delete", "index": idx},
                            n_clicks=0,
                            className="inline button tiny",
                            style={"marginLeft": "8px"},
                        ),
                    ],
                    className="event-item",
                    style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
                )
            )
        if not items:
            return html.Div("No events", style={"opacity": 0.7})
        return items

    def make_empty_fig(title):
        fig = go.Figure()
        fig.add_annotation(text=title, showarrow=False, yref="paper", y=0.5, xref="paper", x=0.5)
        fig.update_layout(margin=dict(l=30, r=10, t=30, b=30))
        return fig

    def convert_ts_for_display(ts_series, display_tz, display_tz_custom, data_tz):
        try:
            ts = pd.to_datetime(ts_series)
            # First, assign the correct base timezone to the stored timestamps (they are naive on disk)
            base = None
            if data_tz == "utc":
                base = "UTC"
            else:
                try:
                    base = tzlocal() if tzlocal is not None else None
                except Exception:
                    base = None

            if base is not None:
                try:
                    ts = ts.dt.tz_localize(base)
                except Exception:
                    # fallback: try scalar localize if Series.dt failed
                    try:
                        ts = pd.DatetimeIndex(ts).tz_localize(base)
                    except Exception:
                        pass

            # Convert to target display timezone and drop tz to keep axes naive
            try:
                target = None
                if display_tz == "local" and tzlocal is not None:
                    target = tzlocal()
                elif display_tz == "custom" and display_tz_custom and gettz is not None:
                    target = gettz(display_tz_custom)
                else:
                    target = "UTC"
                if target is not None:
                    ts = pd.DatetimeIndex(ts).tz_convert(target).tz_localize(None)
            except Exception:
                pass
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
        Output("display-tz-custom-wrap", "style"),
        [Input("display-tz", "value")],
    )
    def _toggle_custom_tz(display_value):
        if display_value == "custom":
            return {"display": "block"}
        return {"display": "none"}

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
            Input("datafile-store", "data"),
            Input("data-source-mode", "value"),
            Input("container-auto-top", "value"),
            Input("container-top-n", "value"),
            Input("container-select", "value"),
            Input("event-store", "data"),
            Input("event-auto-store", "data"),
            Input("event-display-options", "value"),
            Input("display-tz", "value"),
            Input("display-tz-custom", "value"),
            Input("data-tz", "value"),
        ],
    )
    def update_graphs(
        n,
        time_range,
        theme_value,
        smoothing_window,
        data_path,
        source_mode,
        auto_top,
        top_n,
        selected_manual,
        events_data,
        auto_events,
        event_display_options,
        display_tz,
        display_tz_custom,
        data_tz,
    ):
        # Load data (respect selected source mode)
        df = load_data(data_path or datafile, time_range, source_mode or "auto")
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
            convert_ts_for_display(df["timestamp"], display_tz, display_tz_custom, data_tz)
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

        # Process Tree handled by manual callback; nothing to build here

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
            f"Data source: {data_path or datafile} | Data TZ: UTC-naive (stored); Displayed in:"
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

    # Manual process tree callback
    @app.callback(
        [
            Output("proctree-totals", "children"),
            Output("proctree-procs-by-cmd", "figure"),
            Output("proctree-threads-by-cmd", "figure"),
            Output("proctree-tree-md", "children"),
            Output("proctree-last-summary", "data"),
        ],
        [Input("proctree-run", "n_clicks")],
        [State("proctree-pid", "value"), State("proctree-verbose", "value"), State("theme-toggle", "value")],
        prevent_initial_call=False,
    )
    def run_process_tree(n_clicks, proctree_pid, proctree_verbose_vals, theme_value):
        try:
            pid_val = int(proctree_pid) if proctree_pid is not None else None
        except Exception:
            pid_val = None
        verbose_flag = "verbose" in (proctree_verbose_vals or [])

        totals_text = "Enter a PID to inspect its process tree."
        fig_procs = go.Figure()
        fig_threads = go.Figure()
        tree_md = ""
        if pid_val is None or pid_val <= 0:
            return totals_text, fig_procs, fig_threads, tree_md, {}

        try:
            summary = get_process_tree_summary(pid_val, verbose=verbose_flag)
        except Exception as e:
            totals_text = f"Error: {e}"
            return totals_text, fig_procs, fig_threads, tree_md, {}

        totals = summary.get("totals", {})
        totals_text = (
            f"Total Processes: {totals.get('total_processes', 0)} • Total Threads: {totals.get('total_threads', 0)}"
        )
        agg = summary.get("aggregated_by_command", [])
        # Build bars
        if agg:
            cmds = [a.get("command", "") for a in agg]
            procs_counts = [a.get("processes", 0) for a in agg]
            threads_counts = [a.get("total_threads", 0) for a in agg]
            fig_procs.add_trace(go.Bar(x=cmds, y=procs_counts, name="Processes"))
            fig_threads.add_trace(go.Bar(x=cmds, y=threads_counts, name="Threads"))
            fig_procs.update_layout(title="Processes by Command", xaxis_title="Command", yaxis_title="Count")
            fig_threads.update_layout(title="Threads by Command", xaxis_title="Command", yaxis_title="Threads")
            apply_theme(fig_procs, theme_value)
            apply_theme(fig_threads, theme_value)
            style_minimal_figure(fig_procs, theme_value)
            style_minimal_figure(fig_threads, theme_value)
            fig_procs.update_layout(showlegend=False)
            fig_threads.update_layout(showlegend=False)
        # Build tree text (simple PPID-based indentation)
        plist = summary.get("processes", [])
        if totals.get("total_processes", 0) == 0:
            # Likely PID not visible in this namespace/container; provide a helpful hint.
            tree_text = (
                "No processes found. This PID may not be visible from this runtime (PID namespace). "
                "Try using the PID search above, or run the dashboard on the same host/namespace as the target process."
            )
            return (
                totals_text,
                fig_procs,
                fig_threads,
                f"```text\n{tree_text}\n```",
                summary,
            )
        by_ppid = {}
        for p in plist:
            by_ppid.setdefault(p.get("ppid"), []).append(p)
        for k in by_ppid:
            by_ppid[k] = sorted(by_ppid[k], key=lambda x: x.get("pid", 0))
        root = pid_val
        lines = []

        def walk(pid, indent):
            kids = by_ppid.get(pid, [])
            for i, child in enumerate(kids):
                prefix = "├─" if i < len(kids) - 1 else "└─"
                lines.append(
                    f"{indent}{prefix} {child.get('name','?')}({child.get('pid')}) [threads={child.get('threads',0)}]"
                )
                walk(child.get("pid"), indent + ("│  " if i < len(kids) - 1 else "   "))

        root_entry = next((p for p in plist if p.get("pid") == root), None)
        if root_entry:
            lines.insert(0, f"{root_entry.get('name','?')}({root}) [threads={root_entry.get('threads',0)}]")
            walk(root, "")
        tree_md = f"```text\n{chr(10).join(lines)}\n```"
        return totals_text, fig_procs, fig_threads, tree_md, summary

    # Theme-aware high-contrast styles for PID search controls
    @app.callback(
        Output("proctree-suggestions", "style"),
        [Input("theme-toggle", "value")],
    )
    def style_proctree_dropdown(theme_value):
        light = {
            "width": "420px",
            "display": "inline-block",
            "color": "#111",
            "backgroundColor": "#ffffff",
            "border": "1px solid #888",
        }
        dark = {
            "width": "420px",
            "display": "inline-block",
            "color": "#eee",
            "backgroundColor": "#222",
            "border": "1px solid #555",
        }
        return dark if theme_value == "dark" else light

    @app.callback(
        Output("proctree-search", "style"),
        [Input("theme-toggle", "value")],
    )
    def style_proctree_search(theme_value):
        base = {"width": "320px", "marginLeft": "6px", "marginRight": "8px"}
        if theme_value == "dark":
            base.update({"backgroundColor": "#222", "color": "#eee", "border": "1px solid #555"})
        else:
            base.update({"backgroundColor": "#fff", "color": "#111", "border": "1px solid #888"})
        return base

    # Toggle between text and graph tree containers
    @app.callback(
        [Output("proctree-cyto-container", "style"), Output("proctree-tree-text-container", "style")],
        [Input("proctree-view-mode", "value")],
    )
    def toggle_tree_view(view_mode):
        # Always show the Graph container when Graph is selected, even if dash-cytoscape
        # is not installed, so the fallback help message is visible.
        if view_mode == "graph":
            return {"display": "block"}, {"display": "none"}
        # default to text view
        return {"display": "none"}, {"display": "block"}

    # Build cytoscape elements from last summary
    if CY_AVAILABLE:

        @app.callback(
            Output("proctree-graph", "elements"),
            [Input("proctree-last-summary", "data")],
        )
        def build_cytoscape_elements(summary):
            try:
                plist = (summary or {}).get("processes", [])
                if not plist:
                    return []
                nodes = []
                edges = []
                pids = set()
                for p in plist:
                    pid = p.get("pid")
                    name = p.get("name") or "?"
                    threads = int(p.get("threads") or 0)
                    pids.add(pid)
                    nodes.append(
                        {
                            "data": {
                                "id": str(pid),
                                "label": f"{name}({pid}) t={threads}",
                                "threads": threads,
                            }
                        }
                    )
                for p in plist:
                    pid = p.get("pid")
                    ppid = p.get("ppid")
                    if ppid in pids and pid in pids and ppid is not None and pid is not None:
                        edges.append({"data": {"source": str(ppid), "target": str(pid)}})
                return nodes + edges
            except Exception:
                return []

        @app.callback(
            Output("proctree-graph", "stylesheet"),
            [Input("theme-toggle", "value")],
        )
        def cytoscape_stylesheet(theme_value):
            # map threads to size/color
            node_color_dark = "#4aa3ff"
            node_color_light = "#1f77b4"
            text_color_dark = "#e5e5e5"
            text_color_light = "#222222"
            edge_color_dark = "#888"
            edge_color_light = "#aaa"
            base = [
                {
                    "selector": "node",
                    "style": {
                        "label": "data(label)",
                        "font-size": 10,
                        "color": (text_color_dark if theme_value == "dark" else text_color_light),
                        "background-color": (node_color_dark if theme_value == "dark" else node_color_light),
                        "width": "mapData(threads, 0, 64, 20, 60)",
                        "height": "mapData(threads, 0, 64, 20, 60)",
                        "text-valign": "center",
                        "text-halign": "center",
                    },
                },
                {
                    "selector": "edge",
                    "style": {
                        "line-color": (edge_color_dark if theme_value == "dark" else edge_color_light),
                        "target-arrow-color": (edge_color_dark if theme_value == "dark" else edge_color_light),
                        "target-arrow-shape": "triangle",
                        "curve-style": "bezier",
                        "width": 1.5,
                    },
                },
            ]
            return base

        @app.callback(
            Output("proctree-node-details", "children"),
            [Input("proctree-graph", "tapNodeData")],
            [State("proctree-last-summary", "data")],
        )
        def show_node_details(tap_node, summary):
            try:
                if not tap_node:
                    return ""
                pid = int(tap_node.get("id"))
                plist = (summary or {}).get("processes", [])
                ent = next((p for p in plist if p.get("pid") == pid), None)
                if not ent:
                    return ""
                name = ent.get("name") or "?"
                ppid = ent.get("ppid")
                threads = ent.get("threads")
                return f"Selected: {name} ({pid}) — PPID={ppid}, Threads={threads}"
            except Exception:
                return ""

        # Force a layout re-run whenever elements change
        @app.callback(
            Output("proctree-graph", "layout"),
            [Input("proctree-graph", "elements")],
        )
        def refresh_cyto_layout(elements):
            return {"name": "breadthfirst", "directed": True}

    # Status helper under the graph container
    @app.callback(
        Output("proctree-graph-status", "children"),
        [Input("proctree-view-mode", "value"), Input("proctree-last-summary", "data")],
    )
    def update_graph_status(view_mode, summary):
        if view_mode != "graph":
            return ""
        if not CY_AVAILABLE:
            return "Graph view requires dash-cytoscape. Install with: pip install dash-cytoscape"
        plist = (summary or {}).get("processes", [])
        if not plist:
            return "No graph data yet. Click 'Inspect' after entering a valid PID."
        # count edges
        by_ppid = {}
        for p in plist:
            by_ppid.setdefault(p.get("ppid"), []).append(p)
        edge_count = sum(len(v) for k, v in by_ppid.items() if k is not None)
        return f"Graph ready: {len(plist)} node(s), {edge_count} edge(s). Tip: click a node to see details."

    # Snapshot current summary
    @app.callback(
        [Output("proctree-snapshot", "data"), Output("proctree-snapshot-status", "children")],
        [Input("proctree-snapshot-btn", "n_clicks")],
        [State("proctree-last-summary", "data")],
        prevent_initial_call=True,
    )
    def take_snapshot(n_clicks, summary):
        try:
            if not summary or not summary.get("processes"):
                return dash.no_update, "No current tree to snapshot. Run Inspect first."
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            snap = {"timestamp": ts, "summary": summary}
            return snap, f"Snapshot captured at {ts}."
        except Exception:
            return dash.no_update, "Failed to capture snapshot."

    # Diff current summary to snapshot
    @app.callback(
        Output("proctree-diff-result", "children"),
        [Input("proctree-diff-btn", "n_clicks")],
        [State("proctree-snapshot", "data"), State("proctree-last-summary", "data")],
        prevent_initial_call=True,
    )
    def diff_to_snapshot(n_clicks, snapshot, current):
        try:
            if not snapshot or not snapshot.get("summary"):
                return "No snapshot available. Click 'Take Snapshot' after Inspect."
            snap = snapshot.get("summary") or {}
            snap_plist = snap.get("processes", [])
            cur_plist = (current or {}).get("processes", [])
            snap_by_pid = {p.get("pid"): p for p in snap_plist}
            cur_by_pid = {p.get("pid"): p for p in cur_plist}
            added = sorted([pid for pid in cur_by_pid.keys() if pid not in snap_by_pid])
            removed = sorted([pid for pid in snap_by_pid.keys() if pid not in cur_by_pid])
            changed = []
            for pid in set(cur_by_pid.keys()).intersection(snap_by_pid.keys()):
                t0 = int(snap_by_pid[pid].get("threads") or 0)
                t1 = int(cur_by_pid[pid].get("threads") or 0)
                if t0 != t1:
                    changed.append((pid, t0, t1))
            lines = []
            lines.append(f"Added: {len(added)}")
            if added:
                lines.extend([f"  + {pid}" for pid in added[:50]])
                if len(added) > 50:
                    lines.append("  …")
            lines.append(f"Removed: {len(removed)}")
            if removed:
                lines.extend([f"  - {pid}" for pid in removed[:50]])
                if len(removed) > 50:
                    lines.append("  …")
            lines.append(f"Thread changes: {len(changed)}")
            if changed:
                for pid, t0, t1 in changed[:50]:
                    lines.append(f"  ~ {pid}: {t0} -> {t1}")
                if len(changed) > 50:
                    lines.append("  …")
            return dcc.Markdown("```text\n" + "\n".join(lines) + "\n```")
        except Exception:
            return "Diff failed."

    # Populate container dropdown options dynamically
    @app.callback(
        Output("container-select", "options"),
        [Input("interval-component", "n_intervals"), Input("time-range", "value"), Input("datafile-store", "data")],
    )
    def update_container_options(n, time_range, data_path):
        try:
            df = load_data(data_path or datafile, time_range)
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

    # Clientside callback to sync CSS theme via data-theme attribute
    app.clientside_callback(
        """
        function(theme){
            var isLight = (theme === 'light');
            var body = document.body, html = document.documentElement;
            if (isLight) { body.setAttribute('data-theme','light'); html.setAttribute('data-theme','light'); }
            else { body.removeAttribute('data-theme'); html.removeAttribute('data-theme'); }
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
            Input({"type": "event-delete", "index": ALL}, "n_clicks"),
        ],
        [
            State("event-name", "value"),
            State("event-date", "date"),
            State("event-time", "value"),
            State("event-store", "data"),
            State("event-auto-store", "data"),
            State("time-range", "value"),
            State("display-tz", "value"),
            State("display-tz-custom", "value"),
            State("datafile-store", "data"),
        ],
    )
    def manage_events(
        n_add,
        n_now,
        n_clear,
        upload_contents,
        delete_clicks,
        name,
        date_val,
        time_val,
        data,
        auto_data,
        time_range,
        display_tz,
        display_tz_custom,
        data_path,
    ):
        data = data or []
        triggered = getattr(dash, "callback_context", None)
        trig_id = ""
        if triggered and triggered.triggered:
            trig_id = triggered.triggered[0]["prop_id"].split(".")[0]

        # Helper to get latest timestamp from datafile
        def latest_data_ts():
            df = load_data(data_path or datafile, time_range)
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

        # Handle per-item delete via pattern-matching id
        if trig_id.startswith("{") and "event-delete" in trig_id:
            try:
                obj = json.loads(trig_id)
                idx = int(obj.get("index", -1))
            except Exception:
                idx = -1
            if 0 <= idx < len(data):
                data.pop(idx)
            return data, render_event_list(data)

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
                # Combine date + time into a single wall time in selected display tz
                if date_val:
                    t_str = (time_val or "00:00:00").strip()
                    # normalize time format HH:MM[:SS]
                    parts = t_str.split(":")
                    if len(parts) == 1:
                        t_str = f"{parts[0]}:00:00"
                    elif len(parts) == 2:
                        t_str = f"{parts[0]}:{parts[1]}:00"
                    try:
                        ts = from_text_to_utc_naive(f"{date_val} {t_str}")
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
            return new_data, render_event_list(new_data)

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
                return new_data, render_event_list(new_data)
            except Exception:
                return data, render_event_list(data)

        # default: just render existing
        return data, render_event_list(data)

    # Auto event triggers (watch points)
    @app.callback(
        [Output("event-auto-store", "data"), Output("watch-state-store", "data")],
        [Input("interval-component", "n_intervals"), Input("clear-auto-events-btn", "n_clicks")],
        [
            State("watch-enable", "value"),
            State("watch-cpu", "value"),
            State("watch-mem", "value"),
            State("watch-threads", "value"),
            State("watch-procs", "value"),
            State("event-auto-store", "data"),
            State("watch-state-store", "data"),
            State("datafile-store", "data"),
            State("time-range", "value"),
        ],
    )
    def apply_watch_points(
        n,
        n_clear_auto,
        enable_vals,
        thr_cpu,
        thr_mem,
        thr_thr,
        thr_prc,
        auto_events,
        watch_state,
        data_path,
        time_range,
    ):
        auto_events = auto_events or []
        watch_state = watch_state or {"cpu": False, "mem": False, "threads": False, "procs": False}
        enabled = set(enable_vals or [])
        # If clear button triggered, reset auto events and watch state immediately
        triggered = getattr(dash, "callback_context", None)
        trig_id = ""
        if triggered and triggered.triggered:
            trig_id = triggered.triggered[0]["prop_id"].split(".")[0]
        if trig_id == "clear-auto-events-btn":
            # Clear auto events and set watch_state according to current readings
            try:
                df_now = load_data(data_path or datafile, time_range)
            except Exception:
                return [], watch_state
            if df_now.empty or "timestamp" not in df_now.columns:
                return [], watch_state
            latest_now = df_now.iloc[-1]
            # Compute over-threshold flags to avoid immediate re-trigger if still over
            new_state = {"cpu": False, "mem": False, "threads": False, "procs": False}
            try:
                if "cpu" in enabled and thr_cpu is not None:
                    cpu_cols = [c for c in df_now.columns if c.startswith("cpu_") and c.endswith("_utilization")]
                    cpu_avg = float(latest_now[cpu_cols].mean()) if cpu_cols else None
                    new_state["cpu"] = cpu_avg is not None and cpu_avg >= float(thr_cpu)
            except Exception:
                pass
            try:
                if "mem" in enabled and thr_mem is not None:
                    used = float(latest_now.get("sys_used", 0.0))
                    total = float(latest_now.get("sys_total", 0.0))
                    mem_pct = (used / total * 100.0) if total > 0 else None
                    new_state["mem"] = mem_pct is not None and mem_pct >= float(thr_mem)
            except Exception:
                pass
            try:
                if "threads" in enabled and thr_thr is not None:
                    thr_count = float(latest_now.get("system_thread_count", 0))
                    new_state["threads"] = thr_count >= float(thr_thr)
            except Exception:
                pass
            try:
                if "procs" in enabled and thr_prc is not None:
                    prc_count = float(latest_now.get("system_process_count", 0))
                    new_state["procs"] = prc_count >= float(thr_prc)
            except Exception:
                pass
            return [], new_state
        try:
            df = load_data(data_path or datafile, time_range)
        except Exception:
            return auto_events, watch_state
        if df.empty or "timestamp" not in df.columns:
            return auto_events, watch_state
        latest = df.iloc[-1]
        ts = latest.get("timestamp")
        try:
            ts_norm = pd.to_datetime(normalize_ts(ts)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_norm = str(ts)

        new_events = []
        # CPU avg threshold
        if "cpu" in enabled and thr_cpu is not None:
            try:
                cpu_cols = [c for c in df.columns if c.startswith("cpu_") and c.endswith("_utilization")]
                cpu_avg = float(latest[cpu_cols].mean()) if cpu_cols else None
                if cpu_avg is not None:
                    over = cpu_avg >= float(thr_cpu)
                    if over and not watch_state.get("cpu", False):
                        new_events.append(
                            {
                                "name": f"CPU% > {float(thr_cpu):.0f} (now {cpu_avg:.1f})",
                                "timestamp": ts_norm,
                            }
                        )
                    watch_state["cpu"] = over
            except Exception:
                pass
        # Memory percent threshold
        if "mem" in enabled and thr_mem is not None:
            try:
                used = float(latest.get("sys_used", 0.0))
                total = float(latest.get("sys_total", 0.0))
                mem_pct = (used / total * 100.0) if total > 0 else None
                if mem_pct is not None:
                    over = mem_pct >= float(thr_mem)
                    if over and not watch_state.get("mem", False):
                        new_events.append(
                            {
                                "name": f"Mem% > {float(thr_mem):.0f} (now {mem_pct:.1f})",
                                "timestamp": ts_norm,
                            }
                        )
                    watch_state["mem"] = over
            except Exception:
                pass
        # Threads threshold
        if "threads" in enabled and thr_thr is not None:
            try:
                thr_count = float(latest.get("system_thread_count", 0))
                over = thr_count >= float(thr_thr)
                if over and not watch_state.get("threads", False):
                    new_events.append(
                        {
                            "name": f"Threads > {float(thr_thr):.0f} (now {thr_count:.0f})",
                            "timestamp": ts_norm,
                        }
                    )
                watch_state["threads"] = over
            except Exception:
                pass
        # Processes threshold
        if "procs" in enabled and thr_prc is not None:
            try:
                prc_count = float(latest.get("system_process_count", 0))
                over = prc_count >= float(thr_prc)
                if over and not watch_state.get("procs", False):
                    new_events.append(
                        {
                            "name": f"Processes > {float(thr_prc):.0f} (now {prc_count:.0f})",
                            "timestamp": ts_norm,
                        }
                    )
                watch_state["procs"] = over
            except Exception:
                pass

        if new_events:
            return (auto_events + new_events), watch_state
        return auto_events, watch_state

    # ----------------------------
    # Tracer integration
    # ----------------------------
    import threading

    _tracer_lock = threading.Lock()
    _tracer_obj = {"tracer": None, "thread": None}  # type: ignore

    def _is_running():
        t = _tracer_obj.get("thread")
        return t is not None and t.is_alive()

    @app.callback(
        [Output("tracer-status", "children"), Output("datafile-store", "data")],
        [
            Input("tracer-start-btn", "n_clicks"),
            Input("tracer-stop-btn", "n_clicks"),
            Input("tracer-reset-btn", "n_clicks"),
            Input("tracer-snapshot-btn", "n_clicks"),
        ],
        [
            State("tracer-output-path", "value"),
            State("tracer-sample-interval", "value"),
            State("tracer-write-interval", "value"),
            State("tracer-options", "value"),
            State("datafile-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def manage_tracer(n_start, n_stop, n_reset, n_snap, out_path, sample_iv, write_iv, options, current_path):
        trig = getattr(dash, "callback_context", None)
        trig_id = ""
        if trig and trig.triggered:
            trig_id = trig.triggered[0]["prop_id"].split(".")[0]

        out_path = (out_path or current_path or datafile).strip()
        enable_gpu = (options or []) and ("gpu" in (options or []))
        enable_docker = (options or []) and ("docker" in (options or []))
        use_utc = (options or []) and ("utc" in (options or []))

        if trig_id == "tracer-start-btn":
            with _tracer_lock:
                if _is_running():
                    return (f"Tracer already running -> {out_path}", out_path)
                tracer = SystemTracer(
                    sample_interval=float(sample_iv or 5.0),
                    # Keep data in memory when running from dashboard; only snapshot persists
                    write_interval=0.0,
                    output_file=out_path,
                    enable_gpu=bool(enable_gpu),
                    enable_docker=bool(enable_docker),
                    use_utc=bool(use_utc),
                    write_final=False,
                )
                th = threading.Thread(target=tracer.run, kwargs={"duration": None, "verbose": False}, daemon=True)
                _tracer_obj["tracer"] = tracer
                _tracer_obj["thread"] = th
                th.start()
            return (f"Tracer started -> {out_path}", out_path)

        if trig_id == "tracer-stop-btn":
            with _tracer_lock:
                tracer = _tracer_obj.get("tracer")
                th = _tracer_obj.get("thread")
                if tracer is not None:
                    try:
                        tracer.stop()
                    except Exception:
                        pass
                if th is not None:
                    try:
                        th.join(timeout=2.0)
                    except Exception:
                        pass
                _tracer_obj["tracer"] = None
                _tracer_obj["thread"] = None
            return ("Tracer stopped.", out_path)

        if trig_id == "tracer-reset-btn":
            with _tracer_lock:
                tracer = _tracer_obj.get("tracer")
                if tracer is not None:
                    try:
                        tracer.reset()
                    except Exception:
                        pass
            return ("Tracer buffer reset.", out_path)

        if trig_id == "tracer-snapshot-btn":
            with _tracer_lock:
                tracer = _tracer_obj.get("tracer")
                if tracer is not None:
                    try:
                        tracer.set_output_file(out_path)
                        path = tracer.snapshot(out_path)
                        return (f"Snapshot saved to {path}", path)
                    except Exception as e:
                        return (f"Snapshot failed: {e}", out_path)
            # If tracer not running, still write empty/new file to path
            try:
                # Create a unique suffixed filename if destination exists
                base_path = (out_path or "system_monitor.parquet").strip()
                root, ext = os.path.splitext(base_path)
                if not ext:
                    ext = ".parquet"
                    root = base_path  # original base without extension
                    base_path = base_path + ext
                path = base_path
                if os.path.exists(path):
                    idx = 0
                    while True:
                        candidate = f"{root}_{idx}{ext}"
                        if not os.path.exists(candidate):
                            path = candidate
                            break
                        idx += 1
                pd.DataFrame([]).to_parquet(path)
                return (f"Snapshot (empty) saved to {path}", path)
            except Exception as e:
                return (f"Snapshot failed: {e}", out_path)

        return ("", out_path)

    # Start the server
    print(f"Starting dashboard server on http://{host}:{port}/")
    print(f"Using data file: {datafile}")
    print(f"Refresh interval: {interval} seconds")
    print("Press Ctrl+C to stop the server")
    app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()  # This invokes the Click command
