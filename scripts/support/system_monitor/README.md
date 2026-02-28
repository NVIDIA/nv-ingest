# System Monitor

A lightweight system tracing and dashboard toolkit with two primary workflows:

1) Real-time monitoring: start/stop tracing directly from the dashboard UI and visualize live metrics.
2) Offline exploration: collect Parquet/CSV on one system and explore it on another system without a running tracer.

This package contains:
- `system_monitor.py`: Dash dashboard (UI, charts, event annotations, process tree).
- `system_tracer.py`: Tracer library and CLI (collects metrics, writes Parquet atomically).


## Requirements

- Python 3.9+
- Core: `pandas`, `psutil`, `plotly`, `dash`, `click`
- Optional:
  - `pyarrow` (preferred) or `fastparquet` for Parquet
  - `dash-cytoscape` for process tree graph view (text view works without it)
  - `docker` Python package and a local Docker daemon for container metrics
  - `pynvml` for NVIDIA GPU metrics (if NVIDIA drivers are present)

Install common dependencies (example):

```bash
pip install pandas psutil plotly dash click pyarrow
# Optional extras
pip install fastparquet dash-cytoscape docker pynvml
```

Environment setup
- Ensure the `system_monitor` package is importable at the top level. If you are running from the repo without installing, set PYTHONPATH so `python -m system_monitor` works:

```bash
export PYTHONPATH=$(pwd)/scripts/support:$PYTHONPATH
```

Alternatively, install the package into your environment (recommended for reuse). If you maintain a packaging config, use `pip install -e .` at the repo root.


## Quickstart A: Real-time Monitoring (single machine)

Launch the dashboard on the machine you want to monitor and control tracing from the UI.

```bash
python -m system_monitor --datafile system_monitor.parquet --port 8050
```

- Open the URL printed (default http://0.0.0.0:8050).
- In the left sidebar under Tracing:
  - Set Output Parquet Path (defaults to `system_monitor.parquet`).
  - Adjust Sampling and Write Interval.
  - Toggle Enable GPU / Enable Docker as needed.
  - Click Start to begin live tracing. Click Stop to end. Click Snapshot Now to force an immediate write.
- The graphs update as data is written. You can:
  - Switch theme (Light/Dark)
  - Change time range and smoothing
  - Add/import events and toggle event markers
  - Inspect the process tree (text or Cytoscape if installed)

Notes
- Writes are atomic (tmp + replace) to avoid partial reads.
- If `pyarrow` is unavailable, the tracer falls back to `fastparquet` via pandas.
- Docker/GPU stats are optional and automatically disabled if their deps/daemons are unavailable.
- Timezones: Graphs default to Local display time. You can switch graphs to UTC or a custom IANA zone. Use the "Data timezone (source)" selector if your data was recorded in UTC.


## Quickstart B: Offline Data Collection and Exploration (two machines)

Use one system to collect data (headless), then transfer the file to another system for exploration in the dashboard.

1) Collect on Source (headless CLI):

```bash
# Run continuously until interrupted (local timestamps by default)
python -m system_monitor.system_tracer run \
  --output /tmp/system_monitor.parquet \
  --sample-interval 2 \
  --write-interval 10

# Record timestamps in UTC instead of local
python -m system_monitor.system_tracer run \
  --output /tmp/system_monitor_utc.parquet \
  --sample-interval 2 \
  --write-interval 10 \
  --utc

# Or run for a fixed duration (e.g., 5 minutes)
python -m system_monitor.system_tracer run \
  --output /tmp/system_monitor.parquet \
  --sample-interval 2 \
  --write-interval 10 \
  --duration 300
```

2) Transfer the Parquet/CSV to your analysis machine:

```bash
scp source:/tmp/system_monitor.parquet ./
```

3) Explore on Destination (no tracer needed):

```bash
python -m system_monitor --datafile ./system_monitor.parquet --port 8050
```

- The dashboard loads the provided file and renders metrics.
- Tracing controls in the UI only affect the local machine; they are independent of the loaded file.


## Process Tree Inspection

From dashboard:
- Go to the Process Tree view.
- Enter a PID or use the PID finder to search by name/command.
- Click Inspect to load the tree and thread counts.
- If `dash-cytoscape` is installed, toggle to Graph view for a visual tree with node details.

From CLI:

```bash
python -m system_monitor.system_tracer proctree <PID>
```


## Events and Timezones

- Add events via the sidebar date/time picker.
- Import CSV with two columns: `event,timestamp`. Timestamps are normalized internally.
- Display timezone can be set to Local, UTC, or a custom IANA zone. Event markers and data align accordingly.


## Tips & Troubleshooting

- Parquet engines: Install `pyarrow` for best compatibility. `fastparquet` is used as a fallback.
- Docker metrics: Requires the Docker daemon running and the `docker` Python package. If unavailable, container graphs will be empty.
- GPU metrics: Requires `pynvml` and NVIDIA drivers. If unavailable, GPU graphs will be empty.
- Assets override: Set `SYSTEM_MONITOR_ASSETS` to point to a custom assets directory if desired.
- Permissions: Some process/thread info may require elevated privileges; run as a user with sufficient permissions if you see AccessDenied errors.


## Programmatic API (optional)

Collect a one-off snapshot in Python:

```python
from system_monitor.system_tracer import collect_system_snapshot
snap = collect_system_snapshot(enable_gpu=False, enable_docker=False)
```

Run tracer to Parquet in-process:

```python
from system_monitor.system_tracer import monitor_to_parquet
monitor_to_parquet(output_file="system_monitor.parquet", sample_interval=2, write_interval=10)
```

UTC vs Local
- Tracer defaults to local timestamps. Pass `--utc` to record timestamps in UTC.
- Dashboard defaults to Local display time. Use the "Display timezone" control to switch to UTC or a custom IANA zone, and "Data timezone (source)" to inform the dashboard whether your stored data timestamps are Local or UTC.


## License

Internal project module; follow repository licensing and contribution guidelines.
