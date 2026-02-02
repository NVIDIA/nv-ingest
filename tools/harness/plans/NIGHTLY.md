# Nightly Benchmark Runner

Automated benchmarks with Slack reporting and historical tracking.

## Quick Start

```bash
cd tools/harness

export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
uv run nv-ingest-harness-nightly

# Options
uv run nv-ingest-harness-nightly --skip-slack        # No Slack
uv run nv-ingest-harness-nightly --skip-fresh-start  # Use running services
uv run nv-ingest-harness-nightly --dry-run           # Show config only
uv run nv-ingest-harness-nightly --note "Testing new embedding model"
```

## Configuration

### `nightly_config.yaml`

```yaml
# Benchmark runs
runs:
  # E2E tests (ingestion only, no recall)
  e2e:
    - bo20  # Latency benchmark (20 docs, 496 pages)

  # E2E + Recall tests (ingestion + recall evaluation)
  e2e_recall:
    - bo767        # Throughput + recall (767 docs)
    - earnings     # Earnings consulting dataset
    - financebench # Financebench full dataset

# Recall configuration (applies to all e2e_recall runs)
recall:
  reranker_mode: both  # Options: "none", "with", "both"
  top_k: 10

# Sinks configuration
sinks:
  slack:
    enabled: true
    # webhook_url: Set via SLACK_WEBHOOK_URL environment variable

  history:
    enabled: true
    # db_path: Set via HARNESS_HISTORY_DB environment variable
    retention_days: 90  # Auto-prune runs older than 90 days

# Infrastructure
infrastructure:
  fresh_start: true  # Run nv-clean && nv-start before benchmarks
  readiness_timeout: 600  # Seconds to wait for services
```

Datasets are defined in `test_configs.yaml`. The nightly runner looks up paths and extraction settings by dataset name.

## Sinks

Pluggable modules for processing results:

```python
class Sink(ABC):
    def __init__(self, sink_config: dict[str, Any]): ...
    def initialize(self, session_name: str, env_data: dict[str, Any]) -> None: ...
    def process_result(self, result: dict[str, Any], entry_config: dict[str, Any] | None = None) -> None: ...
    def finalize(self) -> None: ...
```

**SlackSink** - Posts to Slack via `SLACK_WEBHOOK_URL`

**HistorySink** - SQLite storage at `history.db` (override via `HARNESS_HISTORY_DB`)

### Adding Custom Sinks

```python
# sinks/my_sink.py
from nv_ingest_harness.sinks.base import Sink

class MySink(Sink):
    def __init__(self, sink_config):
        self.enabled = sink_config.get("enabled", True)
    
    def initialize(self, session_name, env_data): pass
    def process_result(self, result, entry_config=None): pass
    def finalize(self): pass
```

Register in `sinks/__init__.py`, add config to `nightly_config.yaml`.

## Baselines

Define pass/fail requirements in `reporting/baselines.py`:

```python
DATASET_BASELINES = {
    "bo20": {
        "result_count": {"expected": 20, "required": True},
        "total_pages": {"expected": 496, "required": True},
        "ingestion_time_s": {"max": 70, "warn_threshold": 60},
        "pages_per_second": {"min": 7.0},
        "failure_count": {"expected": 0, "required": True},
    },
    "bo767": {
        "result_count": {"expected": 767, "required": True},
        "pages_per_second": {"min": 15.0},
        "failure_count": {"expected": 0, "required": True},
        "recall_multimodal_@5_no_reranker": {"min": 0.75},
        "recall_multimodal_@5_reranker": {"min": 0.80},
    },
}
```

| Type | Description |
|------|-------------|
| `expected` | Exact match |
| `min` / `max` | Bounds |
| `warn_threshold` | Warning trigger (with `max`) |
| `required` | Blocks overall success if failed |

## Historical Tracking

```python
from nv_ingest_harness.sinks.history import HistorySink

runs = HistorySink.get_recent_runs("bo20", limit=10)
trend = HistorySink.get_trend("bo20", "pages_per_second", days=30)
```

### Schema

```sql
CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    session_name TEXT,
    dataset TEXT NOT NULL,
    git_commit TEXT,
    result_count INTEGER,
    failure_count INTEGER,
    ingestion_time_s REAL,
    pages_per_second REAL,
    total_pages INTEGER,
    text_chunks INTEGER,
    table_chunks INTEGER,
    chart_chunks INTEGER,
    retrieval_time_s REAL,
    recall_at_5 REAL,
    recall_at_5_reranker REAL,
    requirements_met INTEGER,
    raw_json TEXT
);
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--config PATH` | Custom config file (default: `nightly_config.yaml`) |
| `--skip-slack` | Disable Slack posting |
| `--skip-history` | Disable SQLite storage |
| `--skip-fresh-start` | Use running services instead of restarting |
| `--dry-run` | Show configuration without executing |
| `--replay PATH` | Replay results from artifact dir(s) to Slack (can specify multiple) |

## Replay

If Slack posting fails (e.g., expired webhook), replay results after fixing:

```bash
uv run nv-ingest-harness-nightly \
  --replay artifacts/bo20_20260109_053712_UTC \
  --replay artifacts/bo767_20260109_053814_UTC
```

## CI/CD

```yaml
# GitHub Actions
jobs:
  nightly:
    runs-on: self-hosted
    env:
      SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
    steps:
      - uses: actions/checkout@v4
      - run: cd tools/harness && uv sync && uv run nv-ingest-harness-nightly
```

```bash
# Cron (2 AM daily)
0 2 * * * cd /path/to/tools/harness && source ~/setup_env.sh && \
  SLACK_WEBHOOK_URL="..." uv run nv-ingest-harness-nightly >> /var/log/nightly.log 2>&1
```

## Slack Output

```
OVERALL STATUS  ✅ success (4/4)
-    bo20       ✅ success
-    bo767      ✅ success
-    earnings   ✅ success
-    financebench ✅ success

ENVIRONMENT
-    hostname   dgx-h100-01
-    gpu        8x NVIDIA H100
-    git_commit a1b2c3d4

RESULTS
bo20            ✅ success
-    ingestion_time_s    52.31s (00m : 52.31s)
-    pages_per_second    9.48
-    result_count        20/20
-    total_pages         496/496
-    All requirements met ✅

bo767           ✅ success
-    ingestion_time_s    1847.23s (30m : 47.23s)
-    pages_per_second    29.63
-    result_count        767/767
-    recall_multimodal_@5_no_reranker  0.840
-    recall_multimodal_@5_reranker     0.912
-    All requirements met ✅
```
