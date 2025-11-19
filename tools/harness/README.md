# nv-ingest Integration Testing Framework

A configurable, dataset-agnostic testing framework for end-to-end validation of nv-ingest pipelines. This framework uses structured YAML configuration for type safety, validation, and parameter management.

## Quick Start

### Prerequisites
- Docker and Docker Compose running
- Python environment with nv-ingest-client
- Access to test datasets

### Run Your First Test

```bash
# 1. Navigate to the tests directory
cd tools/harness/

# 2. Install dependencies
uv sync

# 3. Run with a pre-configured dataset (assumes services are running)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# Or use a custom path that uses the "active" configuration
uv run nv-ingest-harness-run --case=e2e --dataset=/path/to/your/data

# With managed infrastructure (starts/stops services)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed
```

## Configuration System

### YAML Configuration (`test_configs.yaml`)

The framework uses a structured YAML file for all test configuration. Configuration is organized into logical sections:

#### Active Configuration

The `active` section contains your current test settings. Edit these values directly for your test runs:

```yaml
active:
  # Dataset
  dataset_dir: /path/to/your/dataset
  test_name: null  # Auto-generated if null
  
  # API Configuration
  api_version: v1  # v1 or v2
  pdf_split_page_count: null  # V2 only: pages per chunk (null = default 32)
  
  # Infrastructure
  hostname: localhost
  readiness_timeout: 600
  profiles: [retrieval, table-structure]
  
  # Runtime
  sparse: true
  gpu_search: false
  embedding_model: auto
  
  # Extraction
  extract_text: true
  extract_tables: true
  extract_charts: true
  extract_images: false
  extract_infographics: true
  text_depth: page
  table_output_format: markdown
  
  # Pipeline (optional steps)
  enable_caption: false
  enable_split: false
  split_chunk_size: 1024
  split_chunk_overlap: 150
  
  # Storage
  spill_dir: /tmp/spill
  artifacts_dir: null
  collection_name: null
```

#### Pre-Configured Datasets

Each dataset includes its path, extraction settings, and recall evaluator in one place:

```yaml
datasets:
  bo767:
    path: /raid/jioffe/bo767
    extract_text: true
    extract_tables: true
    extract_charts: true
    extract_images: false
    extract_infographics: false
    recall_dataset: bo767  # Evaluator for recall testing
  
  bo20:
    path: /raid/jioffe/bo20
    extract_text: true
    extract_tables: true
    extract_charts: true
    extract_images: true
    extract_infographics: false
    recall_dataset: null  # bo20 does not have recall
  
  earnings:
    path: /raid/jioffe/earnings_conusulting
    extract_text: true
    extract_tables: true
    extract_charts: true
    extract_images: false
    extract_infographics: false
    recall_dataset: earnings  # Evaluator for recall testing
```

**Automatic Configuration**: When you use `--dataset=bo767`, the framework automatically:
- Sets the dataset path
- Applies the correct extraction settings (text, tables, charts, images, infographics)
- Configures the recall evaluator (if applicable)

**Usage:**
```bash
# Single dataset - configs applied automatically
uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# Multiple datasets (sweeping) - each gets its own config
uv run nv-ingest-harness-run --case=e2e --dataset=bo767,earnings,bo20

# Custom path still works (uses active section config)
uv run nv-ingest-harness-run --case=e2e --dataset=/custom/path
```

**Dataset Extraction Settings:**

| Dataset | Text | Tables | Charts | Images | Infographics | Recall |
|---------|------|--------|--------|--------|--------------|--------|
| `bo767` | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| `earnings` | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| `bo20` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| `financebench` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `single` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

### Configuration Precedence

Settings are applied in order of priority:

**Environment variables > Dataset-specific config (path + extraction + recall_dataset) > YAML active config**

**Note**: CLI arguments are only used for runtime decisions (which test to run, which dataset, execution mode). All configuration values come from YAML or environment variables.

Example:
```bash
# YAML active section has api_version: v1
# Dataset bo767 has extract_images: false
# Override via environment variable (highest priority)
EXTRACT_IMAGES=true API_VERSION=v2 uv run nv-ingest-harness-run --case=e2e --dataset=bo767
# Result: Uses bo767 path, but extract_images=true (env override) and api_version=v2 (env override)
```

**Precedence Details:**
1. **Environment variables** - Highest priority, useful for CI/CD overrides
2. **Dataset-specific config** - Applied automatically when using `--dataset=<name>`
   - Includes: path, extraction settings, recall_dataset
   - Only applies if dataset is defined in `datasets` section
3. **YAML active config** - Base configuration, used as fallback

### Configuration Options Reference

#### Core Options
- `dataset_dir` (string, required): Path to dataset directory
- `test_name` (string): Test identifier (auto-generated if null)
- `api_version` (string): API version - `v1` or `v2`
- `pdf_split_page_count` (integer): PDF splitting page count (V2 only, 1-128)

#### Extraction Options
- `extract_text`, `extract_tables`, `extract_charts`, `extract_images`, `extract_infographics` (boolean): Content extraction toggles
- `text_depth` (string): Text extraction granularity - `page`, `document`, `block`, `line`, etc.
- `table_output_format` (string): Table output format - `markdown`, `html`, `latex`, `pseudo_markdown`, `simple`

#### Pipeline Options
- `enable_caption` (boolean): Enable image captioning
- `enable_split` (boolean): Enable text chunking
- `split_chunk_size` (integer): Chunk size for text splitting
- `split_chunk_overlap` (integer): Overlap for text splitting

#### Infrastructure Options
- `hostname` (string): Service hostname
- `readiness_timeout` (integer): Docker startup timeout in seconds
- `profiles` (list): Docker compose profiles

#### Runtime Options
- `sparse` (boolean): Use sparse embeddings
- `gpu_search` (boolean): Use GPU for search
- `embedding_model` (string): Embedding model name (`auto` for auto-detection)
- `llm_summarization_model` (string): LLM model for summarization (used by `e2e_with_llm_summary`)

#### Storage Options
- `spill_dir` (string): Temporary processing directory
- `artifacts_dir` (string): Test output directory (auto-generated if null)
- `collection_name` (string): Milvus collection name (auto-generated as `{test_name}_multimodal` if null, deterministic - no timestamp)

#### Trace Capture Options
- `enable_traces` (boolean): When `true`, the `e2e` case requests parent-level trace payloads from the V2 API (see `docs/docs/extraction/v2-api-guide.md`) and emits per-stage summaries.
- `trace_output_dir` (string or null): Optional override for where raw trace JSON files are written. Defaults to `<artifact_dir>/traces` when null.

When traces are enabled:

1. Each processed document gets a raw payload written to `trace_output_dir/<index>_<source>.json` that contains the complete `"trace::"` dictionary plus a small stage summary. These files match the structure shown in `scripts/private_local/trace.json`.
2. `results.json` gains a `trace_summary` block that aggregates resident-time seconds per stage and lists total compute seconds per document. Example:

```json
"trace_summary": {
  "documents": 20,
  "output_dir": "/raid/.../scripts/tests/artifacts/bo20_20251119_183733_UTC/traces",
  "stage_totals": {
    "pdf_extractor": {
      "documents": 20,
      "total_resident_s": 32.18,
      "avg_resident_s": 1.61,
      "max_resident_s": 2.04,
      "min_resident_s": 1.42
    }
  },
  "document_totals": [
    {"document_index": 0, "source_id": "doc001.pdf", "total_resident_s": 1.58},
    {"document_index": 1, "source_id": "doc002.pdf", "total_resident_s": 1.64}
    // ...
  ]
}
```

**Enable via YAML or environment variables:**

- YAML: set `enable_traces: true` (and optionally `trace_output_dir`) in the `active` section.
- Environment: `ENABLE_TRACES=true TRACE_OUTPUT_DIR=/raid/jioffe/traces python run.py --case=e2e --dataset=bo20`

> ℹ️ Tracing is most valuable with V2 + PDF splitting enabled. Follow the guidance in `docs/docs/extraction/v2-api-guide.md` to ensure `api_version=v2` and `pdf_split_page_count` are configured.

### Valid Configuration Values

**text_depth**: `block`, `body`, `document`, `header`, `line`, `nearby_block`, `other`, `page`, `span`

**table_output_format**: `html`, `image`, `latex`, `markdown`, `pseudo_markdown`, `simple`

**api_version**: `v1`, `v2`

Configuration is validated on load with helpful error messages.

## Running Tests

### Basic Usage

```bash
# Run with default YAML configuration (assumes services are running)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# With document-level analysis
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --doc-analysis

# With managed infrastructure (starts/stops services)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed
```

### Dataset Sweeping

Run multiple datasets in a single command - each dataset automatically gets its native extraction configuration:

```bash
# Sweep multiple datasets
uv run nv-ingest-harness-run --case=e2e --dataset=bo767,earnings,bo20

# Each dataset runs sequentially with its own:
# - Extraction settings (from dataset config)
# - Artifact directory (timestamped per dataset)
# - Results summary at the end

# With managed infrastructure (services start once, shared across all datasets)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767,earnings,bo20 --managed

# E2E+Recall sweep (each dataset ingests then evaluates recall)
uv run nv-ingest-harness-run --case=e2e_recall --dataset=bo767,earnings

# Recall-only sweep (evaluates existing collections)
uv run nv-ingest-harness-run --case=recall --dataset=bo767,earnings
```

**Sweep Behavior:**
- Services start once (if `--managed`) before the sweep
- Each dataset gets its own artifact directory
- Each dataset automatically applies its extraction config from `datasets` section
- Summary printed at end showing success/failure for each dataset
- Services stop once at end (unless `--keep-up`)

### Using Environment Variables

```bash
# Override via environment (useful for CI/CD)
API_VERSION=v2 EXTRACT_TABLES=false uv run nv-ingest-harness-run --case=e2e

# Temporary changes without editing YAML
DATASET_DIR=/custom/path uv run nv-ingest-harness-run --case=e2e
```

## Test Scenarios

### Available Tests

| Name | Description | Configuration Needed | Status |
|------|-------------|----------------------|--------|
| `e2e` | Dataset-agnostic E2E ingestion | `active` section only | ✅ Primary (YAML config) |
| `e2e_with_llm_summary` | E2E with LLM summarization via UDF | `active` section only | ✅ Available (YAML config) |
| `recall` | Recall evaluation against existing collections | `active` + `recall` sections | ✅ Available (YAML config) |
| `e2e_recall` | Fresh ingestion + recall evaluation | `active` + `recall` sections | ✅ Available (YAML config) |

**Note**: Legacy test cases (`dc20_e2e`, `dc20_v2_e2e`) have been moved to `scripts/private_local`.

### Configuration Synergy

**For E2E-only users:**
- Only configure `active` section
- `collection_name` in active: auto-generates from `test_name` or dataset basename if `null` (deterministic, no timestamp)
- Collection name pattern: `{test_name}_multimodal` (e.g., `bo767_multimodal`, `earnings_consulting_multimodal`)
- `recall` section is optional (not used unless running recall tests)
- **Note**: You can run `recall` later against the same collection created by `e2e`

**For Recall-only users:**
- Configure `active` section: `hostname`, `sparse`, `gpu_search`, etc. (for evaluation)
- Configure `recall` section: `recall_dataset` (required)
- Set `test_name` in active to match your existing collection (collection must be `{test_name}_multimodal`)
- `collection_name` in active is ignored (recall generates `{test_name}_multimodal`)

**For E2E+Recall users:**
- Configure `active` section: `dataset_dir`, `test_name`, extraction settings, etc.
- Configure `recall` section: `recall_dataset` (required)
- Collection naming: e2e_recall automatically creates `{test_name}_multimodal` collection
- `collection_name` in active is ignored (e2e_recall forces `{test_name}_multimodal` pattern)

### Example Configurations

**V2 API with PDF Splitting:**
```yaml
# Edit test_configs.yaml active section:
active:
  api_version: v2
  pdf_split_page_count: 32
  extract_text: true
  extract_tables: true
  extract_charts: true
```

**Text-Only Processing:**
```yaml
active:
  extract_text: true
  extract_tables: false
  extract_charts: false
  extract_images: false
  extract_infographics: false
```

**RAG with Text Chunking:**
```yaml
active:
  enable_split: true
  split_chunk_size: 1024
  split_chunk_overlap: 150
```

**Multimodal with Image Extraction:**
```yaml
active:
  extract_text: true
  extract_tables: true
  extract_charts: true
  extract_images: true
  extract_infographics: true
  enable_caption: true
```

## Recall Testing

Recall testing evaluates retrieval accuracy against ground truth query sets. Two test cases are available:

### Test Cases

**`recall`** - Recall-only evaluation against existing collections:
- Skips ingestion (assumes collections already exist)
- Loads existing collections from Milvus
- Evaluates recall using multimodal queries (all datasets are multimodal-only)
- Supports reranker comparison (no reranker, with reranker, or reranker-only)

**`e2e_recall`** - Fresh ingestion + recall evaluation:
- Performs full ingestion pipeline
- Creates multimodal collection during ingestion
- Evaluates recall immediately after ingestion
- Combines ingestion metrics with recall metrics

### Reranker Configuration

Three modes via `reranker_mode` setting:

1. **No reranker** (default): `reranker_mode: none`
   - Runs evaluation without reranker only

2. **Both modes**: `reranker_mode: both`
   - Runs evaluation twice: once without reranker, once with reranker
   - Useful for comparing reranker impact

3. **Reranker only**: `reranker_mode: with`
   - Runs evaluation with reranker only
   - Faster when you only need reranked results

### Collection Naming

**Deterministic Collection Names (No Timestamps)**

All test cases use deterministic collection names (no timestamps) to enable:
- Reusing collections across test runs
- Running recall evaluation after e2e ingestion
- Consistent collection naming patterns

**Collection Name Patterns:**

All test cases use the same consistent pattern: `{test_name}_multimodal`

| Test Case | Pattern | Example |
|-----------|---------|---------|
| `e2e` | `{test_name}_multimodal` | `bo767_multimodal` |
| `e2e_with_llm_summary` | `{test_name}_multimodal` | `bo767_multimodal` |
| `e2e_recall` | `{test_name}_multimodal` | `bo767_multimodal` |
| `recall` | `{test_name}_multimodal` | `bo767_multimodal` |

**Benefits:**
- ✅ Run `e2e` then `recall` separately - they use the same collection
- ✅ Consistent naming across all test cases
- ✅ Deterministic names (no timestamps) enable collection reuse

**Recall Collections:**
- A single multimodal collection is created for recall evaluation
- Pattern: `{test_name}_multimodal`
- Example: `bo767_multimodal`
- All datasets evaluate against this multimodal collection (no modality-specific collections)

**Note**: Artifact directories still use timestamps for tracking over time (e.g., `bo767_20251106_180859_UTC`), but collection names are deterministic.

### Multimodal-Only Evaluation

All datasets use **multimodal-only** evaluation:
- Ground truth queries contain all content types (text, tables, charts)
- Single collection contains all extracted content types
- Simplified evaluation interface (no modality filtering)

### Ground Truth Files

**bo767 dataset:**
- Ground truth file: `bo767_query_gt.csv` (consolidated multimodal queries)
- Located in repo `data/` directory
- Default `ground_truth_dir: null` automatically uses `data/` directory
- Custom path can be specified via `ground_truth_dir` config

**Other datasets** (finance_bench, earnings, audio):
- Ground truth files must be obtained separately (not in public repo)
- Set `ground_truth_dir` to point to your ground truth directory
- Dataset-specific evaluators are extensible (see `recall_utils.py`)

### Configuration

Edit the `recall` section in `test_configs.yaml`:

```yaml
recall:
  # Reranker configuration
  reranker_mode: none  # Options: "none", "with", "both"

  # Recall evaluation settings
  recall_top_k: 10
  ground_truth_dir: null  # null = use repo data/ directory
  recall_dataset: bo767  # Required: must be explicitly set (bo767, finance_bench, earnings, audio)
```

### Usage Examples

**Recall-only (existing collections):**
```bash
# Evaluate existing bo767 collections (no reranker)
# recall_dataset automatically set from dataset config
uv run nv-ingest-harness-run --case=recall --dataset=bo767

# With reranker only (set reranker_mode in YAML recall section)
uv run nv-ingest-harness-run --case=recall --dataset=bo767

# Sweep multiple datasets for recall evaluation
uv run nv-ingest-harness-run --case=recall --dataset=bo767,earnings
```

**E2E + Recall (fresh ingestion):**
```bash
# Fresh ingestion with recall evaluation
# recall_dataset automatically set from dataset config
uv run nv-ingest-harness-run --case=e2e_recall --dataset=bo767

# Sweep multiple datasets (each ingests then evaluates)
uv run nv-ingest-harness-run --case=e2e_recall --dataset=bo767,earnings
```

**Dataset configuration:**
- **Dataset path**: Automatically set from `datasets` section when using `--dataset=<name>`
- **Extraction settings**: Automatically applied from `datasets` section
- **recall_dataset**: Automatically set from `datasets` section (e.g., `bo767`, `earnings`, `finance_bench`)
  - Can be overridden via environment variable: `RECALL_DATASET=bo767`
- **test_name**: Auto-generated from dataset name or basename of path (can set in YAML `active` section)
- **Collection naming**: `{test_name}_multimodal` (automatically generated for recall cases)
- All datasets evaluate against the same `{test_name}_multimodal` collection (multimodal-only)

### Output

Recall results are included in `results.json`:
```json
{
  "recall_results": {
    "no_reranker": {
      "1": 0.554,
      "3": 0.746,
      "5": 0.807,
      "10": 0.857
    },
    "with_reranker": {
      "1": 0.601,
      "3": 0.781,
      "5": 0.832,
      "10": 0.874
    }
  }
}
```

Metrics are also logged via `kv_event_log()`:
- `recall_multimodal_@{k}_no_reranker`
- `recall_multimodal_@{k}_with_reranker`
- `recall_eval_time_s_no_reranker`
- `recall_eval_time_s_with_reranker`

## Sweeping Parameters

### Dataset Sweeping (Recommended)

The easiest way to test multiple datasets is using dataset sweeping:

```bash
# Test multiple datasets - each gets its native config automatically
uv run nv-ingest-harness-run --case=e2e --dataset=bo767,earnings,bo20

# Each dataset runs with its pre-configured extraction settings
# Results are organized in separate artifact directories
```

### Parameter Sweeping

To sweep through different parameter values:

1. **Edit** `test_configs.yaml` - Update values in the `active` section
2. **Run** the test: `uv run nv-ingest-harness-run --case=e2e --dataset=<name>`
3. **Analyze** results in `artifacts/<test_name>_<timestamp>/`
4. **Repeat** steps 1-3 for next parameter combination

Example parameter sweep workflow:
```bash
# Test 1: Baseline V1
vim test_configs.yaml  # Set: api_version=v1, extract_tables=true
uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# Test 2: V2 with 32-page splitting
vim test_configs.yaml  # Set: api_version=v2, pdf_split_page_count=32
uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# Test 3: V2 with 8-page splitting
vim test_configs.yaml  # Set: pdf_split_page_count=8
uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# Test 4: Tables disabled (override via env var)
EXTRACT_TABLES=false uv run nv-ingest-harness-run --case=e2e --dataset=bo767
```

**Note**: Each test run creates a new timestamped artifact directory, so you can compare results across sweeps.

## Execution Modes

### Attach Mode (Default)

```bash
uv run nv-ingest-harness-run --case=e2e --dataset=bo767
```

- **Default behavior**: Assumes services are already running
- Runs test case only (no service management)
- Faster for iterative testing
- Use when Docker services are already up
- `--no-build` and `--keep-up` flags are ignored in attach mode

### Managed Mode

```bash
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed
```

- Starts Docker services automatically
- Waits for service readiness (configurable timeout)
- Runs test case
- Collects artifacts
- Stops services after test (unless `--keep-up`)

**Managed mode options:**
```bash
# Skip Docker image rebuild (faster startup)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --no-build

# Keep services running after test (useful for multi-test scenarios)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --keep-up
```

## Artifacts and Logging

All test outputs are collected in timestamped directories:

```
tools/harness/artifacts/<test_name>_<timestamp>_UTC/
├── results.json         # Consolidated test metadata and results
├── stdout.txt          # Complete test output
└── e2e.json            # Structured metrics and events
```

**Note**: Artifact directories use timestamps for tracking test runs over time, while collection names are deterministic (no timestamps) to enable collection reuse and recall evaluation.

### Results Structure

`results.json` contains:
- **Runner metadata**: case name, timestamp, git commit, infrastructure mode
- **Test configuration**: API version, extraction settings, dataset info
- **Test results**: chunks created, timing, performance metrics

### Document Analysis

Enable per-document element breakdown:

```bash
uv run nv-ingest-harness-run --case=e2e --doc-analysis
```

**Sample Output:**
```
Document Analysis:
  document1.pdf: 44 elements (text: 15, tables: 13, charts: 15, images: 0, infographics: 1)
  document2.pdf: 14 elements (text: 9, tables: 0, charts: 4, images: 0, infographics: 1)
```

This provides:
- Element counts by type for each document
- Useful for understanding dataset characteristics
- Helps identify processing bottlenecks
- Validates extraction completeness

## Architecture

### Framework Components

**1. Configuration Layer**
- `test_configs.yaml` - Structured configuration file
  - Active test configuration (edit directly)
  - Dataset shortcuts for quick access
- `config.py` - Configuration management
  - YAML loading and parsing
  - Type-safe config dataclass
  - Validation logic with helpful errors
  - Environment variable override support

**2. Test Runner**
- `run.py` - Main orchestration
  - Configuration loading with precedence chain
  - Docker service management (managed mode)
  - Test case execution with config injection
  - Artifact collection and consolidation

**3. Test Cases**
- `cases/e2e.py` - Primary E2E test (✅ YAML-based)
  - Accepts config object directly
  - Type-safe parameter access
  - Full pipeline validation (extract → embed → VDB → retrieval)
  - Transparent configuration logging
- `cases/e2e_with_llm_summary.py` - E2E with LLM (✅ YAML-based)
  - Adds UDF-based LLM summarization
  - Same config-based architecture as e2e.py
- `cases/recall.py` - Recall evaluation (✅ YAML-based)
  - Evaluates retrieval accuracy against existing collections
  - Requires `recall_dataset` in config (from dataset config or env var)
  - Supports reranker comparison modes (none, with, both)
  - Multimodal-only evaluation against `{test_name}_multimodal` collection
- `cases/e2e_recall.py` - E2E + Recall (✅ YAML-based)
  - Combines ingestion (via e2e.py) with recall evaluation (via recall.py)
  - Automatically creates collection during ingestion
  - Requires `recall_dataset` in config (from dataset config or env var)
  - Merges ingestion and recall metrics in results

**4. Shared Utilities**
- `interact.py` - Common testing utilities
  - `embed_info()` - Embedding model detection
  - `milvus_chunks()` - Vector database statistics
  - `segment_results()` - Result categorization by type
  - `kv_event_log()` - Structured logging
  - `pdf_page_count()` - Dataset page counting

### Configuration Flow

```
test_configs.yaml → load_config() → TestConfig object → test case
    (active +        (applies          (validated,
     datasets)        dataset config)    type-safe)
         ↑                    ↑
    Env overrides      Dataset configs
    (highest)          (auto-applied)
```

**Configuration Loading:**
1. Start with `active` section from YAML
2. If `--dataset=<name>` matches a configured dataset:
   - Apply dataset path
   - Apply dataset extraction settings
   - Apply dataset `recall_dataset` (if set)
3. Apply environment variable overrides (if any)
4. Validate and create `TestConfig` object

All test cases receive a validated `TestConfig` object with typed fields, eliminating string parsing errors.

## Development Guide

### Adding New Test Cases

1. **Create test script** in `cases/` directory

2. **Accept config parameter**:
   ```python
   def main(config, log_path: str = "test_results") -> int:
       """
       Test case entry point.
       
       Args:
           config: TestConfig object with all settings
           log_path: Path for structured logging
       
       Returns:
           Exit code (0 = success)
       """
       # Access config directly (type-safe)
       data_dir = config.dataset_dir
       api_version = config.api_version
       extract_text = config.extract_text
       # ...
   ```

3. **Add transparent logging**:
   ```python
   print("=== Test Configuration ===")
   print(f"Dataset: {config.dataset_dir}")
   print(f"API: {config.api_version}")
   print(f"Extract: text={config.extract_text}, tables={config.extract_tables}")
   print("=" * 60)
   ```

4. **Use structured logging**:
   ```python
   from interact import kv_event_log
   
   kv_event_log("ingestion_time_s", elapsed_time, log_path)
   kv_event_log("text_chunks", num_text_chunks, log_path)
   ```

5. **Register case** in `run.py`:
   ```python
   CASES = ["e2e", "e2e_with_llm_summary", "your_new_case"]
   ```

### Extending Configuration

To add new configurable parameters:

1. **Add to `TestConfig` dataclass** in `config.py`:
   ```python
   @dataclass
   class TestConfig:
       # ... existing fields
       new_param: bool = False  # Add with type and default
   ```

2. **Add to YAML** `active` section:
   ```yaml
   active:
     # ... existing config
     new_param: false  # Match Python default
   ```

3. **Add environment variable mapping** in `config.py` (if needed):
   ```python
   env_mapping = {
       # ... existing mappings
       "NEW_PARAM": ("new_param", parse_bool),
   }
   ```

4. **Add validation** (if needed) in `TestConfig.validate()`:
   ```python
   def validate(self) -> List[str]:
       errors = []
       # ... existing validation
       if self.new_param and self.some_other_field is None:
           errors.append("new_param requires some_other_field to be set")
       return errors
   ```

5. **Update this README** with parameter description

### Testing Different Datasets

The framework is dataset-agnostic and supports multiple approaches:

**Option 1: Use pre-configured dataset (Recommended)**
```bash
# Dataset configs automatically applied
uv run nv-ingest-harness-run --case=e2e --dataset=bo767
```

**Option 2: Add new dataset to YAML**
```yaml
datasets:
  my_dataset:
    path: /path/to/your/dataset
    extract_text: true
    extract_tables: true
    extract_charts: true
    extract_images: false
    extract_infographics: false
    recall_dataset: null  # or set to evaluator name if applicable
```
Then use: `uv run nv-ingest-harness-run --case=e2e --dataset=my_dataset`

**Option 3: Use custom path (uses active section config)**
```bash
uv run nv-ingest-harness-run --case=e2e --dataset=/path/to/your/dataset
```

**Option 4: Environment variable override**
```bash
# Override specific settings via env vars
EXTRACT_IMAGES=true uv run nv-ingest-harness-run --case=e2e --dataset=bo767
```

**Best Practice**: For repeated testing, add your dataset to the `datasets` section with its native extraction settings. This ensures consistent configuration and enables dataset sweeping.

## Additional Resources

- **Configuration**: See `config.py` for complete field list and validation logic
- **Test utilities**: See `interact.py` for shared helper functions  
## Profiling & Trace Capture Workflow

### Baseline vs RC Trace Runs

1. **Prep environment**
   ```bash
   source ~/setup_env.sh
   nv-restart             # baseline (local build)
   # or
   nv-stop && nv-start-release   # RC image
   ```
2. **Enable traces + V2 split settings (env vars or YAML)**
   ```bash
   export API_VERSION=v2
   export PDF_SPLIT_PAGE_COUNT=32          # use 16/64/etc. as needed
   export ENABLE_TRACES=true
   # Optional: export TRACE_OUTPUT_DIR=/raid/jioffe/traces/baseline
   ```
3. **Run datasets (repeat for bo20 + bo767, baseline + RC)**
   ```bash
   python scripts/tests/run.py --case=e2e --dataset=bo20
   python scripts/tests/run.py --case=e2e --dataset=bo767
   ```
4. **Collect artifacts**
   - `stdout.txt` → console logs
   - `results.json` → includes `trace_summary` with per-stage resident seconds
   - `traces/*.json` → raw payloads identical to `scripts/private_local/trace.json`
   - Use descriptive folder names (e.g., copy artifacts to `artifacts/baseline_bo20/`) to keep baseline vs RC side-by-side.
5. **Visualize stage timings**
   ```bash
   # Generates results.stage_time.png + textual summary
   python scripts/tests/tools/plot_stage_totals.py \
     scripts/tests/artifacts/<run>/results.json \
     --sort total \
     --keep-nested \
     --exclude-network
   ```
   - `--keep-nested` preserves nested entries such as chart/table OCR workloads
   - `--exclude-network` hides broker/Redis wait time so the chart focuses on Ray stages

### Future Trace Visualization Case (roadmap)

With `trace_summary` + raw dumps in place, a follow-on `cases/e2e_profile.py` can:
- Accept one or two artifact directories (baseline vs RC) as input.
- Load each run’s `trace_summary` and optional raw trace JSON to compute deltas.
- Emit per-stage bar/violin charts (resident vs wall clock) plus CSV summaries.
- Store outputs under `trace_profile/` inside each artifact directory.

Open questions before implementing:
- Preferred plotting backend (`matplotlib`, `plotly`, or Altair) and whether CI should emit PNG, HTML, or both.
- Artifact naming for comparisons (e.g., `trace_profile/baseline_vs_rc/bo20.png`).
- Regression thresholds and alerting expectations (what % delta constitutes a failure?).
- Should the visualization case automatically diff against the most recent baseline or just emit standalone assets?

Capturing these answers (plus a few representative trace samples) will let a future sprint land the visualization workflow quickly.
- **Docker setup**: See project root README for service management commands
- **API documentation**: See `docs/` for API version differences

The framework prioritizes clarity, type safety, and validation to support reliable testing of nv-ingest pipelines.
