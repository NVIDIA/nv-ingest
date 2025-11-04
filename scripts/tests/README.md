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
cd scripts/tests

# 2. Edit the configuration
vim test_configs.yaml  # Update dataset_dir to point to your data

# 3. Run the test (assumes services are already running)
python run.py --case=e2e

# Or override settings via CLI
python run.py --case=e2e --dataset=/path/to/data --api-version=v2
```

**Important**: All test commands should be run from the `scripts/tests/` directory.

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

#### Dataset Shortcuts

Define common dataset paths for quick access:

```yaml
datasets:
  bo20: /path/to/bo20
  bo767: /path/to/bo767
  single: data/multimodal_test.pdf
```

Usage:
```bash
python run.py --case=e2e --dataset=bo767  # Uses /path/to/bo767
```

### Configuration Precedence

Settings are applied in order of priority:

**CLI arguments > Environment variables > YAML active config**

Example:
```bash
# YAML has api_version: v1
# Override via environment variable
API_VERSION=v2 python run.py --case=e2e

# Override via CLI (highest priority)
python run.py --case=e2e --api-version=v2
```

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
- `collection_name` (string): Milvus collection name (auto-generated if null)

### Valid Configuration Values

**text_depth**: `block`, `body`, `document`, `header`, `line`, `nearby_block`, `other`, `page`, `span`

**table_output_format**: `html`, `image`, `latex`, `markdown`, `pseudo_markdown`, `simple`

**api_version**: `v1`, `v2`

Configuration is validated on load with helpful error messages.

## Running Tests

### Basic Usage

```bash
# Run with default YAML configuration
python run.py --case=e2e

# With document-level analysis
python run.py --case=e2e --doc-analysis

# With managed infrastructure (starts/stops Docker)
python run.py --case=e2e --managed
```

### Using CLI Overrides

```bash
# Override dataset
python run.py --case=e2e --dataset=/path/to/data

# Override API version
python run.py --case=e2e --api-version=v2

# Override readiness timeout
python run.py --case=e2e --readiness-timeout=300

# Combine multiple overrides
python run.py --case=e2e --dataset=/data/pdfs --api-version=v2 --doc-analysis
```

### Using Environment Variables

```bash
# Override via environment (useful for CI/CD)
API_VERSION=v2 EXTRACT_TABLES=false python run.py --case=e2e

# Temporary changes without editing YAML
DATASET_DIR=/custom/path python run.py --case=e2e
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
- `collection_name` in active: auto-generates with timestamp if `null`, or use custom name
- `recall` section is optional (not used unless running recall tests)

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

Three modes via two boolean flags:

1. **No reranker** (default): `use_reranker=false, reranker_only=false`
   - Runs evaluation without reranker only

2. **Both modes**: `use_reranker=true, reranker_only=false`
   - Runs evaluation twice: once without reranker, once with reranker
   - Useful for comparing reranker impact

3. **Reranker only**: `use_reranker=true, reranker_only=true`
   - Runs evaluation with reranker only
   - Faster when you only need reranked results

### Collection Naming

A single multimodal collection is created for recall evaluation:
- Pattern: `{test_name}_multimodal`
- Example: `bo767_multimodal`
- All datasets evaluate against this multimodal collection (no modality-specific collections)

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
  use_reranker: false
  reranker_only: false

  # Recall evaluation settings
  recall_top_k: 10
  ground_truth_dir: null  # null = use repo data/ directory
  recall_dataset: bo767  # Required: must be explicitly set (bo767, finance_bench, earnings, audio)
```

### Usage Examples

**Recall-only (existing collections):**
```bash
# Evaluate existing bo767 collections
python run.py --case=recall --test-name=bo767

# With reranker comparison
python run.py --case=recall --test-name=bo767 --use-reranker
```

**E2E + Recall (fresh ingestion):**
```bash
# Fresh ingestion with recall evaluation
python run.py --case=e2e_recall --dataset=/raid/jioffe/bo767
```

**Dataset configuration:**
- `dataset_dir` (active section): Where PDFs are located (for ingestion)
- `test_name` (active section): Used to generate collection name `{test_name}_multimodal`
- `recall_dataset` (recall section): **Required** - Which evaluator to use (bo767, finance_bench, earnings, audio)
  - Set in `test_configs.yaml` recall section: `recall_dataset: bo767`
  - Or via environment variable: `RECALL_DATASET=bo767`
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

To sweep through different parameter values:

1. **Edit** `test_configs.yaml` - Update values in the `active` section
2. **Run** the test: `python run.py --case=e2e`
3. **Analyze** results in `artifacts/<test_name>_<timestamp>/`
4. **Repeat** steps 1-3 for next parameter combination

Example sweep workflow:
```bash
# Test 1: Baseline V1
vim test_configs.yaml  # Set: api_version=v1, extract_tables=true
python run.py --case=e2e --dataset=/data/pdfs

# Test 2: V2 with 32-page splitting
vim test_configs.yaml  # Set: api_version=v2, pdf_split_page_count=32
python run.py --case=e2e --dataset=/data/pdfs

# Test 3: V2 with 8-page splitting
vim test_configs.yaml  # Set: pdf_split_page_count=8
python run.py --case=e2e --dataset=/data/pdfs

# Test 4: Tables disabled
vim test_configs.yaml  # Set: extract_tables=false
python run.py --case=e2e --dataset=/data/pdfs
```

**Note**: Each test run creates a new timestamped artifact directory, so you can compare results across sweeps.

## Execution Modes

### Attach Mode (Default)

```bash
python run.py --case=e2e
```

- Assumes services are already running (typical development workflow)
- Runs test case only
- Faster for iterative testing
- Use when Docker services are already up (e.g., via `nv-start`)

### Managed Mode

```bash
python run.py --case=e2e --managed
```

- Starts Docker services automatically
- Waits for service readiness (configurable timeout)
- Runs test case
- Collects artifacts
- Optionally tears down services (use `--keep-up` to preserve)

Additional managed mode options:
```bash
# Skip Docker image rebuild (faster startup)
python run.py --case=e2e --managed --no-build

# Keep services running after test
python run.py --case=e2e --managed --keep-up
```

## Artifacts and Logging

All test outputs are collected in timestamped directories:

```
scripts/tests/artifacts/<test_name>_<timestamp>_UTC/
├── results.json         # Consolidated test metadata and results
├── stdout.txt          # Complete test output
└── e2e.json            # Structured metrics and events
```

### Results Structure

`results.json` contains:
- **Runner metadata**: case name, timestamp, git commit, infrastructure mode
- **Test configuration**: API version, extraction settings, dataset info
- **Test results**: chunks created, timing, performance metrics

### Document Analysis

Enable per-document element breakdown:

```bash
python run.py --case=e2e --doc-analysis
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
                         ↑                    ↑
                    Env overrides        CLI overrides
```

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

**Option 1: Edit YAML directly**
```yaml
active:
  dataset_dir: /path/to/your/dataset
```

**Option 2: Use CLI override**
```bash
python run.py --case=e2e --dataset=/path/to/your/dataset
```

**Option 3: Add dataset shortcut**
```yaml
datasets:
  my_dataset: /path/to/your/dataset
```
Then use: `python run.py --case=e2e --dataset=my_dataset`

**Option 4: Environment variable**
```bash
DATASET_DIR=/path/to/your/dataset python run.py --case=e2e
```

Choose the approach that fits your workflow. For repeated testing, add a dataset shortcut. For one-off tests, use CLI override.

## Additional Resources

- **Configuration**: See `config.py` for complete field list and validation logic
- **Test utilities**: See `interact.py` for shared helper functions  
- **Docker setup**: See project root README for service management commands
- **API documentation**: See `docs/` for API version differences

The framework prioritizes clarity, type safety, and validation to support reliable testing of nv-ingest pipelines.
