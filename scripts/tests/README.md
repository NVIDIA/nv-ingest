# nv-ingest Integration Testing Framework

A configurable, dataset-agnostic testing framework for end-to-end validation of nv-ingest pipelines. This framework uses structured YAML configuration for type safety, validation, and easy parameter management.

## ✨ **Recent Improvements**

- **YAML-based configuration** - Structured, type-safe config with validation
- **Direct config access** - No more string parsing, clean config objects
- **Dataset shortcuts** - Quick access to common datasets
- **CLI overrides** - Override any setting via command line
- **Clear precedence** - CLI args > Environment variables > YAML config
- **Smart validation** - Catch configuration errors before tests run
- **Dataset-agnostic testing** - One `e2e` case works with any dataset
- **Document analysis** - Per-document element breakdowns with `--doc-analysis`

## Quick Start

### Prerequisites
- Docker and Docker Compose running
- Python environment with nv-ingest-client
- Access to test datasets (e.g., `/raid/jioffe/bo20`, `/raid/jioffe/bo767`)

### Run Your First Test

```bash
# 1. Navigate to the tests directory
cd scripts/tests

# 2. Check the default configuration
cat test_configs.yaml

# 3. Edit the active configuration if needed
vim test_configs.yaml  # Update dataset_dir, extraction settings, etc.

# 4. Run the test (assumes services are already running)
python run.py --case=e2e

# Or override settings via CLI
python run.py --case=e2e --dataset=bo767 --api-version=v2
```

**Important**: All test commands should be run from the `scripts/tests/` directory.

## Configuration System

### YAML Configuration (`test_configs.yaml`)

The framework uses a structured YAML file for all test configuration. Configuration is organized into logical sections:

#### Active Configuration

The `active` section contains your current test settings:

```yaml
active:
  # Dataset
  dataset_dir: /your/dataset/path
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
  spill_dir: /raid/jioffe/tmp/spill
  artifacts_dir: null
  collection_name: null
```

#### Dataset Shortcuts

Define common dataset paths for quick access:

```yaml
datasets:
  bo20: /raid/jioffe/bo20
  bo767: /raid/jioffe/bo767
  single: data/multimodal_test.pdf
```

Usage:
```bash
python run.py --case=e2e --dataset=bo767  # Uses /raid/jioffe/bo767
```

#### Preset Configurations (Documentation)

The `presets` section documents common configurations for reference:

```yaml
presets:
  baseline_v1:
    api_version: v1
    extract_text: true
    extract_tables: true
    extract_charts: true
  
  v2_split_32:
    api_version: v2
    pdf_split_page_count: 32
```

*Note: Presets are currently documentation-only. To use a preset config, manually update the `active` section.*

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
- `table_output_format` (string): Table output format - `markdown`, `html`, `latex`, etc.

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
python run.py --case=e2e --dataset=bo767

# Override API version
python run.py --case=e2e --api-version=v2

# Override readiness timeout
python run.py --case=e2e --readiness-timeout=300

# Combine multiple overrides
python run.py --case=e2e --dataset=bo20 --api-version=v2 --doc-analysis
```

### Using Environment Variables

```bash
# Override via environment
API_VERSION=v2 EXTRACT_TABLES=false python run.py --case=e2e

# Useful for temporary changes without editing YAML
DATASET_DIR=/custom/path python run.py --case=e2e
```

## Test Scenarios

### Current Tests

| Name | Description | Validates | Status |
|------|-------------|-----------|--------|
| e2e | Dataset-Agnostic E2E | Extract → Embed → VDB Upload → Retrieval | ✅ Primary |
| dc20_e2e | Legacy DC20 E2E | Extract → Embed → VDB Upload → Retrieval | ✅ Backwards Compatible |
| e2e_with_llm_summary | E2E with LLM | Extract → Embed → VDB Upload → Retrieval + LLM | ✅ Available |

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

## Sweeping Parameters

To sweep through different parameter values:

1. **Edit** `test_configs.yaml` - Update values in the `active` section
2. **Run** the test: `python run.py --case=e2e`
3. **Analyze** results in `artifacts/<test_name>_<timestamp>/`
4. **Repeat** steps 1-3 for next parameter combination

Example sweep workflow:
```bash
# Test 1: Baseline
vim test_configs.yaml  # Set: api_version=v1, extract_tables=true
python run.py --case=e2e --dataset=bo20

# Test 2: V2 with splitting
vim test_configs.yaml  # Set: api_version=v2, pdf_split_page_count=32
python run.py --case=e2e --dataset=bo20

# Test 3: Tables disabled
vim test_configs.yaml  # Set: extract_tables=false
python run.py --case=e2e --dataset=bo20
```

*Note: Automated parameter sweeping will be added in a future PR.*

## Execution Modes

### Attach Mode (Default)

```bash
python run.py --case=e2e
```

- Assumes services are already running (typical development workflow)
- Runs test case only
- Faster for iterative testing

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
# Skip Docker image rebuild
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
- Test runner metadata (case, timestamp, commit, infrastructure mode)
- Test configuration (API version, extraction settings, etc.)
- Test results (chunks created, timing, performance metrics)

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
- Page-level granularity for capacity planning
- Performance insights for optimization

## Architecture

### Framework Components

1. **`test_configs.yaml`** - Structured configuration file
   - Active test configuration
   - Dataset shortcuts
   - Preset documentation

2. **`config.py`** - Configuration management
   - YAML loading and parsing
   - Type-safe config dataclass
   - Validation logic
   - Environment variable overrides

3. **`run.py`** - Main test runner
   - Configuration loading
   - Docker service management (managed mode)
   - Test case execution
   - Artifacts collection

4. **`cases/e2e.py`** - Primary test implementation
   - Accepts config object directly
   - Transparent configuration logging
   - Comprehensive metrics collection
   - Full pipeline validation

5. **`interact.py`** - Shared utilities
   - `embed_info()` - Embedding model detection
   - `milvus_chunks()` - Vector database statistics
   - `segment_results()` - Result categorization
   - `kv_event_log()` - Structured logging

### Configuration Flow

```
test_configs.yaml → load_config() → TestConfig object → test case
                         ↑
                    CLI overrides
                         ↑
                    Env var overrides
```

All test cases receive a validated `TestConfig` object with typed fields.

## Development Guide

### Adding New Test Cases

1. **Create test script** in `cases/` directory
2. **Accept config parameter**:
   ```python
   def main(config, log_path: str = "test_results") -> int:
       # Access config directly
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
   # ...
   ```
4. **Use structured logging**:
   ```python
   kv_event_log("metric_name", value, log_path)
   ```
5. **Register case** in `run.py` CASES list

### Extending Configuration

To add new configurable parameters:

1. **Add to `TestConfig` dataclass** in `config.py`:
   ```python
   @dataclass
   class TestConfig:
       # ... existing fields
       new_param: bool = False
   ```

2. **Add to YAML** `active` section default:
   ```yaml
   active:
     # ... existing config
     new_param: false
   ```

3. **Add environment variable mapping** in `config.py`:
   ```python
   env_mapping = {
       # ... existing mappings
       "NEW_PARAM": ("new_param", parse_bool),
   }
   ```

4. **Update this README** with parameter description

### Testing Different Datasets

The framework is dataset-agnostic:

1. **Option 1: Edit YAML**
   ```yaml
   active:
     dataset_dir: /path/to/your/dataset
   ```

2. **Option 2: Use CLI**
   ```bash
   python run.py --case=e2e --dataset=/path/to/your/dataset
   ```

3. **Option 3: Add dataset shortcut**
   ```yaml
   datasets:
     my_dataset: /path/to/your/dataset
   ```
   Then: `python run.py --case=e2e --dataset=my_dataset`

## Troubleshooting

### Configuration Validation Errors

If you see validation errors, check:
- `text_depth` is one of: page, document, block, line, etc.
- `table_output_format` is one of: markdown, html, latex, etc.
- `api_version` is either v1 or v2
- `pdf_split_page_count` is only set when `api_version` is v2

### Dataset Not Found

Ensure the dataset path exists:
```bash
ls -ld /raid/jioffe/bo20  # Check path accessibility
```

### Services Not Ready

If using `--managed` mode and services don't start:
- Increase `readiness_timeout` in YAML or via CLI
- Check Docker service logs: `docker compose logs nv-ingest-ms-runtime`
- Verify compose profiles match your deployment

## Migration Guide

If you have existing tests using the old `.env` configuration:

1. **Backup** your `.env` file:
   ```bash
   cp .env .env.backup
   ```

2. **Update** `test_configs.yaml` active section with your values

3. **Test** the new configuration:
   ```bash
   python run.py --case=e2e
   ```

4. **Environment variables still work** as overrides if needed

See `CONFIG_MIGRATION.md` for detailed migration information.

## Future Roadmap

### Planned Enhancements

- **Automated parameter sweeping** - Matrix testing across parameter combinations
- **Profile inheritance** - Extend base profiles with modifications
- **Multi-environment configs** - Separate configs for local/DGX/CI
- **Config diff tool** - Compare configurations across runs
- **Result comparison** - Automated regression detection
- **NGC Dataset Integration** - Portable dataset collections
- **Multi-GPU testing** - Distributed processing validation

### Additional Test Cases

- Multiformat testing - Mixed content validation
- Audio pipeline - Audio-specific ingestion
- Scale testing - Performance validation with large datasets
- Domain-specific testing - Earnings, technical docs, etc.

## Contributing

When adding new features or test cases:

1. **Maintain type safety** - Use config object, not string parsing
2. **Add validation** - Catch errors early in config.validate()
3. **Add transparent logging** - Log all configuration for debugging
4. **Update documentation** - Keep this README current
5. **Follow naming conventions** - Use descriptive, consistent naming
6. **Test thoroughly** - Validate with different configurations

The framework prioritizes type safety, validation, and structured configuration to support the evolving needs of nv-ingest testing.
