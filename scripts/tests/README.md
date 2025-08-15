# nv-ingest Integration Testing Framework

A configurable, dataset-agnostic testing framework for end-to-end validation of nv-ingest pipelines. This framework is designed to be modular, transparent, and easy to extend for different datasets and test scenarios.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Access to test datasets (default: `/datasets`) NOTE: Future versions 
- Python environment with nv-ingest-client

### Run Your First Test
```bash
# 1. Navigate to the tests directory
cd scripts/tests

# 2. Create your configuration file
cp env.example .env

# 3. Update dataset path in .env (required)
# Edit DATASET_DIR=/path/to/your/dataset

# 4. Run the test (from scripts/tests directory)
python run.py dc20_e2e
```

**Important**: All test commands should be run from the `scripts/tests/` directory. The framework expects to find configuration files and test cases relative to this location.

## Configuration System

### Environment Variables (`.env` file)

The framework is fully configurable via environment variables. All parameters can be set in your `.env` file:

#### Required Configuration
```bash
# Dataset path - MUST be set for each test
DATASET_DIR=/datasets/bo20
```

#### Test Configuration
```bash
# Test identification
TEST_NAME=dc20                    # Affects collection naming
COLLECTION_NAME=                  # Override auto-generated name

# Infrastructure
HOSTNAME=localhost                # Service hostname
PROFILES=retrieval,table-structure # Docker compose profiles
READINESS_TIMEOUT=600            # Service startup timeout (seconds)
```

#### Extraction Parameters (Fully Configurable)
```bash
# Content extraction toggles
EXTRACT_TEXT=true                 # Extract text content
EXTRACT_TABLES=true              # Extract table structures
EXTRACT_CHARTS=true              # Extract chart/graph content
EXTRACT_IMAGES=false             # Extract image content
EXTRACT_INFOGRAPHICS=true        # Extract infographic content

# Extraction behavior
TEXT_DEPTH=page                  # Text extraction depth (page/document)
TABLE_OUTPUT_FORMAT=markdown     # Table format (markdown/html)
```

#### Runtime Configuration
```bash
# Vector database settings
SPARSE=true                      # Enable sparse embeddings
GPU_SEARCH=false                 # Use GPU for vector search

# Directories
SPILL_DIR=/tmp/spill            # Temporary processing directory
ARTIFACTS_DIR=                   # Test artifacts output (auto-generated if empty)
LOG_PATH=                        # Log output path (managed by run.py)

# Model configuration
EMBEDDING_NIM_MODEL_NAME=        # Override embedding model (auto-detected)
```

## Test Scenarios

### Current Tests

| Name | Description | Validates | Dataset | Status |
|------|-------------|-----------|---------|--------|
| dc20_e2e | Digital Corpora E2E | Extract → Embed → VDB Upload → Retrieval | `/datasets/bo20` | ✅ Active |

### Example Test Configurations

#### Text-Only Processing (Audio Datasets)
```bash
# In .env file:
DATASET_DIR=/datasets/audio
TEST_NAME=audio_test
EXTRACT_TABLES=false
EXTRACT_CHARTS=false
EXTRACT_IMAGES=false
EXTRACT_INFOGRAPHICS=false
```

#### Full Multiformat Extraction
```bash
# In .env file:
DATASET_DIR=/datasets/multiformat
TEST_NAME=multiformat_test
EXTRACT_IMAGES=true
TABLE_OUTPUT_FORMAT=html
```

## Architecture

### Framework Components

1. **`run.py`** - Main test runner
   - Manages Docker services (managed mode)
   - Handles artifacts collection
   - Sets up logging paths
   - Provides infrastructure abstraction

2. **`cases/dc20_e2e.py`** - Test implementation
   - Fully configurable via environment variables
   - Transparent configuration logging
   - Comprehensive metrics collection
   - Modular design for future expansion

3. **`utils.py`** - Shared utilities
   - `embed_info()` - Embedding model detection
   - `milvus_chunks()` - Vector database statistics
   - `segment_results()` - Result categorization
   - `kv_event_log()` - Structured logging

### Execution Modes

#### Managed Mode (Default)
```bash
# From scripts/tests/ directory:
python run.py dc20_e2e
```
- Starts Docker services automatically
- Waits for service readiness
- Runs test case
- Collects artifacts
- Optionally tears down services

#### Attach Mode
```bash
# From scripts/tests/ directory:
python run.py --infra attach dc20_e2e
```
- Assumes services are already running
- Runs test case only
- User manages service lifecycle

### Artifacts and Logging

All test outputs are collected in timestamped directories:
```
scripts/tests/artifacts/20240813_171507/
├── summary.json          # Test metadata and results
├── stdout.txt           # Complete test output
└── dc20_e2e.json       # Structured metrics and events
```

### Programmatic Access

For automated tools and CI/CD integration:

```python
from scripts.interact import load_test_artifacts

# Load and validate test results
artifacts = load_test_artifacts('artifacts/20240813_171507')

# Check test success
success = artifacts.summary.return_code == 0

# Extract metrics (example for DC20 test)
if 'dc20_e2e.json' in artifacts.test_results:
    events = artifacts.test_results['dc20_e2e.json']
    ingestion_time = next(e['value'] for e in events if e['metric_name'] == 'ingestion_time_s')
```

Models: `TestSummary`, `DC20E2EResults`, `TestArtifacts` in `scripts.interact`

## Development Guide

### Adding New Test Cases

1. **Create test script** in `cases/` directory
2. **Follow configuration pattern**:
   ```python
   def _get_env(name: str, default: str | None = None) -> str | None:
       val = os.environ.get(name)
       return val if val and val != "" else default
   ```
3. **Add transparent logging**:
   ```python
   print(f"=== Configuration ===")
   print(f"Dataset: {dataset_dir}")
   # ... log all parameters
   ```
4. **Use structured logging**:
   ```python
   kv_event_log("metric_name", value, log_path)
   ```

### Extending Configuration

To add new configurable parameters:

1. **Add to `env.example`** with documentation
2. **Read in test script** using `_get_env()`
3. **Add to configuration logging**
4. **Update this README** with parameter description

### Testing Different Datasets

The framework is dataset-agnostic. To test with new datasets:

1. **Set `DATASET_DIR`** to your dataset path
2. **Adjust extraction parameters** for dataset type
3. **Update `TEST_NAME`** for clear identification
4. **Run test** with your configuration

## Future Roadmap

### Modular Architecture Vision

The framework is designed to support a modular testing architecture:

```bash
# Future modular scripts (chainable, dataset-agnostic):
ingest_documents.py    # Ingestion only
index_documents.py     # Vector database upload only  
retrieve_documents.py  # Retrieval testing only

# Composed in test scripts:
your_test.sh          # Chains scripts with parameters
```

### Planned Expansions

#### Additional Test Cases
- **Multiformat Testing** - Mixed content extraction validation
- **Audio Pipeline** - Audio-specific ingestion and retrieval
- **Earnings Analysis** - Domain-specific document processing
- **Foundation RAG** - Large-scale RAG workflow validation
- **Scale Testing** - Performance validation with larger datasets (dc767, dc10k)

#### Dataset Management
- **NGC Dataset Integration** - Portable dataset collections through NVIDIA GPU Cloud
- **Dataset Validation** - Pre-test dataset integrity checks

#### Infrastructure Enhancements
- **Multi-GPU Testing** - Distributed processing validation
- **Performance Benchmarking** - Automated performance regression detection
- **CI/CD Integration** - Automated testing in development pipelines

## Contributing

When adding new features or test cases:

1. **Maintain configurability** - All parameters should be environment-driven
2. **Add transparent logging** - Log all configuration for debugging
3. **Update documentation** - Keep this README current
4. **Follow naming conventions** - Use descriptive, consistent naming
5. **Test thoroughly** - Validate with different configurations

The framework prioritizes simplicity, configurability, and extensibility to support the evolving needs of nv-ingest testing.
