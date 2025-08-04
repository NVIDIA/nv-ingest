# Ray Logging Configuration

The NV-Ingest pipeline supports comprehensive Ray logging configuration through environment variables. This enables fine-grained control over Ray's logging behavior and resource usage across distributed workers.

## Quick Start - Preset Configurations

Use the package-level preset variable for instant configuration:

```bash
# Production deployment - minimal logging, maximum performance
export INGEST_RAY_LOG_LEVEL=PRODUCTION

# Development work - balanced logging (default)
export INGEST_RAY_LOG_LEVEL=DEVELOPMENT  

# Debugging issues - maximum visibility
export INGEST_RAY_LOG_LEVEL=DEBUG
```

## Preset Levels

### PRODUCTION
Optimized for production deployments with minimal logging overhead:
- **Log Level**: ERROR only
- **Log to Driver**: Disabled (worker logs stay in worker files)
- **Import Warnings**: Disabled
- **Usage Stats**: Disabled  
- **Storage**: 10GB total (1GB × 10 files)
- **Deduplication**: Enabled
- **Encoding**: TEXT

### DEVELOPMENT (Default)
Balanced configuration for development work:
- **Log Level**: INFO
- **Log to Driver**: Enabled
- **Import Warnings**: Enabled
- **Usage Stats**: Enabled
- **Storage**: 20GB total (1GB × 20 files) 
- **Deduplication**: Enabled
- **Encoding**: TEXT

### DEBUG
Maximum visibility for troubleshooting:
- **Log Level**: DEBUG
- **Log to Driver**: Enabled
- **Import Warnings**: Enabled
- **Usage Stats**: Enabled
- **Storage**: 20GB total (512MB × 40 files)
- **Deduplication**: Disabled (see all duplicate messages)
- **Encoding**: JSON with function names and line numbers

## Log Verbosity Control

Ray generates significant logging output by default. The following configuration reduces log volume:

```bash
# Suppress informational messages, show only warnings and errors
export RAY_LOGGING_LEVEL=WARNING

# Prevent worker logs from appearing in driver process output
export RAY_LOG_TO_DRIVER=false

# Suppress Ray initialization warnings
export RAY_DISABLE_IMPORT_WARNING=1
```

## Configuration Reference

### Package-Level Preset
| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `INGEST_RAY_LOG_LEVEL` | PRODUCTION, DEVELOPMENT, DEBUG | DEVELOPMENT | Sets all Ray logging variables to preset defaults |

### Core Logging Control
| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `RAY_LOGGING_LEVEL` | DEBUG, INFO, WARNING, ERROR, CRITICAL | INFO | DEBUG: All Ray internals visible. WARNING: Only significant events logged |
| `RAY_LOGGING_ENCODING` | TEXT, JSON | TEXT | TEXT: Human-readable. JSON: Machine-parseable with structured fields |
| `RAY_LOGGING_ADDITIONAL_ATTRS` | Comma-separated list | (empty) | Adds Python logger fields like function names, line numbers to each log entry |

### Log Flow Control
| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `RAY_DEDUP_LOGS` | 0, 1 | 1 | 1: Repeated messages show "[repeated 5x]". 0: All duplicate messages printed |
| `RAY_LOG_TO_DRIVER` | true, false | true | true: Worker logs appear in main process. false: Worker logs only in worker log files |

### File Rotation
| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `RAY_LOGGING_ROTATE_BYTES` | Bytes | 1073741824 (1GB) | Log file size before creating new file. Prevents unbounded disk usage |
| `RAY_LOGGING_ROTATE_BACKUP_COUNT` | Integer | 19 | Number of old log files retained. Total storage = (count + 1) × file size |

### Ray Internal Logging
| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `RAY_DISABLE_IMPORT_WARNING` | 0, 1 | 0 | 1: Suppresses "Ray X.Y.Z started" and import warnings during initialization |
| `RAY_USAGE_STATS_ENABLED` | 0, 1 | 1 | 0: Disables telemetry collection and related log messages |

## Configuration Examples

### Using Presets
```bash
# Production deployment
export INGEST_RAY_LOG_LEVEL=PRODUCTION
# Result: ERROR level, no driver logs, 10GB storage, minimal overhead

# Development with custom override
export INGEST_RAY_LOG_LEVEL=DEVELOPMENT
export RAY_LOGGING_LEVEL=WARNING  # Override just the log level
# Result: WARNING level with all other DEVELOPMENT defaults

# Maximum debugging
export INGEST_RAY_LOG_LEVEL=DEBUG
# Result: DEBUG level, JSON output, function names, no deduplication
```

### Manual Configuration (Legacy)
```bash
# Minimal Logging
export RAY_LOGGING_LEVEL=ERROR
export RAY_LOG_TO_DRIVER=false
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=1
```
**Result**: Only critical errors logged. Worker logs isolated. ~95% reduction in log volume.

### Structured Logging for Analysis
```bash
export INGEST_RAY_LOG_LEVEL=DEVELOPMENT
export RAY_LOGGING_ENCODING=JSON
export RAY_LOGGING_ADDITIONAL_ATTRS=name,funcName,lineno,thread,process
```
**Result**: Machine-parseable JSON with metadata for log aggregation systems.

### Custom Storage Limits
```bash
# 5GB total log storage (500MB × 10 files)
export RAY_LOGGING_ROTATE_BYTES=524288000
export RAY_LOGGING_ROTATE_BACKUP_COUNT=9
```
**Result**: Automatic cleanup when logs exceed 5GB. Oldest files removed first.

## Log Output Examples

### Default (INFO level)
```
2024-01-15 10:30:15,123 INFO worker.py:1234 -- Task task_id=abc123 started
2024-01-15 10:30:15,124 INFO worker.py:1235 -- Processing batch size=100
2024-01-15 10:30:15,125 INFO worker.py:1236 -- Task task_id=abc123 completed
```

### WARNING level
```
2024-01-15 10:30:20,456 WARNING worker.py:1240 -- Task retry attempt 2/3
2024-01-15 10:30:25,789 ERROR worker.py:1245 -- Task failed: Connection timeout
```

### JSON encoding (DEBUG preset)
```json
{"asctime": "2024-01-15 10:30:15,123", "levelname": "INFO", "filename": "worker.py", "lineno": 1234, "message": "Task started", "name": "ray.worker", "funcName": "execute_task", "job_id": "01000000", "worker_id": "abc123", "task_id": "def456"}
```

## Storage and Performance

**Preset Storage Limits**:
- **PRODUCTION**: 10GB total (1GB × 10 files)
- **DEVELOPMENT**: 20GB total (1GB × 20 files)  
- **DEBUG**: 20GB total (512MB × 40 files)

**Performance Impact by Preset**:
- **PRODUCTION**: ~5% CPU reduction, ~200MB memory savings in large clusters
- **DEVELOPMENT**: Balanced performance and visibility
- **DEBUG**: ~10% CPU overhead for detailed logging, higher memory usage

**Log Storage**:
- Log files stored in `/tmp/ray/session_*/logs/` per node
- Automatic rotation prevents disk exhaustion
- Oldest files removed when storage limit reached

## Implementation Notes

- **Preset Priority**: `INGEST_RAY_LOG_LEVEL` sets defaults, individual env vars override
- Environment variables must be set before pipeline initialization
- Invalid values automatically fall back to defaults with warning messages
- Configuration changes require pipeline restart to take effect
- Log deduplication uses message content hashing for efficiency
