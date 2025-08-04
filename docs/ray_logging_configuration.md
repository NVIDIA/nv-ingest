# Ray Logging Configuration

The NV-Ingest pipeline supports comprehensive Ray logging configuration through environment variables. This enables fine-grained control over Ray's logging behavior and resource usage across distributed workers.

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

### Minimal Logging (Production)
```bash
export RAY_LOGGING_LEVEL=ERROR
export RAY_LOG_TO_DRIVER=false
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=1
```
**Result**: Only critical errors logged. Worker logs isolated. ~95% reduction in log volume.

### Debug Configuration
```bash
export RAY_LOGGING_LEVEL=DEBUG
export RAY_LOG_TO_DRIVER=true
export RAY_DEDUP_LOGS=0
export RAY_LOGGING_ENCODING=JSON
export RAY_LOGGING_ADDITIONAL_ATTRS=name,funcName,lineno
```
**Result**: Full Ray internals visible. All messages duplicated. Structured output with code locations.

### Structured Logging for Analysis
```bash
export RAY_LOGGING_ENCODING=JSON
export RAY_LOGGING_ADDITIONAL_ATTRS=name,funcName,lineno,thread,process
```
**Result**: Machine-parseable JSON with metadata for log aggregation systems.

### Custom Storage Limits
```bash
# 10GB total log storage (512MB × 20 files)
export RAY_LOGGING_ROTATE_BYTES=536870912
export RAY_LOGGING_ROTATE_BACKUP_COUNT=19
```
**Result**: Automatic cleanup when logs exceed 10GB. Oldest files removed first.

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

### JSON encoding
```json
{"asctime": "2024-01-15 10:30:15,123", "levelname": "INFO", "filename": "worker.py", "lineno": 1234, "message": "Task started", "job_id": "01000000", "worker_id": "abc123", "task_id": "def456"}
```

## Storage and Performance

**Default Configuration**:
- Log files stored in `/tmp/ray/session_*/logs/` per node
- 20GB maximum storage (1GB active + 19GB backups)
- Automatic rotation prevents disk exhaustion

**Performance Impact**:
- `RAY_LOGGING_LEVEL=ERROR`: ~2% CPU reduction in logging overhead
- `RAY_LOG_TO_DRIVER=false`: Reduces driver memory usage by ~100MB in large clusters
- `RAY_DEDUP_LOGS=1`: ~50% reduction in duplicate message processing

## Implementation Notes

- Environment variables must be set before pipeline initialization
- Invalid values automatically fall back to defaults with warning messages
- Configuration changes require pipeline restart to take effect
- Log deduplication uses message content hashing for efficiency
