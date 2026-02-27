# Configure Ray Logging

[NeMo Retriever Library](overview.md) uses [Ray](https://docs.ray.io/en/latest/index.html) for logging. 
You can use environment variables for fine-grained control over [Ray's logging behavior](https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html). 
In addition, NeMo Retriever Library provides preset configurations that you can use to quickly update Ray logging behavior.


!!! important

    You must set environment variables before you initialize the pipeline, and you must restart the pipeline if you change variable values.



## Quick Start - Use Preset Configurations

To get started quickly, use one of the NeMo Retriever Library package-level preset variables. 
Run the code below that corresponds to your use case; production, development, or debugging. 
The log levels are explained following.

!!! tip

    After you set a preset configuration, you can also override individual variables.

```bash
# Production deployment - minimal logging, maximum performance
export INGEST_RAY_LOG_LEVEL=PRODUCTION

# Development work (default) - balanced logging
export INGEST_RAY_LOG_LEVEL=DEVELOPMENT  

# Debugging issues - maximum logging and visibility
export INGEST_RAY_LOG_LEVEL=DEBUG
```


### PRODUCTION Log Level

The `PRODUCTION` log level is optimized for production deployments with minimal logging overhead. 

- **Storage Limit** – 10GB total (1GB × 10 files)
- **Performance Impact** – ~5% CPU reduction, ~200MB memory savings in large clusters

This log level uses the following settings:

- **Log Level** – ERROR only
- **Log to Driver** – Disabled (worker logs stay in worker files)
- **Import Warnings** – Disabled
- **Usage Stats** – Disabled  
- **Storage** – 10GB total (1GB × 10 files)
- **Deduplication** – Enabled
- **Encoding** – TEXT


### DEVELOPMENT Log Level

The `DEVELOPMENT` log level is a balanced configuration for development work, 
and is the default log level.

- **Storage Limit** – 20GB total (1GB × 20 files)  
- **Performance Impact** – Balanced performance and visibility

This log level uses the following settings:

- **Log Level** – INFO
- **Log to Driver** – Enabled
- **Import Warnings** – Enabled
- **Usage Stats** – Enabled
- **Storage** – 20GB total (1GB × 20 files) 
- **Deduplication** – Enabled
- **Encoding** – TEXT


### DEBUG Log Level

The `DEBUG` log level provides maximum visibility for troubleshooting issues. 

- **Storage Limit** – 20GB total (512MB × 40 files)
- **Performance Impact** – ~10% CPU overhead for detailed logging, higher memory usage

This log level uses the following settings:

- **Log Level** – DEBUG
- **Log to Driver** – Enabled
- **Import Warnings** – Enabled
- **Usage Stats** – Enabled
- **Storage** – 20GB total (512MB × 40 files)
- **Deduplication** – Disabled (see all duplicate messages)
- **Encoding** – JSON with function names and line numbers



## Configuration Reference

The following are the environment variables that you can set to control Ray logging behavior. 
If you specify an invalid value, the variable reverts to the default value with a warning message.

| Variable                          | Type                             | Description | Valid Values | Default | 
|-----------------------------------|----------------------------------|-------------|--------------|---------| 
| `INGEST_RAY_LOG_LEVEL`            | NeMo Retriever Library preset | Set multiple Ray logging variables to optimize for specific use cases. | `PRODUCTION`, `DEVELOPMENT`, `DEBUG` | `DEVELOPMENT` | 
| `RAY_DEDUP_LOGS`                  | Log flow control                 | Specify whether to log multiple instances of repeated events or to combine into a single entry. 1 to combine repeated messages (for example, `[repeated 5x]`). | `0`, `1` | `1` | 
| `RAY_DISABLE_IMPORT_WARNING`      | Ray internal logging             | `1` to suppresses `Ray X.Y.Z started` message and other warnings during initialization. | `0`, `1` | `0` | 
| `RAY_LOG_TO_DRIVER`               | Log flow control                 | `true`to log worker messages in the main process. `false` to log worker messages in worker log files. | `true`, `false` | `true` | 
| `RAY_LOGGING_ADDITIONAL_ATTRS`    | Core logging control             | Add Python logger fields like function names, line numbers to each log entry. | Comma-separated list | (empty) | 
| `RAY_LOGGING_ENCODING`            | Core logging control             | Specify the format for log messages. | `TEXT`, `JSON` | `TEXT` | 
| `RAY_LOGGING_LEVEL`               | Core logging control             | Specify what events to log. `DEBUG` to log all Ray internals. `WARNING` to log only significant events. | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` | 
| `RAY_LOGGING_ROTATE_BACKUP_COUNT` | File rotation                    | Specify the number of old log files retained. Total storage = (count + 1) × file size. | Integer | `19` | 
| `RAY_LOGGING_ROTATE_BYTES`        | File rotation                    | Specify the log file size before Ray creates a new log file. Use this to prevent unbounded disk usage. | Bytes | `1073741824` (1GB) | 
| `RAY_USAGE_STATS_ENABLED`         | Ray internal logging             | `1` to enable telemetry collection and related log messages. `0` to disable.  | `0`, `1` | `1` | 



## Configuration Examples

### Use a Preset With A Manual Override

The following example uses the `DEVELOPMENT` preset and then overrides the `RAY_LOGGING_LEVEL` behavior.

```bash
export INGEST_RAY_LOG_LEVEL=DEVELOPMENT  # Use the DEVELOPMENT preset
export RAY_LOGGING_LEVEL=WARNING         # Override just the log level
```

### Log Verbosity Control

By default, Ray generates significant logging output. 
The following example configures Ray to reduce log volume.

```bash
export RAY_DISABLE_IMPORT_WARNING=1  # Suppress Ray initialization warnings
export RAY_LOGGING_LEVEL=WARNING     # Suppress informational messages, show only warnings and errors
export RAY_LOG_TO_DRIVER=false       # Prevent worker logs from appearing in driver process output
```


### Minimal Logging (Legacy)

The following example minimizes logging. 
Only critical errors are logged. 
Worker logs are isolated. 
This reduces log volume by approximately 95%.

!!! tip

    You can achieve the same effect by setting the `INGEST_RAY_LOG_LEVEL` to `PRODUCTION`.

```bash
export RAY_LOGGING_LEVEL=ERROR
export RAY_LOG_TO_DRIVER=false
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=1
```


### Structured Logging for Analysis

The following example results in machine-parseable JSON with metadata for log aggregation systems.

```bash
export INGEST_RAY_LOG_LEVEL=DEVELOPMENT
export RAY_LOGGING_ENCODING=JSON
export RAY_LOGGING_ADDITIONAL_ATTRS=name,funcName,lineno,thread,process
```


### Set Custom Storage Limits

The following example automatically cleans up files when logs exceed 5GB. The oldest files are removed first.

```bash
# 5GB total log storage (500MB × 10 files)
export RAY_LOGGING_ROTATE_BYTES=524288000
export RAY_LOGGING_ROTATE_BACKUP_COUNT=9
```



## Log Output Examples

### INFO level (Default)

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
{
    "asctime": "2024-01-15 10:30:15,123", 
    "levelname": "INFO", 
    "filename": "worker.py", 
    "lineno": 1234, 
    "message": "Task started", 
    "name": "ray.worker", 
    "funcName": "execute_task", 
    "job_id": "01000000", 
    "worker_id": "abc123", 
    "task_id": "def456"}
```


## Related Topics

- [Environment Variables](https://docs.nvidia.com/nemo/retriever/latest/extraction/environment-config/)
