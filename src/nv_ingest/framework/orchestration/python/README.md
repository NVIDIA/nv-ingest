# Python Orchestration Framework

This directory contains a simplified Python-based orchestration framework for nv-ingest that operates without Ray dependencies. It provides a lightweight alternative for local processing and testing.

## Overview

The Python orchestration framework consists of:

1. **SimpleBroker Starter** - Utility to start a SimpleBroker in a separate process
2. **Message Broker Task Source** - Reads messages from the broker locally (no Ray)
3. **Message Broker Task Sink** - Writes processed messages back to the broker locally (no Ray)
4. **Pipeline Orchestrator** - Chains Python functions between source and sink

## Architecture

```
Client → SimpleBroker → Source → [Processing Functions] → Sink → SimpleBroker → Client
```

The framework enables:
- **No-op Pipeline**: Messages pass through unchanged for testing
- **Function Chain Pipeline**: Apply a sequence of Python functions to messages
- **Local Processing**: All components run locally without distributed computing overhead

## Components

### 1. Broker Starter (`broker_starter.py`)

Starts a SimpleBroker server in a separate process.

```python
from nv_ingest.framework.orchestration.python import start_simple_message_broker

broker_config = {
    "host": "localhost",
    "port": 8080,
    "broker_params": {
        "max_queue_size": 1000
    }
}

broker_process = start_simple_message_broker(broker_config)
```

### 2. Message Broker Task Source (`message_broker_task_source.py`)

Reads messages from the broker and converts them to `IngestControlMessage` objects.

```python
from nv_ingest.framework.orchestration.python import PythonMessageBrokerTaskSource
from nv_ingest.framework.orchestration.python.message_broker_task_source import (
    PythonMessageBrokerTaskSourceConfig,
    SimpleClientConfig
)

source_config = PythonMessageBrokerTaskSourceConfig(
    broker_client=SimpleClientConfig(
        client_type="simple",
        host="localhost",
        port=8080
    ),
    task_queue="task_queue",
    poll_interval=0.1
)

source = PythonMessageBrokerTaskSource(source_config)
```

### 3. Message Broker Task Sink (`message_broker_task_sink.py`)

Processes messages and sends results back to the broker.

```python
from nv_ingest.framework.orchestration.python import PythonMessageBrokerTaskSink
from nv_ingest.framework.orchestration.python.message_broker_task_sink import (
    PythonMessageBrokerTaskSinkConfig,
    SimpleClientConfig
)

sink_config = PythonMessageBrokerTaskSinkConfig(
    broker_client=SimpleClientConfig(
        client_type="simple",
        host="localhost",
        port=8080
    ),
    poll_interval=0.1
)

sink = PythonMessageBrokerTaskSink(sink_config)
```

### 4. Pipeline Orchestrator (`pipeline.py`)

Orchestrates the flow of messages through a chain of processing functions.

```python
from nv_ingest.framework.orchestration.python import PythonPipeline

def my_processing_function(message):
    # Process the message
    return message

pipeline = PythonPipeline(
    source=source,
    sink=sink,
    processing_functions=[my_processing_function]
)

pipeline.run()
```

## Usage Examples

### No-op Pipeline (Testing)

```python
# Create pipeline with no processing functions
pipeline = PythonPipeline(source, sink, processing_functions=[])
pipeline.run(max_iterations=10)
```

### Custom Processing Pipeline

```python
def add_metadata(message):
    """Add processing metadata to message."""
    for task in message.tasks:
        task.task_properties['processed_at'] = time.time()
    return message

def validate_content(message):
    """Validate message content."""
    # Validation logic here
    return message

pipeline = PythonPipeline(
    source=source,
    sink=sink,
    processing_functions=[add_metadata, validate_content]
)

pipeline.run()
```

## Configuration

Both source and sink support Redis and Simple broker clients:

### Simple Client Configuration

```python
SimpleClientConfig(
    client_type="simple",
    host="localhost",
    port=8080,
    max_retries=5,
    max_backoff=5.0,
    connection_timeout=30.0
)
```

### Redis Client Configuration

```python
RedisClientConfig(
    client_type="redis",
    host="localhost",
    port=6379,
    max_retries=5,
    max_backoff=5.0,
    connection_timeout=30.0,
    broker_params=BrokerParamsRedis(
        db=0,
        use_ssl=False
    )
)
```

## Testing

### Running the No-op Pipeline Test

```bash
cd src/nv_ingest/framework/orchestration/python
python test_noop_pipeline.py
```

This test:
1. Starts a SimpleBroker
2. Creates source and sink components
3. Runs a no-op pipeline
4. Submits test messages
5. Verifies messages pass through successfully

### Running the Example

```bash
cd src/nv_ingest/framework/orchestration/python
python example_usage.py
```

This example demonstrates a pipeline with custom processing functions.

## API Reference

### PythonMessageBrokerTaskSource

**Methods:**
- `get_message()` - Get the next message from the broker
- `start()` - Start the source
- `stop()` - Stop the source
- `pause()` - Pause message fetching
- `resume()` - Resume message fetching
- `get_stats()` - Get source statistics

### PythonMessageBrokerTaskSink

**Methods:**
- `process_message(control_message)` - Process and send a message
- `start()` - Start the sink
- `stop()` - Stop the sink
- `get_stats()` - Get sink statistics

### PythonPipeline

**Methods:**
- `run(max_iterations=None, poll_interval=0.1)` - Run the pipeline
- `run_single_iteration()` - Run one pipeline iteration
- `start()` - Start the pipeline
- `stop()` - Stop the pipeline
- `get_stats()` - Get pipeline statistics

## Differences from Ray Implementation

| Feature | Ray Implementation | Python Implementation |
|---------|-------------------|----------------------|
| **Execution** | Distributed across Ray actors | Local execution in single process |
| **Concurrency** | Ray's actor model | Threading for pause/resume |
| **Scalability** | Horizontal scaling | Single machine |
| **Dependencies** | Requires Ray cluster | No external dependencies |
| **Use Case** | Production workloads | Development, testing, simple processing |

## Future Enhancements

The framework is designed to be extensible. Future enhancements may include:

1. **Multi-threading Support** - Process messages in parallel threads
2. **Batch Processing** - Process multiple messages at once
3. **Error Recovery** - Advanced error handling and retry mechanisms
4. **Metrics Collection** - Enhanced monitoring and metrics
5. **Configuration Validation** - More robust configuration validation
6. **Plugin System** - Pluggable processing functions

## Integration with Existing Clients

The Python orchestration framework is compatible with existing nv-ingest clients. Clients can submit jobs to the SimpleBroker using the same interface, and the Python pipeline will process them locally.

```python
# Existing client code works unchanged
from nv_ingest_client.client import NvIngestClient

client = NvIngestClient(
    message_client_hostname="localhost",
    message_client_port=8080,
    message_client_type="simple"
)

# Submit jobs as usual
job_result = client.submit_job(job_spec)
```

This enables a smooth transition from Ray-based processing to Python-based processing for development and testing scenarios.
