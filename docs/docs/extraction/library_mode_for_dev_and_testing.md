# NV-Ingest: Library-Mode Execution

## Overview

NV-Ingest is typically deployed as a cluster of containers for robust, scalable production use. However, for lightweight testing and experimentation, we also provide a **Library-Mode** variant. This mode allows users to launch the main pipeline service directly within a Python process while assuming all other services (e.g., embedding, storage, OCR) are hosted remotely in the cloud.

This setup is not intended for production, but is ideal for:
- Local development
- Quick-start testing
- Experimentation

Communication in this mode is handled via a simplified, 3-way handshake message broker called `SimpleBroker`.

## Example Files

### 1. `launch_libmode_service.py`

This example launches the pipeline service in a subprocess and keeps it running until interrupted (e.g., via `Ctrl+C`). It listens for ingestion requests on port `7671` from an external client.

```python
def main():
config_data = {}

    config_data = {key: value for key, value in config_data.items() if value is not None}
    ingest_config = PipelineCreationSchema(**config_data)

    try:
        _ = run_pipeline(
            ingest_config,
            block=True,
            disable_dynamic_scaling=True,
            run_in_subprocess=True,
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
```

### 2. `launch_libmode_and_run_ingestor.py`

This variant starts the service in-process and immediately runs an ingestion client against it in the same parent process.

```python
def run_ingestor():
client = NvIngestClient(
message_client_allocator=SimpleClient,
message_client_port=7671,
message_client_hostname="localhost"
)

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            paddle_output_format="markdown",
            extract_infographics=False,
            text_depth="page",
        )
        .split(chunk_size=1024, chunk_overlap=150)
    )

    try:
        results, failures = ingestor.ingest(show_progress=False, return_failures=True)
        logger.info("Ingestion completed successfully.")
        if failures:
            logger.warning(f"Ingestion completed with {len(failures)} failures:")
            for i, failure in enumerate(failures):
                logger.warning(f"  [{i}] {failure}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

    print("\nIngest done.")
    print(f"Got {len(results)} results.")

def main():
config_data = {}
config_data = {key: value for key, value in config_data.items() if value is not None}
ingest_config = PipelineCreationSchema(**config_data)

    try:
        pipeline = run_pipeline(
            ingest_config,
            block=False,
            disable_dynamic_scaling=True,
            run_in_subprocess=True,
        )
        time.sleep(10)
        run_ingestor()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
    finally:
        pipeline.stop()
        logger.info("Shutting down pipeline...")

if __name__ == "__main__":
main()
```

---

## `run_pipeline` Function

The `run_pipeline` function is the main entry point to start an NV-Ingest pipeline. It supports both in-process and subprocess execution.

### Parameters

- **`ingest_config: PipelineCreationSchema`**  
  Configuration object specifying how the pipeline should be constructed.

- **`block: bool = True`**
    - `True`: Run the pipeline synchronously; function returns after completion.
    - `False`: Return an interface for external pipeline control.

- **`disable_dynamic_scaling: Optional[bool] = None`**
    - `True`: Disable autoscaling regardless of global settings.
    - `None`: Use the global default behavior.

- **`dynamic_memory_threshold: Optional[float] = None`**
    - A value between `0.0` and `1.0`. If dynamic scaling is enabled, triggers autoscaling when memory usage crosses this threshold.

- **`run_in_subprocess: bool = False`**
    - `True`: Launch the pipeline in a separate Python subprocess.
    - `False`: Run inline within the current process.

- **`stdout: Optional[TextIO] = None`**
    - Redirect subprocess `stdout` to a file or stream.
    - If `None`, defaults to `/dev/null`.

- **`stderr: Optional[TextIO] = None`**
    - Redirect subprocess `stderr` to a file or stream.
    - If `None`, defaults to `/dev/null`.

### Returns

Depending on configuration:

- **In-process, `block=True`** → `float` (elapsed time in seconds)
- **In-process, `block=False`** → `RayPipelineInterface`
- **Subprocess, `block=True`** → `0.0` (on completion)
- **Subprocess, `block=False`** → `RayPipelineSubprocessInterface`

### Exceptions

- **`RuntimeError`** — Subprocess failed to start or exited with error.
- **`Exception`** — Any other failure during pipeline setup or execution.

> **Note:**  
> Attempting to run a library-mode process **co-located** with a Docker Compose deployment will not work out of the box.  
> The Docker Compose deployment typically creates a firewall rule or port mapping that captures traffic to port `7671`.  
> This will prevent the `SimpleBroker` in library-mode from receiving messages, as traffic will be redirected away from it.  
> Always ensure that libmode is run in isolation, without an active containerized deployment listening on the same port.
