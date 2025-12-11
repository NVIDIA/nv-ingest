# Deploy Without Containers (Library Mode) for NeMo Retriever Extraction

[NeMo Retriever extraction](overview.md) is typically deployed as a cluster of containers for robust, scalable production use. 

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.

In addition, you can use library mode, which is intended for the following cases:

- Local development
- Experimentation and testing
- Small-scale workloads, such as workloads of fewer than 100 documents


By default, library mode depends on NIMs that are hosted on build.nvidia.com. 
In library mode you launch the main pipeline service directly within a Python process, 
while all other services (such as embedding and storage) are hosted remotely in the cloud.

To get started using library mode, you need the following:

- Linux operating systems (Ubuntu 22.04 or later recommended) or MacOS
- Python 3.12
- We strongly advise using an isolated Python virtual env, such as provided by [uv](https://docs.astral.sh/uv/getting-started/installation/) or [conda](https://github.com/conda-forge/miniforge)



## Step 1: Prepare Your Environment

Use the following procedure to prepare your environment.

1. Run the following code to create your NV Ingest Conda environment.

    ```
       uv venv --python 3.12 nvingest && \
         source nvingest/bin/activate && \
         uv pip install nv-ingest==25.9.0 nv-ingest-api==25.9.0 nv-ingest-client==25.9.0 milvus-lite==2.4.12
    ```

    !!! tip

        To confirm that you have activated your Conda environment, run `which python` and confirm that you see `nvingest` in the result. You can do this before any python command that you run.

2. Set or create a .env file that contains your NVIDIA Build API key and other environment variables.

    !!! note

        If you have an NGC API key, you can use it here. For more information, refer to [Generate Your NGC Keys](ngc-api-key.md) and [Environment Configuration Variables](environment-config.md).

    - To set your variables, use the following code.

        ```
        export NVIDIA_API_KEY=nvapi-<your key>
        ```
    - To add your variables to a .env file, include the following.

        ```
        NVIDIA_API_KEY=nvapi-<your key>
        ```


## Step 2: Ingest Documents

You can submit jobs programmatically by using Python.

!!! tip

    For more Python examples, refer to [NV-Ingest: Python Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/python_client_usage.ipynb).


If you have a very high number of CPUs, and see the process hang without progress, 
we recommend that you use `taskset` to limit the number of CPUs visible to the process. 
Use the following code.

```
taskset -c 0-3 python your_ingestion_script.py
```

On a 4 CPU core low end laptop, the following code should take about 10 seconds.

```python
import time

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

def main():
    # Start the pipeline subprocess for library mode
    run_pipeline(block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    # gpu_cagra accelerated indexing is not available in milvus-lite
    # Provide a filename for milvus_uri to use milvus-lite
    milvus_uri = "milvus.db"
    collection_name = "test"
    sparse = False

    # do content extraction from files
    ingestor = (
        Ingestor(client=client)
        .files("data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            table_output_format="markdown",
            extract_infographics=True,
            # extract_method="nemoretriever_parse", #Slower, but maximally accurate, especially for PDFs with pages that are scanned images
            text_depth="page",
        )
        .embed()
        .vdb_upload(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            sparse=sparse,
            # for llama-3.2 embedder, use 1024 for e5-v5
            dense_dim=2048,
        )
    )

    print("Starting ingestion..")
    t0 = time.time()

    # Return both successes and failures
    # Use for large batches where you want successful chunks/pages to be committed, while collecting detailed diagnostics for failures.
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)

    # Return only successes
    # results = ingestor.ingest(show_progress=True)

    t1 = time.time()
    print(f"Total time: {t1 - t0} seconds")

    # results blob is directly inspectable
    if results:
        print(ingest_json_results_to_blob(results[0]))

    # (optional) Review any failures that were returned
    if failures:
        print(f"There were {len(failures)} failures. Sample: {failures[0]}")

if __name__ == "__main__":
    main()
```

!!! note

    To use library mode with nemoretriever_parse, uncomment `extract_method="nemoretriever_parse"` in the previous code. For more information, refer to [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md).


You can see the extracted text that represents the content of the ingested test document.

```shell
Starting ingestion..
Total time: 9.243880033493042 seconds

TestingDocument
A sample document with headings and placeholder text
Introduction
This is a placeholder document that can be used for any purpose. It contains some 
headings and some placeholder text to fill the space. The text is not important and contains 
no real value, but it is useful for testing. Below, we will have some simple tables and charts 
that we can use to confirm Ingest is working as expected.
Table 1
This table describes some animals, and some activities they might be doing in specific 
locations.
Animal Activity Place
Gira@e Driving a car At the beach
Lion Putting on sunscreen At the park
Cat Jumping onto a laptop In a home o@ice
Dog Chasing a squirrel In the front yard
Chart 1
This chart shows some gadgets, and some very fictitious costs.

... document extract continues ...
```

## Step 3: Query Ingested Content

To query for relevant snippets of the ingested content, and use them with an LLM to generate answers, use the following code.

```python
import os
from openai import OpenAI
from nv_ingest_client.util.milvus import nvingest_retrieval

milvus_uri = "milvus.db"
collection_name = "test"
sparse=False

queries = ["Which animal is responsible for the typos?"]

retrieved_docs = nvingest_retrieval(
    queries,
    collection_name,
    milvus_uri=milvus_uri,
    hybrid=sparse,
    top_k=1,
)

# simple generation example
extract = retrieved_docs[0][0]["entity"]["text"]
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.environ["NVIDIA_API_KEY"]
)

prompt = f"Using the following content: {extract}\n\n Answer the user query: {queries[0]}"
print(f"Prompt: {prompt}")
completion = client.chat.completions.create(
  model="nvidia/llama-3.1-nemotron-70b-instruct",
  messages=[{"role":"user","content": prompt}],
)
response = completion.choices[0].message.content

print(f"Answer: {response}")
```

```shell
Prompt: Using the following content: Table 1
| This table describes some animals, and some activities they might be doing in specific locations. | This table describes some animals, and some activities they might be doing in specific locations. | This table describes some animals, and some activities they might be doing in specific locations. |
| Animal | Activity | Place |
| Giraffe | Driving a car | At the beach |
| Lion | Putting on sunscreen | At the park |
| Cat | Jumping onto a laptop | In a home office |
| Dog | Chasing a squirrel | In the front yard |

 Answer the user query: Which animal is responsible for the typos?
Answer: A clever query!

Based on the provided Table 1, I'd make an educated inference to answer your question. Since the activities listed are quite unconventional for the respective animals (e.g., a giraffe driving a car, a lion putting on sunscreen), it's likely that the table is using humor or hypothetical scenarios.

Given this context, the question "Which animal is responsible for the typos?" is probably a tongue-in-cheek inquiry, as there's no direct information in the table about typos or typing activities.

However, if we were to make a playful connection, we could look for an animal that's:

1. Typically found in a setting where typing might occur (e.g., an office).
2. Engaging in an activity that could potentially lead to typos (e.g., interacting with a typing device).

Based on these loose criteria, I'd jokingly point to:

**Cat** as the potential culprit, since it's:
        * Located "In a home office"
        * Engaged in "Jumping onto a laptop", which could theoretically lead to accidental keystrokes or typos if the cat were to start "walking" on the keyboard!

Please keep in mind that this response is purely humorous and interpretative, as the table doesn't explicitly mention typos or provide a straightforward answer to the question.
```



## Logging Configuration

Nemo Retriever extraction uses [Ray](https://docs.ray.io/en/latest/index.html) for logging. 
For details, refer to [Configure Ray Logging](ray-logging.md).

By default, library mode runs in quiet mode to minimize startup noise. 
Quiet mode automatically configures the following environment variables.

| Variable                             | Quiet Mode Value | Description |
|--------------------------------------|------------------|-------------|
| `INGEST_RAY_LOG_LEVEL`               | `PRODUCTION`     | Sets Ray logging to ERROR level to reduce noise. |
| `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` | `0`              | Silences Ray accelerator warnings |
| `OTEL_SDK_DISABLED`                  | `true`           | Disables OpenTelemetry trace export errors |


If you want to see detailed startup logs for debugging, use one of the following options:

- Set `quiet=False` when you run the pipeline as shown following.

    ```python
    run_pipeline(block=False, disable_dynamic_scaling=True, run_in_subprocess=True, quiet=False)
    ```

- Set the environment variables manually before you run the pipeline as shown following.

    ```bash
    export INGEST_RAY_LOG_LEVEL=DEVELOPMENT  # or DEBUG for maximum verbosity
    ```



## Library Mode Communication and Advanced Examples

Communication in library mode is handled through a simplified, 3-way handshake message broker called `SimpleBroker`.

Attempting to run a library-mode process co-located with a Docker Compose deployment does not work by default. 
The Docker Compose deployment typically creates a firewall rule or port mapping that captures traffic to port `7671`,
which prevents the `SimpleBroker` from receiving messages. 
Always ensure that you use library mode in isolation, without an active containerized deployment listening on the same port.


### Example `launch_libmode_service.py`

This example launches the pipeline service in a subprocess, 
and keeps it running until it is interrupted (for example, by pressing `Ctrl+C`). 
It listens for ingestion requests on port `7671` from an external client.

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

### Example `launch_libmode_and_run_ingestor.py`

This example starts the pipeline service in-process, 
and immediately runs an ingestion client against it in the same parent process.

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



## The `run_pipeline` Function Reference

The `run_pipeline` function is the main entry point to start the Nemo Retriever Extraction pipeline. 
It can run in-process or as a subprocess.

The `run_pipeline` function accepts the following parameters.

| Parameter                | Type                   | Default | Required? | Description                                     |
|--------------------------|------------------------|---------|-----------|-------------------------------------------------|
| ingest_config            | PipelineCreationSchema | —       | Yes       | A configuration object that specifies how the pipeline should be constructed. |
| run_in_subprocess        | bool                   | False   | Yes       | `True` to launch the pipeline in a separate Python subprocess. `False` to run in the current process. |
| block                    | bool                   | True    | Yes       | `True` to run the pipeline synchronously. The function returns after it finishes. `False` to return an interface for external pipeline control. |
| disable_dynamic_scaling  | bool                   | None    | No        | `True` to disable autoscaling regardless of global settings. `None` to use the global default behavior. |
| dynamic_memory_threshold | float                  | None    | No        | A value between `0.0` and `1.0`. If dynamic scaling is enabled, triggers autoscaling when memory usage crosses this threshold. |
| stdout                   | TextIO                 | None    | No        | Redirect the subprocess `stdout` to a file or stream. If `None`, defaults to `/dev/null`. |
| stderr                   | TextIO                 | None    | No        | Redirect subprocess `stderr` to a file or stream. If `None`, defaults to `/dev/null`. |
| libmode                  | bool                   | True    | No        | `True` to load the default library mode pipeline configuration when `ingest_config` is `None`. |
| quiet                    | bool                   | None    | No        | `True` to suppress verbose startup logs (PRODUCTION preset). `None` defaults to `True` when `libmode=True`. Set to `False` for verbose output. |


The `run_pipeline` function returns the following values, depending on the parameters that you set:

- **run_in_subprocess=False and block=True**  — The function returns a `float` that represents the elapsed time in seconds.
- **run_in_subprocess=False and block=False** — The function returns a `RayPipelineInterface` object.
- **run_in_subprocess=True  and block=True**  — The function returns `0.0`.
- **run_in_subprocess=True  and block=False** — The function returns a `RayPipelineInterface` object.


The `run_pipeline` throws the following errors:

- **RuntimeError** — A subprocess failed to start, or exited with error.
- **Exception** — Any other failure during pipeline setup or execution.



## Related Topics

- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy With Helm](helm.md)
- [Notebooks](notebooks.md)
- [Multimodal PDF Data Extraction](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
