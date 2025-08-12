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

- Linux operating systems (Ubuntu 22.04 or later recommended)
- Python 3.12
- We strongly advise using an isolated Python virtual env, such as provided by [uv](https://docs.astral.sh/uv/getting-started/installation/) or [conda](https://github.com/conda-forge/miniforge)



## Step 1: Prepare Your Environment

Use the following procedure to prepare your environment.

1. Run the following code to create your NV Ingest Conda environment.

    ```
       uv venv --python 3.12 nvingest && \
         source nvingest/bin/activate && \
         uv pip install nv-ingest==25.6.2 nv-ingest-api==25.6.2 nv-ingest-client==25.6.2
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
import logging, os, time, sys

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# Start the pipeline subprocess for library mode
config = PipelineCreationSchema()

run_pipeline(config, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
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
        paddle_output_format="markdown",
        extract_infographics=True,
        # Slower, but maximally accurate, especially for PDFs with pages that are scanned images
        # extract_method="nemoretriever_parse",
        text_depth="page"
    ).embed()
    .vdb_upload(
        collection_name=collection_name,
        milvus_uri=milvus_uri,
        sparse=sparse,
        # for llama-3.2 embedder, use 1024 for e5-v5
        dense_dim=2048
    )
)

print("Starting ingestion..")
t0 = time.time()
results, failures = ingestor.ingest(show_progress=True, return_failures=True)
t1 = time.time()
print(f"Time taken: {t1 - t0} seconds")

# results blob is directly inspectable
print(ingest_json_results_to_blob(results[0]))
if failures:
    print(f"There were {len(failures)} failures. Sample: {failures[0]}")
```

!!! note

    To use library mode with nemoretriever_parse, uncomment `extract_method="nemoretriever_parse"` in the previous code. For more information, refer to [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md).

!!! important "About return_failures and vdb_upload"

    - `ingestor.ingest(..., return_failures=False)` (default): returns only successful results. If `.vdb_upload(...)` is configured and any jobs fail, `ingest()` raises `RuntimeError` and does not upload (all-or-nothing).
    - `ingestor.ingest(..., return_failures=True)`: returns `(results, failures)`. If `.vdb_upload(...)` is configured and some jobs fail, `ingest()` uploads only the successful results and does not raise; inspect `failures` for remediation.

You can see the extracted text that represents the content of the ingested test document.

```shell
Starting ingestion..
Time taken: 9.243880033493042 seconds

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
from openai import OpenAI
from nv_ingest_client.util.milvus import nvingest_retrieval
import os

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
Prompt: Using the following content: TestingDocument
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

 Answer the user query: Which animal is responsible for the typos?
Answer: A clever query!

After carefully examining the provided content, I'd like to point out the potential "typos" (assuming you're referring to the unusual or intentionally incorrect text) and attempt to playfully "assign blame" to an animal based on the context:

1. **Gira@e** (instead of Giraffe) - **Animal blamed: Giraffe** (Table 1, first row)
	* The "@" symbol in "Gira@e" suggests a possible typo or placeholder character, which we'll humorously attribute to the Giraffe's alleged carelessness.
2. **o@ice** (instead of Office) - **Animal blamed: Cat**
	* The same "@" symbol appears in "o@ice", which is related to the Cat's activity in the same table. Perhaps the Cat was in a hurry while typing and introduced the error?

So, according to this whimsical analysis, both the **Giraffe** and the **Cat** are "responsible" for the typos, with the Giraffe possibly being the more egregious offender given the more blatant character substitution in its name.
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
