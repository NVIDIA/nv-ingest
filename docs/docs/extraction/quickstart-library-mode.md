# Deploy Without Containers (Library Mode) for NeMo Retriever Extraction

For small-scale workloads, such as workloads of fewer than 100 documents, you can use library mode setup. 
Library mode depends on NIMs that are already self-hosted, or, by default, NIMs that are hosted on build.nvidia.com.

To get started using [NeMo Retriever extraction](overview.md) in library mode, you need the following:

- Linux operating systems (Ubuntu 22.04 or later recommended)
- Python 3.12
- We strongly advise using an isolated Python virtual env, such as provided by [uv](https://docs.astral.sh/uv/getting-started/installation/) or [conda](https://github.com/conda-forge/miniforge)



## Step 1: Prepare Your Environment

Use the following procedure to prepare your environment.

1. Run the following code to create your NV Ingest Conda environment.

    ```
       uv venv --python 3.12 nvingest && \
         source nvingest/bin/activate && \
         uv pip install nv-ingest==25.6.2 nv-ingest-api==25.6.2 nv-ingest-client==25.6.3
    ```

    !!! tip

        To confirm that you have activated your Conda environment, run `which python` and confirm that you see `nvingest` in the result. You can do this before any python command that you run.

2. Set or create a .env file that contains your NVIDIA Build API key and other environment variables.

    !!! note

        If you use an NGC personal key, then you should provide the same value for all keys, but you must specify each environment variable individually. In the past, you could create an API key. If you have an API key, you can still use that. For more information, refer to [Generate Your NGC Keys](ngc-api-key.md) and [Environment Configuration Variables](environment-config.md).

    - To set your variables, use the following code.

        ```
        export NVIDIA_BUILD_API_KEY=nvapi-<your key>
        export NVIDIA_API_KEY=nvapi-<your key>
        ```
    - To add your variables to an .env file, include the following.

        ```
        NVIDIA_BUILD_API_KEY=nvapi-<your key>
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
results = ingestor.ingest(show_progress=True)
t1 = time.time()
print(f"Time taken: {t1 - t0} seconds")

# results blob is directly inspectable
print(ingest_json_results_to_blob(results[0]))
```

!!! note

    To use library mode with nemoretriever_parse, uncomment `extract_method="nemoretriever_parse"` in the previous code. For more information, refer to [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md).

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
  api_key = os.environ["NVIDIA_BUILD_API_KEY"]
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



## Related Topics

- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy With Helm](helm.md)
- [Notebooks](notebooks.md)
- [Multimodal PDF Data Extraction](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
