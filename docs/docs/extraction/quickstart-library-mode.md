# Quickstart Guide for NeMo Retriever Extraction (Library Mode)

For small-scale workloads, such as workloads of fewer than 100 documents, you can use library mode setup. 
Library mode depends on NIMs that are already self-hosted, or, by default, NIMs that are hosted on build.nvidia.com.

To get started using [NeMo Retriever extraction](overview.md) in library mode, you need the following:

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [Conda Python environment and package manager](https://github.com/conda-forge/miniforge)
- [Python version 3.10](https://www.python.org/downloads/release/python-3100/)



## Step 1: Prepare Your Environment

Use the following procedure to prepare your environment.

1. Run the following code to create your NVIDIA Ingest Conda environment.

    ```
    conda create -y --name nvingest python=3.10
    conda activate nvingest
    conda install -y -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia nvidia/label/dev::nv_ingest nvidia/label/dev::nv_ingest_client nvidia/label/dev::nv_ingest_api
    pip install opencv-python llama-index-embeddings-nvidia pymilvus 'pymilvus[bulk_writer, model]' milvus-lite dotenv ffmpeg nvidia-riva-client
    ```

    !!! tip

        To confirm that you have activated your Conda environment, run `which pip` and `which python`, and confirm that you see `nvingest` in the result. You can do this before any pip or python command that you run.


2. Create a .env file that contains your NVIDIA Build API key. For more information, refer to [Environment Configuration Variables](environment-config.md).

    ```
    NVIDIA_BUILD_API_KEY=nvapi-<your key>
    NVIDIA_API_KEY=nvapi-<your key>

    PADDLE_HTTP_ENDPOINT=<endpoint>
    PADDLE_INFER_PROTOCOL=http
    YOLOX_HTTP_ENDPOINT=<endpoint>
    YOLOX_INFER_PROTOCOL=http
    YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT=<endpoint>
    YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL=http
    YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT=<endpoint>
    YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL=http
    EMBEDDING_NIM_ENDPOINT=<endpoint>
    EMBEDDING_NIM_MODEL_NAME=<model>
    VLM_CAPTION_ENDPOINT=<endpoint>
    VLM_CAPTION_MODEL_NAME=<model>
    # NemoRetriever Parse:
    NEMORETRIEVER_PARSE_HTTP_ENDPOINT=<enpoint>
    NEMORETRIEVER_PARSE_INFER_PROTOCOL=http
    ```



## Step 2: Ingest Documents

You can submit jobs programmatically by using Python.

!!! tip

    For more Python examples, refer to [NV-Ingest: Python Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/python_client_usage.ipynb).

```python
# must be first: pipeline subprocesses pickup config from env variables
from dotenv import load_dotenv
load_dotenv(".env")

import logging, os, time
from nv_ingest.util.logging.configuration import configure_logging as configure_local_logging

from nv_ingest.util.pipeline.pipeline_runners import start_pipeline_subprocess
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient
from nv_ingest.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# Start the pipeline subprocess for library mode                       
config = PipelineCreationSchema()                                                  
print(config)

pipeline_process = start_pipeline_subprocess(config)         

client = NvIngestClient(                                                                          
    message_client_allocator=SimpleClient,                                           
    message_client_port=7671,                                                                
    message_client_hostname="localhost"         
)                                                                  

# Note: gpu_cagra accelerated indexing is not yet available in milvus-lite
# Provide a filename for milvus_uri to use milvus-lite
milvus_uri = "milvus.db"                
collection_name = "test"      
sparse=False

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
        # extract_method="nemoretriever_parse", # Slower, but maximally accurate, especially for PDFs with pages that are scanned images
        text_depth="page"
    ).embed()
    .caption()
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
results = ingestor.ingest()
t1 = time.time()
print(f"Time taken: {t1-t0} seconds")

# results blob is directly inspectable
print(ingest_json_results_to_blob(results[0]))
```

!!! note

    To use library mode with nemoretriever_parse, uncomment `extract_method="nemoretriever_parse"` in the previous code. For more information, refer to [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md).

You can see the extracted text that represents the content of the ingested test document.

```shell
...
Chart 2
This chart shows some average frequency ranges for speaker drivers.
Conclusion
This is the conclusion of the document. It has some more placeholder text, but the most 
important thing is that this is the conclusion. As we end this document, we should have 
been able to extract 2 tables, 2 charts, and some text including 3 bullet points.
image_caption:[]

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
```



## Step 3: Query Ingested Content

To query for relevant snippets of the ingested content, and use it with LLMs to generate answers, use the following code.

```python
from openai import OpenAI
from nv_ingest_client.util.milvus import nvingest_retrieval

#queries = ["Where and what is the dog doing?"]
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
 Answer the user query: Which animal is responsible for the typos?
Answer: A clever query!

After carefully examining the provided content, I'll do my best to deduce the answer:
**Inferences and Observations:**
1. **Typos identification**: I've spotted potential typos in two places:
        * "Gira@e" (likely meant to be "Giraffe")
        * "o@ice" (likely meant to be "office")
2. **Association with typos**: Both typos are related to specific animal entries in "Table 1".

**Answer:**
Based on the observations, I'll playfully attribute the "responsibility" for the typos to:
* **Giraffe** (for "Gira@e") and
* **Cat** (for "o@ice", as it's associated with "In a home o@ice")

Please keep in mind that this response is light-hearted and intended for entertainment, as typos are simply errors in the provided text, not genuinely caused by animals.
```



## Related Topics

- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Quickstart (Self-Hosted)](quickstart-guide.md)
- [Notebooks](notebooks.md)
- [Multimodal PDF Data Extraction](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
