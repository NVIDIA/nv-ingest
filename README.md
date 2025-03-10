<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## NVIDIA-Ingest: Multi-modal data extraction

NVIDIA-Ingest is a scalable, performance-oriented content and metadata extraction SDK for a variety of input formats. NV-Ingest includes support for parsing PDFs, text files, Microsoft Word and PowerPoint documents, plain images, and audio files. NV-Ingest uses specialized NVIDIA NIMs (self-hosted microservices, or hosted on build.nvidia.com) to find, contextualize, and extract text, tables, charts, and unstructured images that you can use in downstream generative applications.

> [!Note]
> NVIDIA Ingest is also known as NV-Ingest and NeMo Retriever Extraction.

NVIDIA Ingest enables parallelization of the process of splitting documents into pages where contents are classified (as tables, charts, images, text), extracted into discrete content, and further contextualized via optical character recognition (OCR) into a well defined JSON schema. From there, NVIDIA Ingest can optionally manage computation of embeddings for the extracted content, and also optionally manage storing into a vector database [Milvus](https://milvus.io/).

### Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quickstart](#quickstart)
4. [Repo Structure](#repo-structure)
5. [Notices](#notices)

## Introduction

## What NVIDIA-Ingest Is ✔️

NV-Ingest is a library and microservice service that does the following:

- Accept a job specification that contains a document payload and a set of ingestion tasks to perform on that payload.
- Store the result of each job to retrieve later. The result is a dictionary that contains a list of metadata that describes the objects extracted from the base document, and processing annotations and timing/trace data.
- Support multiple methods of extraction for each document type to balance trade-offs between throughput and accuracy. For example, for .pdf documents nv-ingest supports extraction through pdfium, [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse), Unstructured.io, and Adobe Content Extraction Services.
- Support various types of before and after processing operations, including text splitting and chunking, transform and filtering, embedding generation, and image offloading to storage.

NV-Ingest supports the following file types:

- `pdf`
- `docx`
- `pptx`
- `jpeg`
- `png`
- `svg`
- `tiff`
- `txt`

## Prerequisites

For production-level performance and scalability, we recommend that you deploy the pipeline and supporting NIMs by using Docker Compose or Kubernetes ([helm charts](helm)). For more information, refer to [prerequisites](https://docs.nvidia.com/nv-ingest/user-guide/getting-started/prerequisites).

## Library Mode Quickstart

For small-scale workloads, such as workloads of fewer than 100 PDFs, you can use library mode setup. Library mode set up depends on NIMs that are already self-hosted, or, by default, NIMs that are hosted on build.nvidia.com.

Library mode deployment of nv-ingest requires:

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [Conda Python environment and package manager](https://github.com/conda-forge/miniforge)
- Python 3.10

### Step 1: Prepare Your Environment
```
conda create -y --name nvingest python=3.10
conda activate nvingest
conda install -y -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia nvidia/label/dev::nv_ingest nvidia/label/dev::nv_ingest_client nvidia/label/dev::nv_ingest_api
pip install opencv-python llama-index-embeddings-nvidia pymilvus 'pymilvus[bulk_writer, model]' milvus-lite dotenv ffmpeg nvidia-riva-client

# Temporary workaround: remove after conda package fixes in place
git clone https://github.com/nvidia/nv-ingest
cd nv-ingest
pip install -e ./api
pip install -e ./client
```

2. Create a .env file that contains your NVIDIA Build API key. For more information, refer to [Environment Configuration Variables](docs/docs/extraction/environment-config.md).

```
NVIDIA_BUILD_API_KEY=nvapi-...
NVIDIA_API_KEY=nvapi-...

# For nvidians using nvdev endpoints:
# After release, public build endpoints are builtin to library setup
PADDLE_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/nvdev/cv/baidu/paddleocr
PADDLE_INFER_PROTOCOL=http

YOLOX_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvdev/nvidia/nemoretriever-page-elements-v2
YOLOX_INFER_PROTOCOL=http

YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvdev/nvidia/nemoretriever-graphic-elements-v1
YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL=http

YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvdev/nvidia/nemoretriever-table-structure-v1
YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL=http

EMBEDDING_NIM_ENDPOINT=https://integrate.api.nvidia.com/v1
EMBEDDING_NIM_MODEL_NAME=nvdev/nvidia/llama-3.2-nv-embedqa-1b-v2

VLM_CAPTION_ENDPOINT=https://ai.api.nvidia.com/v1/gr/nvdev/meta/llama-3.2-11b-vision-instruct/chat/completions
VLM_CAPTION_MODEL_NAME=meta/llama-3.2-11b-vision-instruct

# NemoRetriever Parse:
NEMORETRIEVER_PARSE_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvdev/nvidia/nemoretriever-parse
NEMORETRIEVER_PARSE_INFER_PROTOCOL=http
```

### Step 2: Ingest Documents

You can submit jobs programmatically in Python.

```python
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
# Providing a filename for milvus_uri will use milvus-lite
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
        text_depth="page"
    ).embed()
    .caption(           
        model_name="nvdev/meta/llama-3.2-11b-vision-instruct",
        endpoint_url="https://ai.api.nvidia.com/v1/gr/nvdev/meta/llama-3.2-11b-vision-instruct/chat/completions"
    )
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

You can see the extracted text representing the content of the ingested test document:
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

### Step 3: Query Over Ingested Content

Next you can query for relevant snippets of the ingested content and use it with LLMs to generate answers:

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
    #embedding_endpoint=os.environ["EMBEDDING_NIM_ENDPOINT"],
    #model_name=os.environ["EMBEDDING_NIM_MODEL_NAME"]
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
  #model="nvdev/meta/llama-3.3-70b-instruct",
  #model="meta/llama-3.3-70b-instruct",
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

> [!TIP]
> Beyond inspecting the results, you can read them into things like [llama-index](examples/llama_index_multimodal_rag.ipynb) or [langchain](examples/langchain_multimodal_rag.ipynb) retrieval pipelines.
>
> Please also checkout our [demo using a retrieval pipeline on build.nvidia.com](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag) to query over document content pre-extracted w/ NVIDIA Ingest.

## Repo Structure

Beyond the relevant documentation, examples, and other links above, below is a description of contents in this repo's folders:

1. [.github](.github): GitHub repo configuration files
2. [ci](ci): scripts used to build the nv-ingest container and other packages
3. [client](client): docs and source code for the nv-ingest-cli utility
4. [config](config): various yaml files defining configuration for OTEL, Prometheus
5. [data](data): Sample PDFs provided for testing convenience
6. [docker](docker): houses scripts used by the nv-ingest docker container
7. [docs](docs): Various READMEs describing deployment, metadata schemas, auth and telemetry setup
8. [examples](examples): Example notebooks, scripts, and longer form tutorial content
9. [helm](helm): Documentation for deploying nv-ingest to a Kubernetes cluster via Helm chart
10. [skaffold](skaffold): Skaffold configuration
11. [src](src): source code for the nv-ingest pipelines and service
12. [tests](tests): unit tests for nv-ingest

## Notices

### Third Party License Notice:

If configured to do so, this project will download and install additional third-party open source software projects.
Review the license terms of these open source projects before use:

https://pypi.org/project/pdfservices-sdk/

- **`INSTALL_ADOBE_SDK`**:
  - **Description**: If set to `true`, the Adobe SDK will be installed in the container at launch time. This is
    required if you want to use the Adobe extraction service for PDF decomposition. Please review the
    [license agreement](https://github.com/adobe/pdfservices-python-sdk?tab=License-1-ov-file) for the
    pdfservices-sdk before enabling this option.
- **`DOWNLOAD_LLAMA_TOKENIZER` (Built With Llama):**:
  - **Description**: The Split task uses the `meta-llama/Llama-3.2-1B` tokenizer, which will be downloaded
    from HuggingFace at build time if `DOWNLOAD_LLAMA_TOKENIZER` is set to `True`. Please review the
    [license agreement](https://huggingface.co/meta-llama/Llama-3.2-1B) for Llama 3.2 materials before using this.
    This is a gated model so you'll need to [request access](https://huggingface.co/meta-llama/Llama-3.2-1B) and
    set `HF_ACCESS_TOKEN` to your HuggingFace access token in order to use it.


### Contributing

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original
work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the --signoff (or -s) option when committing your changes:

```
$ git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

#### Full text of the DCO:

```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129

  Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
```

```
  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

  (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
