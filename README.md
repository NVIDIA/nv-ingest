<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## NVIDIA-Ingest: Multi-modal data extraction

NVIDIA-Ingest is a scalable, performance-oriented content and metadata extraction SDK for a variety of input formats. NV-Ingest includes support for parsing PDFs, text files, Microsoft Word and PowerPoint documents, plain images, and audio files. NV-Ingest uses specialized NVIDIA NIMs (self-hosted microservices, or hosted on build.nvidia.com) to find, contextualize, and extract text, tables, charts, and unstructured images that you can use in downstream generative applications.

> [!Note]
> NVIDIA Ingest is also known as NV-Ingest and [NeMo Retriever extraction](https://docs.nvidia.com/nemo/retriever/extraction/overview/).

NVIDIA Ingest enables parallelization of the process of splitting documents into pages where contents are classified (as tables, charts, images, text), extracted into discrete content, and further contextualized via optical character recognition (OCR) into a well defined JSON schema. From there, NVIDIA Ingest can optionally manage computation of embeddings for the extracted content, and also optionally manage storing into a vector database [Milvus](https://milvus.io/).

![Pipeline Overview](https://docs.nvidia.com/nemo/retriever/extraction/images/overview-extraction.png)

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

Create a fresh conda environment in which to install nv-ingest and dependencies.

```shell
conda create -y --name nvingest python=3.10 && \
    conda activate nvingest && \
    conda install -y -c rapidsai -c conda-forge -c nvidia nv_ingest=25.3.0 nv_ingest_client=25.3.0 nv_ingest_api=25.3.0 && \
    pip install opencv-python llama-index-embeddings-nvidia pymilvus 'pymilvus[bulk_writer, model]' milvus-lite nvidia-riva-client unstructured-client
```

Make sure to set your NVIDIA_BUILD_API_KEY and NVIDIA_API_KEY. If you don't have one, you can get one on [build.nvidia.com](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2?snippet_tab=Python&signin=true&api_key=true).
```
#Note: these should be the same value
export NVIDIA_BUILD_API_KEY=nvapi-...
export NVIDIA_API_KEY=nvapi-...
```

### Step 2: Ingest Documents

You can submit jobs programmatically in Python.

Note: Make sure your conda environment is activated. `which python` should indicate that you're using the conda provided python installation (not an OS provided python).
```
which python
/home/dev/miniforge3/envs/nvingest/bin/python
```

If you have a very high number of CPUs and see the process hang without progress, we recommend using taskset to limit the number of CPUs visible to the process:
```
taskset -c 0-3 python your_ingestion_script.py
```

On a 4 CPU core low end laptop, the following should take about 10 seconds:
```python
import logging, os, time, sys
               
from nv_ingest.util.pipeline.pipeline_runners import start_pipeline_subprocess
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient
from nv_ingest.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# Start the pipeline subprocess for library mode                       
config = PipelineCreationSchema()                                                  

pipeline_process = start_pipeline_subprocess(config)
# you can configure the subprocesses to log stderr to stdout for debugging purposes
#pipeline_process = start_pipeline_subprocess(config, stderr=sys.stderr, stdout=sys.stdout)
                                                                          
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
        # Slower, but maximally accurate, especially for PDFs with pages that are scanned images
        #extract_method="nemoretriever_parse",
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
print(f"Time taken: {t1-t0} seconds")

# results blob is directly inspectable
print(ingest_json_results_to_blob(results[0]))
```

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

### Step 3: Query Ingested Content

Below is an example snippet demonstrating how to query for relevant snippets of the ingested content and insert them into a basic prompt for use with an LLM to generate answers.

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

For more information, please check out the [official documentation](https://docs.nvidia.com/nemo/retriever/extraction/overview/).

> [!TIP]
> Beyond inspecting the results, you can read them into things like [llama-index](examples/llama_index_multimodal_rag.ipynb) or [langchain](examples/langchain_multimodal_rag.ipynb) retrieval pipelines.
>
> Please also checkout our [demo using a retrieval pipeline on build.nvidia.com](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag) to query over document content pre-extracted w/ NVIDIA Ingest.

## Repo Structure

Beyond the relevant documentation, examples, and other links above, below is a description of the contents in this repo's folders:

- [.github](https://github.com/NVIDIA/nv-ingest/tree/main/.github): GitHub repo configuration files
- [api](https://github.com/NVIDIA/nv-ingest/tree/main/api): Core API python logic shared across python modules
- [ci](https://github.com/NVIDIA/nv-ingest/tree/main/ci): Scripts used to build the NV-Ingest container and other packages
- [client](https://github.com/NVIDIA/nv-ingest/tree/main/client): Docs and source code for the nv-ingest-cli utility
- [conda](https://github.com/NVIDIA/nv-ingest/tree/main/conda): Conda environment and packaging definitions
- [config](https://github.com/NVIDIA/nv-ingest/tree/main/config): Various .yaml files defining configuration for OTEL, Prometheus
- [data](https://github.com/NVIDIA/nv-ingest/tree/main/data): Sample PDFs provided for testing convenience
- [deploy](https://github.com/NVIDIA/nv-ingest/tree/main/deploy): Brev.dev hosted launchable
- [docker](https://github.com/NVIDIA/nv-ingest/tree/main/docker): Houses scripts used by the nv-ingest docker container
- [docs](https://github.com/NVIDIA/nv-ingest/tree/main/docs/docs): Various READMEs describing deployment, metadata schemas, auth and telemetry setup
- [evaluation](https://github.com/NVIDIA/nv-ingest/tree/main/evaluation): Contains notebooks demonstrating how to test recall accuracy
- [examples](https://github.com/NVIDIA/nv-ingest/tree/main/examples): Example notebooks, scripts, and longer-form tutorial content
- [helm](https://github.com/NVIDIA/nv-ingest/tree/main/helm): Documentation for deploying NV-Ingest to a Kubernetes cluster via Helm chart
- [skaffold](https://github.com/NVIDIA/nv-ingest/tree/main/skaffold): Skaffold configuration
- [src](https://github.com/NVIDIA/nv-ingest/tree/main/src): Source code for the NV-Ingest pipelines and service
- [.devcontainer](https://github.com/NVIDIA/nv-ingest/tree/main/.devcontainer): VSCode containers for local development
- [tests](https://github.com/NVIDIA/nv-ingest/tree/main/tests): Unit tests for NV-Ingest

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
