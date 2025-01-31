<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->


## NVIDIA-Ingest: Multi-modal data extraction

NVIDIA-Ingest is a scalable, performance-oriented document content and metadata extraction microservice. NV-Ingest includes support for parsing PDFs, Word, and PowerPoint documents. NV-Ingest uses specialized NVIDIA NIMs (self-hosted microservices, or hosted on build.nvidia.com) to find, contextualize, and extract text, tables, charts, and images that you can use in downstream generative applications.

NVIDIA Ingest enables parallelization of the process of splitting documents into pages where contents are classified (as tables, charts, images, text), extracted into discrete content, and further contextualized via optical character recognition (OCR) into a well defined JSON schema. From there, NVIDIA Ingest can optionally manage computation of embeddings for the extracted content, and also optionally manage storing into a vector database [Milvus](https://milvus.io/).

### Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quickstart](#quickstart)
4. [Repo Structure](#repo-structure)
5. [Notices](#notices)

## Introduction

### What NVIDIA-Ingest Is ✔️

A library and microservice that:

- Accepts a JSON Job description, containing a document payload, and a set of ingestion tasks to perform on that payload.
- Allows the results of a Job to be retrieved; the result is a JSON dictionary containing a list of Metadata describing objects extracted from the base document, as well as processing annotations and timing/trace data.
- Supports PDF, Docx, pptx, and images.
- Supports multiple methods of extraction for each document type in order to balance trade-offs between throughput and accuracy. For example, for PDF documents we support extraction via pdfium, Unstructured.io, and Adobe Content Extraction Services.
- Supports various types of pre and post processing operations, including text splitting and chunking; transform, and filtering; embedding generation, and image offloading to storage.

### What NVIDIA-Ingest Is Not ✖️

A service that:

- Runs a static pipeline or fixed set of operations on every submitted document.
- Acts as a wrapper for any specific document parsing library.

For production level performance and scalability, we recommend deploying the pipeline and supporting NIMs via docker-compose or kubernetes (via the provided helm charts).

For hardware and software pre-requisites for container and kubernetes (helm) based deployments, please find [our comprehensive doc site](https://docs.nvidia.com/nv-ingest/user-guide/getting-started/prerequisites/).


## Library Mode Quickstart

To facilitate an easier evaluation experience, and for small scale (<100 PDFs) workloads, you can use our "library mode" setup, which depends on NIMs either already self hosted, or, by default, NIMs hosted on build.nvidia.com.

To get started using NVIDIA Ingest in library mode, you need to do a few things:
1. Have a [cuDF compatible GPU](https://github.com/rapidsai/cudf?tab=readme-ov-file#cudagpu-requirements)
2. Create a conda environment and install nv-ingest
```
conda create -y --name nvingest python=3.10
conda activate nvingest

conda install -c nvidia -c rapidsai -c rapidsai-nightly nvidia/label/dev::nv_ingest nvidia/label/dev::nv_ingest_client
# conda does not provide milvus packages
pip install opencv-python llama-index-embeddings-nvidia pymilvus 'pymilvus[bulk_writer, model]' milvus-lite
```
3. [Obtain an API key from build.nvidia.com](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2?snippet_tab=Python&signin=true&api_key=true)
4. Add your key to .env
```.env
NVIDIA_BUILD_API_KEY=...
```

Note: To use library mode with self hosted NIMs: TODO

5. Start ingesting and retrieving documents:
```python
import os, sys, json, time, logging
from dotenv import load_dotenv
load_dotenv(".env")

from nv_ingest.util.pipeline.pipeline_runners import start_pipeline_subprocess
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient
from nv_ingest.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_client.util.milvus import nvingest_retrieval
from nv_ingest.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

from openai import OpenAI

# Start the pipeline subprocess for library mode
config = PipelineCreationSchema()
pipeline_process = start_pipeline_subprocess(config)

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
)

model_name = "nvidia/llama-3.2-nv-embedqa-1b-v2"
# Note: gpu_cagra accelerated indexing is not yet available in milvus-lite
# Providing a filename for milvus_uri will use milvus-lite
milvus_uri = "milvus.db"
collection_name = "test"

# do content extraction from files
ingestor = (
    Ingestor(client=client)
    .files("./data/multimodal_test.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=True,
        text_depth="page"
    ).caption(
        model_name="/meta/llama-3.2-11b-vision-instruct",
    ).embed(
    ).vdb_upload(
        collection_name=collection_name,
        milvus_uri=milvus_uri,
        sparse=True
    )
)

# results blob is directly inspectable
results = ingestor.ingest()

# to inspect text renderings of individual documents:
print(ingest_json_results_to_blob(results[0]))

# hybrid search interface (model name exposed to config between e5-v5 and new llama-embedder)
queries = ["Where and what is the dog doing?"]
embedding_nim_endpoint='https://integrate.api.nvidia.com/v1'
embedding_nim_model_name='nvidia/llama-3.2-nv-embedqa-1b-v2'

retrieved_docs = nvingest_retrieval(
    queries,
    collection_name,
    milvus_uri=milvus_uri,
    hybrid=True,
    embedding_endpoint=f"{embedding_nim_endpoint}",
    top_k=1,
    model_name=embedding_nim_model_name
)

# simple generation example
extract = retrieved_docs[0][0]["entity"]["text"]
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.environ["NVIDIA_BUILD_API_KEY"]
)

prompt = f"Using the following content: {extract}: Answer the user query: {query}"
completion = client.chat.completions.create(
  model="nvdev/meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content": prompt}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

print(completion.choices[0].message.content)
```
The above ingestion, retrieval, and generation pipeline should print:
```
Using the following content: | locations, |
| Animal | Activity | Place |
| Giraffe | Driving a car. | At the beach |
| Lion | Putting on sunscreen | At the park |
| Cat | Jumping onto a laptop | In a home office |
| Dog | Chasing a squirrel | In the front yard |
: Answer the user query: Where and what is the dog doing?

The dog is chasing a squirrel in the front yard.
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
