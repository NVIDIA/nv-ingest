<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->


## NVIDIA-Ingest: Multi-modal data extraction

NVIDIA-Ingest is a scalable, performance-oriented document content and metadata extraction microservice. Including support for parsing PDFs, Word and PowerPoint documents, it uses specialized NVIDIA NIM microservices to find, contextualize, and extract text, tables, charts and images for use in downstream generative applications.

NVIDIA Ingest enables parallelization of the process of splitting documents into pages where contents are classified (as tables, charts, images, text), extracted into discrete content, and further contextualized via optical character recognition (OCR) into a well defined JSON schema. From there, NVIDIA Ingest can optionally manage computation of embeddings for the extracted content, and the process storing into a vector database [Milvus](https://milvus.io/).

### What it is

A microservice that:

- Accepts a JSON Job description, containing a document payload, and a set of ingestion tasks to perform on that
  payload.
- Allows the results of a Job to be retrieved; the result is a JSON dictionary containing a list of Metadata describing
  objects extracted from the base document, as well as processing annotations and timing/trace data.
- Supports PDF, Docx, pptx, and images.
- Supports multiple methods of extraction for each document type in order to balance trade-offs between throughput and
  accuracy. For example, for PDF documents we support extraction via pdfium, Unstructured.io, and Adobe Content Extraction Services.
- Supports various types of pre and post processing operations, including text splitting and chunking; 
  transform, and filtering; embedding generation, and image offloading to storage.

### What it is not

A service that:

- Runs a static pipeline or fixed set of operations on every submitted document.
- Acts as a wrapper for any specific document parsing library.


## Quickstart

To get started using NVIDIA Ingest, you need to do a few things:
1. [Start supporting NIM microservices](#step-1-starting-containers)
2. [Install the NVIDIA Ingest client dependencies in a Python environment](#step-2-installing-python-dependencies)
3. [Submit ingestion job(s)](#step-3-ingesting-documents)
4. [Inspect and consume results](#step-4-inspecting-and-consuming-results)

### Step 1: Starting containers

This example demonstrates how to use the provided [docker-compose.yml](docker-compose.yaml) to build and start all needed services with two commands.

If preferred, you can also [start services one by one](docs/deployment.md), or run on Kubernetes via [our Helm chart](helm/README.md). Also of note are [additional environment variables](docs/environment-config.md) you may wish to configure.

First, git clone the repo:
`git clone https://github.com/nvidia/nv-ingest` and `cd nv-ingest`.

For Docker container images to be able to access to pre-built containers and NIM microservices, create a .env file and set up your API keys in it:
```
NIM_NGC_API_KEY=...
NGC_API_KEY=...
NGC_CLI_API_KEY=...
DATASET_ROOT=<PATH_TO_THIS_REPO>/data
NV_INGEST_ROOT=<PATH_TO_THIS_REPO>
```

To build Docker images locally:

`docker compose build`

Note: As configured by default in [docker-compose.yml](docker-compose.yaml), the YOLOX, DePlot, and CACHED NIM models are each pinned to a dedicated GPU. The PaddleOCR and nv-embedqa-e5 NIM models and the nv-ingest-ms-runtime share a fourth. Thus our minimum requirements are 4x NVIDIA A100 or H100 Tensor Core GPUs. 

To start all services:
`docker compose up`

Please note, NIM containers on their first startup can take 10-15 minutes to pull and fully load models. Also note that by default we have [configured log levels to be verbose](docker-compose.yaml#L31) so it's possible to observe service startup proceeding. You will notice _many_ log messages. You can control this on a per service level via each service's environment variables.

When all services have fully started, `nvidia-smi` should show processes like the following:
```
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1352957      C   tritonserver                                762MiB |
|    1   N/A  N/A   1322081      C   /opt/nim/llm/.venv/bin/python3            63916MiB |
|    2   N/A  N/A   1355175      C   tritonserver                                478MiB |
|    2   N/A  N/A   1367569      C   ...s/python/triton_python_backend_stub       12MiB |
|    3   N/A  N/A   1321841      C   python                                      414MiB |
|    3   N/A  N/A   1352331      C   tritonserver                                478MiB |
|    3   N/A  N/A   1355929      C   ...s/python/triton_python_backend_stub      424MiB |
|    3   N/A  N/A   1373202      C   tritonserver                                414MiB |
+---------------------------------------------------------------------------------------+

```
If it's taking > 1m for `nvidia-smi` to return, it's likely the bus is still busy setting up the models.

Once it completes normally (less than a few seconds), the NIM models are ready.

### Step 2: Installing Python dependencies


On the host, you'll need to create a Python environment and install dependencies:
```
conda create --name nv-ingest-dev python=3.10
conda activate nv-ingest-dev
cd client
pip install -r ./requirements.txt
pip install e .
```

### Step 3: Ingesting Documents

You can submit jobs programmatically in Python or via the nv-ingest-cli tool.

In Python (find the complete example [here](./client/client_examples/examples/python_client_usage.ipynb)):
```
# create and submit a multi modal extraction job
    job_spec = JobSpec(
        document_type=file_type,
        payload=file_content[0],
        source_id=file_name,
        source_name=file_name,
        extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
    )

    extract_task = ExtractTask(
        document_type=file_type,
        extract_text=True,
        extract_images=True,
    )

    job_spec.add_task(extract_task)
    job_id = client.add_job(job_spec)

    client.submit_job(job_id, "morpheus_task_queue")

    result = client.fetch_job_result(job_id)
    # Get back the extracted pdf data
    print(f"Got {len(result)} results")
```

Using the the `nv-ingest-cli` (find the complete example [here](./client/client_examples/examples/cli_client_usage.ipynb)):

```shell
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --client REDIS \
  --client_host=localhost \
  --client_port=6379
```

You should notice output indicating document processing status:
```
INFO:nv_ingest_client.nv_ingest_cli:Processing 1 documents.
INFO:nv_ingest_client.nv_ingest_cli:Output will be written to: ./processed_docs
Processing files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.29file/s, pages_per_sec=1.27]
INFO:nv_ingest_client.cli.util.processing:dedup_images: Avg: 0.63 ms, Median: 0.63 ms, Total Time: 0.63 ms, Total % of Trace Computation: 0.09%
INFO:nv_ingest_client.cli.util.processing:dedup_images_channel_in: Avg: 3.68 ms, Median: 3.68 ms, Total Time: 3.68 ms, Total % of Trace Computation: 0.51%
INFO:nv_ingest_client.cli.util.processing:docx_content_extractor: Avg: 0.95 ms, Median: 0.95 ms, Total Time: 0.95 ms, Total % of Trace Computation: 0.13%
INFO:nv_ingest_client.cli.util.processing:docx_content_extractor_channel_in: Avg: 1.47 ms, Median: 1.47 ms, Total Time: 1.47 ms, Total % of Trace Computation: 0.20%
INFO:nv_ingest_client.cli.util.processing:filter_images: Avg: 1.12 ms, Median: 1.12 ms, Total Time: 1.12 ms, Total % of Trace Computation: 0.15%
INFO:nv_ingest_client.cli.util.processing:filter_images_channel_in: Avg: 3.54 ms, Median: 3.54 ms, Total Time: 3.54 ms, Total % of Trace Computation: 0.49%
INFO:nv_ingest_client.cli.util.processing:job_counter: Avg: 7.66 ms, Median: 7.66 ms, Total Time: 7.66 ms, Total % of Trace Computation: 1.06%
INFO:nv_ingest_client.cli.util.processing:job_counter_channel_in: Avg: 0.26 ms, Median: 0.26 ms, Total Time: 0.26 ms, Total % of Trace Computation: 0.04%
INFO:nv_ingest_client.cli.util.processing:metadata_injection: Avg: 34.42 ms, Median: 34.42 ms, Total Time: 34.42 ms, Total % of Trace Computation: 4.74%
INFO:nv_ingest_client.cli.util.processing:metadata_injection_channel_in: Avg: 0.20 ms, Median: 0.20 ms, Total Time: 0.20 ms, Total % of Trace Computation: 0.03%
INFO:nv_ingest_client.cli.util.processing:pdf_content_extractor: Avg: 619.98 ms, Median: 619.98 ms, Total Time: 619.98 ms, Total % of Trace Computation: 85.42%
INFO:nv_ingest_client.cli.util.processing:pdf_content_extractor_channel_in: Avg: 0.76 ms, Median: 0.76 ms, Total Time: 0.76 ms, Total % of Trace Computation: 0.10%
INFO:nv_ingest_client.cli.util.processing:pptx_content_extractor: Avg: 11.57 ms, Median: 11.57 ms, Total Time: 11.57 ms, Total % of Trace Computation: 1.59%
INFO:nv_ingest_client.cli.util.processing:pptx_content_extractor_channel_in: Avg: 2.02 ms, Median: 2.02 ms, Total Time: 2.02 ms, Total % of Trace Computation: 0.28%
INFO:nv_ingest_client.cli.util.processing:redis_source_network_in: Avg: 16.11 ms, Median: 16.11 ms, Total Time: 16.11 ms, Total % of Trace Computation: 2.22%
INFO:nv_ingest_client.cli.util.processing:redis_task_sink_channel_in: Avg: 2.58 ms, Median: 2.58 ms, Total Time: 2.58 ms, Total % of Trace Computation: 0.36%
INFO:nv_ingest_client.cli.util.processing:redis_task_source: Avg: 18.81 ms, Median: 18.81 ms, Total Time: 18.81 ms, Total % of Trace Computation: 2.59%
INFO:nv_ingest_client.cli.util.processing:Unresolved time: 66.51 ms, Percent of Total Elapsed: 8.39%
INFO:nv_ingest_client.cli.util.processing:Processed 1 files in 0.79 seconds.
INFO:nv_ingest_client.cli.util.processing:Total pages processed: 1
INFO:nv_ingest_client.cli.util.processing:Throughput (Pages/sec): 1.26
INFO:nv_ingest_client.cli.util.processing:Throughput (Files/sec): 1.26
INFO:nv_ingest_client.cli.util.processing:Total timeouts: 0
```

### Step 4: Inspecting and Consuming Results

After the ingestion steps above have completed, you should be able to find `text` and `image` subfolders inside your processed docs folder. Each will contain JSON formatted extracted content and metadata.

When processing has completed, you'll have separate result files for text and image data.

Expected text extracts:
```shell
cat ./processed_docs/text/test.pdf.metadata.json
[{
  "document_type": "text",
  "metadata": {
    "content": "Here is one line of text. Here is another line of text. Here is an image.",
    "content_metadata": {
      "description": "Unstructured text from PDF document.",
      "hierarchy": {
        "block": -1,
        "line": -1,
        "page": -1,
        "page_count": 1,
        "span": -1
      },
      "page_number": -1,
      "type": "text"
    },
    "error_metadata": null,
    "image_metadata": null,
    "source_metadata": {
      "access_level": 1,
      "collection_id": "",
      "date_created": "2024-03-11T14:56:40.125063",
      "last_modified": "2024-03-11T14:56:40.125054",
      "partition_id": -1,
      "source_id": "test.pdf",
      "source_location": "",
      "source_name": "",
      "source_type": "PDF 1.4",
      "summary": ""
    },
    "text_metadata": {
      "keywords": "",
      "language": "en",
      "summary": "",
      "text_type": "document"
    }
  }
]]```

Expected image extracts:
```shell
$ cat ./processed_docs/image/test.pdf.metadata.json
[{
  "document_type": "image",
  "metadata": {
    "content": "<--- Base64 encoded image data --->",
    "content_metadata": {
      "description": "Image extracted from PDF document.",
      "hierarchy": {
        "block": 3,
        "line": -1,
        "page": 0,
        "page_count": 1,
        "span": -1
      },
      "page_number": 0,
      "type": "image"
    },
    "error_metadata": null,
    "image_metadata": {
      "caption": "",
      "image_location": [
        73.5,
        160.7775878906,
        541.5,
        472.7775878906
      ],
      "image_type": "png",
      "structured_image_type": "image_type_1",
      "text": ""
    },
    "source_metadata": {
      "access_level": 1,
      "collection_id": "",
      "date_created": "2024-03-11T14:56:40.125063",
      "last_modified": "2024-03-11T14:56:40.125054",
      "partition_id": -1,
      "source_id": "test.pdf",
      "source_location": "",
      "source_name": "",
      "source_type": "PDF 1.4",
      "summary": ""
    },
    "text_metadata": null
  }
}]
```

We also provide a script for inspecting [extracted images](#image_viewerpy)
```shell
python src/util/image_viewer.py --file_path ./processed_docs/image/test.pdf.metadata.json
```

Beyond inspecting the results, you can read them into something like a llama-index or langchain document query pipeline:

Please also checkout our [demo using a retrieval pipeline on build.nvidia.com](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag) to query over document content pre-extracted w/ NVIDIA Ingest.

## Third Party License Notice:

If configured to do so, this project will download and install additional third-party open source software projects.
Review the license terms of these open source projects before use:

https://pypi.org/project/pdfservices-sdk/

- **`INSTALL_ADOBE_SDK`**:
  - **Description**: If set to `true`, the Adobe SDK will be installed in the container at launch time. This is
    required if you want to use the Adobe extraction service for PDF decomposition. Please review the "
    "[license agreement](https://github.com/adobe/pdfservices-python-sdk?tab=License-1-ov-file) for the
    pdfservices-sdk before enabling this option."


## Contributing

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

### Full text of the DCO:

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
