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


## Prerequisites

### Hardware

| GPU | Family | Memory | # of GPUs |
| ------ | ------ | ------ | ------ |
| H100 | SXM/NVLink or PCIe | 80GB | 2 |
| A100 | SXM/NVLink or PCIe | 80GB | 2 |

### Software

- Linux operating systems (Ubuntu 20.04 or later recommended)
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA Driver >= 535)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


## Quickstart

To get started using NVIDIA Ingest, you need to do a few things:
1. [Start supporting NIM microservices](#step-1-starting-containers)
2. [Install the NVIDIA Ingest client dependencies in a Python environment](#step-2-installing-python-dependencies)
3. [Submit ingestion job(s)](#step-3-ingesting-documents)
4. [Inspect and consume results](#step-4-inspecting-and-consuming-results)

### Step 1: Starting containers

This example demonstrates how to use the provided [docker-compose.yaml](docker-compose.yaml) to start all needed services with a few commands.

If preferred, you can also [start services one by one](docs/deployment.md), or run on Kubernetes via [our Helm chart](helm/README.md). Also of note are [additional environment variables](docs/environment-config.md) you may wish to configure.

First, git clone the repo:
`git clone https://github.com/nvidia/nv-ingest` and `cd nv-ingest`.

To access pre-built containers and NIM microservices, [generate API keys](docs/ngc-api-key.md) and authenticate with NGC with the `docker login` command:
```shell
$ docker login nvcr.io
Username: $oauthtoken
Password: <Your Key>
```

For Docker container images to be able to access NGC resources, create a .env file, and set up your API key in it:
```
NGC_API_KEY=...
DATASET_ROOT=<PATH_TO_THIS_REPO>/data
NV_INGEST_ROOT=<PATH_TO_THIS_REPO>
```

Note: As configured by default in [docker-compose.yaml](docker-compose.yaml), the DePlot NIM is on a dedicated GPU. All other NIMs and the nv-ingest container itself share a second.

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
pip install .
```

### Step 3: Ingesting Documents

You can submit jobs programmatically in Python or via the nv-ingest-cli tool.

+In the below examples, we are doing text, chart, table, and image extraction:
+`extract_text`, - uses PDFium to find and extract text from pages
+`extract_images` - uses PDFium to extract images
+`extract_tables` - uses YOLOX to find tables and charts. Uses PaddleOCR for table extraction, and Deplot, CACHED, and PaddleOCR for chart extraction

Note that `extract_tables` controls extraction for both tables and charts.

In Python (you can find more documentation and examples [here](./client/client_examples/examples/python_client_usage.ipynb)):
```
import logging, time
import concurrent.futures

from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.util.file_processing.extract import extract_file_content

logger = logging.getLogger("nv_ingest_client")

file_name = "data/multimodal_test.pdf"
file_content, file_type = extract_file_content(file_name)

# A JobSpec is an object that defines a document and how it should
# be processed by the nv-ingest service.
job_spec = JobSpec(
  document_type=file_type,
  payload=file_content,
  source_id=file_name,
  source_name=file_name,
  extended_options=
    {
      "tracing_options":
      {
        "trace": True,
        "ts_send": time.time_ns()
      }
    }
)

# configure desired extraction modes here. Multiple extraction
# methods can be defined for a single JobSpec
extract_task = ExtractTask(
  document_type=file_type,
  extract_text=True,
  extract_images=True,
  extract_tables=True
)

job_spec.add_task(extract_task)

# Create the client and inform it about the JobSpec we want to process.
client = NvIngestClient(
  message_client_hostname="localhost", # Host where nv-ingest-ms-runtime is running
  message_client_port=7670 # REST port, defaults to 7670
)
job_id = client.add_job(job_spec)
client.submit_job(job_id, "morpheus_task_queue")


# Nv-Ingest jobs are often "long running". Therefore after
# submission we intermittently check if the job is completed.
def fetch_wait_completed_results(job_id):
  while True:
    try:
      result = client.fetch_job_result(job_id, timeout=60)
      return result
    except TimeoutError:
      print("Job still processing ... aka HTTP 202 received")

# Results are fetched in an async manner. If the job is still running a
# HTTP 202 response is returned and interpreted as a TimeoutError.
# We continue retrying here until our timeout is reached
timeout_seconds = 60
with concurrent.futures.ThreadPoolExecutor() as executor:
  future = executor.submit(fetch_wait_completed_results, job_id)
  
  try:
      # Wait for the result within the specified timeout
      result = future.result(timeout=timeout_seconds)
      print(f"Got {len(result)} results")
  except concurrent.futures.TimeoutError:
      print(f"Job processing did not complete within the specified {timeout_seconds} seconds")
```

Using the the `nv-ingest-cli` (find the complete example [here](./client/client_examples/examples/cli_client_usage.ipynb)):

```shell
nv-ingest-cli \
  --doc ./data/multimodal_test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_tables": "true", "extract_images": "true"}' \
  --client_host=localhost \
  --client_port=7670
```

You should notice output indicating document processing status:
```
INFO:nv_ingest_client.nv_ingest_cli:Processing 1 documents.
INFO:nv_ingest_client.nv_ingest_cli:Output will be written to: ./processed_docs
Processing files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:10<00:00, 10.47s/file, pages_per_sec=0.29]
INFO:nv_ingest_client.cli.util.processing:dedup_images: Avg: 1.02 ms, Median: 1.02 ms, Total Time: 1.02 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:dedup_images_channel_in: Avg: 1.44 ms, Median: 1.44 ms, Total Time: 1.44 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:docx_content_extractor: Avg: 0.66 ms, Median: 0.66 ms, Total Time: 0.66 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:docx_content_extractor_channel_in: Avg: 1.09 ms, Median: 1.09 ms, Total Time: 1.09 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:filter_images: Avg: 0.84 ms, Median: 0.84 ms, Total Time: 0.84 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:filter_images_channel_in: Avg: 7.75 ms, Median: 7.75 ms, Total Time: 7.75 ms, Total % of Trace Computation: 0.07%
INFO:nv_ingest_client.cli.util.processing:job_counter: Avg: 2.13 ms, Median: 2.13 ms, Total Time: 2.13 ms, Total % of Trace Computation: 0.02%
INFO:nv_ingest_client.cli.util.processing:job_counter_channel_in: Avg: 2.05 ms, Median: 2.05 ms, Total Time: 2.05 ms, Total % of Trace Computation: 0.02%
INFO:nv_ingest_client.cli.util.processing:metadata_injection: Avg: 14.48 ms, Median: 14.48 ms, Total Time: 14.48 ms, Total % of Trace Computation: 0.14%
INFO:nv_ingest_client.cli.util.processing:metadata_injection_channel_in: Avg: 0.22 ms, Median: 0.22 ms, Total Time: 0.22 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:pdf_content_extractor: Avg: 10332.97 ms, Median: 10332.97 ms, Total Time: 10332.97 ms, Total % of Trace Computation: 99.45%
INFO:nv_ingest_client.cli.util.processing:pdf_content_extractor_channel_in: Avg: 0.44 ms, Median: 0.44 ms, Total Time: 0.44 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:pptx_content_extractor: Avg: 1.19 ms, Median: 1.19 ms, Total Time: 1.19 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:pptx_content_extractor_channel_in: Avg: 0.98 ms, Median: 0.98 ms, Total Time: 0.98 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:redis_source_network_in: Avg: 12.27 ms, Median: 12.27 ms, Total Time: 12.27 ms, Total % of Trace Computation: 0.12%
INFO:nv_ingest_client.cli.util.processing:redis_task_sink_channel_in: Avg: 2.16 ms, Median: 2.16 ms, Total Time: 2.16 ms, Total % of Trace Computation: 0.02%
INFO:nv_ingest_client.cli.util.processing:redis_task_source: Avg: 8.00 ms, Median: 8.00 ms, Total Time: 8.00 ms, Total % of Trace Computation: 0.08%
INFO:nv_ingest_client.cli.util.processing:Unresolved time: 82.82 ms, Percent of Total Elapsed: 0.79%
INFO:nv_ingest_client.cli.util.processing:Processed 1 files in 10.47 seconds.
INFO:nv_ingest_client.cli.util.processing:Total pages processed: 3
INFO:nv_ingest_client.cli.util.processing:Throughput (Pages/sec): 0.29
INFO:nv_ingest_client.cli.util.processing:Throughput (Files/sec): 0.10
```

### Step 4: Inspecting and Consuming Results

After the ingestion steps above have completed, you should be able to find `text` and `image` subfolders inside your processed docs folder. Each will contain JSON formatted extracted content and metadata.

When processing has completed, you'll have separate result files for text and image data.
```shell
ls -R processed_docs/
processed_docs/:
image  structured  text

processed_docs/image:
multimodal_test.pdf.metadata.json

processed_docs/structured:
multimodal_test.pdf.metadata.json

processed_docs/text:
multimodal_test.pdf.metadata.json
```
You can view the full JSON extracts and the metadata definitions [here](https://github.com/NVIDIA/nv-ingest/blob/main/docs/content-metadata.md).

We also provide a script for inspecting [extracted images](#image_viewerpy)
```shell
pip install tkinter
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
