<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## NVIDIA-Ingest: Multi-modal data extraction

NVIDIA-Ingest is a scalable, performance-oriented document content and metadata extraction microservice. Including support for parsing PDFs, Word and PowerPoint documents, it uses specialized NVIDIA NIM microservices to find, contextualize, and extract text, tables, charts and images for use in downstream generative applications.

> [!Note]
> NVIDIA Ingest is also known as NV-Ingest and NeMo Retriever Extraction.

NVIDIA Ingest enables parallelization of splitting documents into pages where artifacts are classified (such as text, tables, charts, and images), extracted, and further contextualized through optical character recognition (OCR) into a well defined JSON schema. From there, NVIDIA Ingest can optionally manage computation of embeddings for the extracted content, and also optionally manage storing into a vector database [Milvus](https://milvus.io/).

> [!Note]
> Cached and Deplot are deprecated.
> Instead, docker-compose now uses a beta version of the yolox-graphic-elements container.
> With this change, you should now be able to run nv-ingest on a single 80GB A100 or H100 GPU.
> If you want to use the old pipeline, with Cached and Deplot, use the [nv-ingest 24.12.1 release](https://github.com/NVIDIA/nv-ingest/tree/24.12.1).


### Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quickstart](#quickstart)
4. [Repo Structure](#repo-structure)
5. [Notices](#notices)

## Introduction

## What NVIDIA-Ingest Is ✔️

NV-Ingest is a microservice service that does the following:

- Accept a JSON job description, containing a document payload, and a set of ingestion tasks to perform on that payload.
- Allow the results of a job to be retrieved. The result is a JSON dictionary that contains a list of metadata describing objects extracted from the base document, and processing annotations and timing/trace data.
- Support multiple methods of extraction for each document type to balance trade-offs between throughput and accuracy. For example, for .pdf documents, we support extraction through pdfium, Unstructured.io, and Adobe Content Extraction Services.
- Support various types of pre- and post- processing operations, including text splitting and chunking, transform and filtering, embedding generation, and image offloading to storage.

NV-Ingest supports the following file types:

- `pdf`
- `docx`
- `pptx`
- `jpeg`
- `png`
- `svg`
- `tiff`
- `txt`


## What NVIDIA-Ingest Isn't ✖️

NV-Ingest does not do the following:

- Run a static pipeline or fixed set of operations on every submitted document.
- Act as a wrapper for any specific document parsing library.


## Prerequisites

### Hardware

| GPU | Family | Memory | # of GPUs (min.) |
| ------ | ------ | ------ | ------ |
| H100 | SXM or PCIe | 80GB | 1 |
| A100 | SXM or PCIe | 80GB | 1 |

### Software

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA Driver >= `535`, CUDA >= `12.2`)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

> [!Note]
> You install Python in a later step. NVIDIA-Ingest only supports [Python version 3.10](https://www.python.org/downloads/release/python-3100/).

## Quickstart

To get started using NVIDIA Ingest, you need to do a few things:
1. [Start supporting NIM microservices](#step-1-starting-containers) 🏗️
2. [Install the NVIDIA Ingest client dependencies in a Python environment](#step-2-installing-python-dependencies) 🐍
3. [Submit ingestion job(s)](#step-3-ingesting-documents) 📓
4. [Inspect and consume results](#step-4-inspecting-and-consuming-results) 🔍

Optional:
1. [Direct Library Deployment](docs/docs/extraction/quickstart-library-mode.md) 📦

### Step 1: Starting containers

This example demonstrates how to use the provided [docker-compose.yaml](docker-compose.yaml) to start all needed services with a few commands.

> [!IMPORTANT]
> NIM containers on their first startup can take 10-15 minutes to pull and fully load models.

If you prefer, you can also [start services one by one](docs/docs/extraction/quickstart-guide.md), or run on Kubernetes via [our Helm chart](helm/README.md). Also of note are [additional environment variables](docs/docs/extraction/environment-config.md) you may wish to configure.

1. Git clone the repo:
`git clone https://github.com/nvidia/nv-ingest`
2. Change directory to the cloned repo
`cd nv-ingest`.

3. [Generate API keys](docs/docs/extraction/ngc-api-key.md) and authenticate with NGC with the `docker login` command:
```shell
# This is required to access pre-built containers and NIM microservices
$ docker login nvcr.io
Username: $oauthtoken
Password: <Your Key>
```

> [!NOTE]
> During the early access (EA) phase, you must apply for early access here: https://developer.nvidia.com/nemo-microservices-early-access/join.
> When your early access is approved, follow the instructions in the email to create an organization and team, link your profile, and generate your NGC API key.

4. Create an [environment variable](environment-config.md) file that contains your NGC and NVIDA keys.

> [!NOTE]
> If you use an NGC personal key, then you should provide the same value for all keys, but you must specify each environment variable individually.
> In the past, you could create an API key. If you have an API key, you can still use that. For more information, refer to [Generate Your NGC Keys](ngc-api-key.md).

```
# Container images must access resources from NGC.

NGC_API_KEY=<key to download containers from NGC>
NIM_NGC_API_KEY=<key to download model files after containers start>
NVIDIA_BUILD_API_KEY=<key to use NIMs that are hosted on build.nvidia.com>
NVIDIA_API_KEY=<copy of NVIDIA_BUILD_API_KEY, llama-index connectors use this key>
```

> [!NOTE]
> As configured by default in [docker-compose.yaml](docker-compose.yaml#L52), the DePlot NIM is on a dedicated GPU. All other NIMs and the nv-ingest container itself share a second. This is to avoid DePlot and other NIMs competing for VRAM on the same device.
>
> Change the `CUDA_VISIBLE_DEVICES` pinnings as desired for your system within docker-compose.yaml.

> [!IMPORTANT]
> Make sure NVIDIA is set as your default container runtime before running the docker compose command with the command:
> `sudo nvidia-ctk runtime configure --runtime=docker --set-as-default`

> [!NOTE]
> The most accurate tokenizer based splitting depends on the [llama-3.2 tokenizer](https://huggingface.co/meta-llama/Llama-3.2-1B). To download this model at container build time, you must set `DOWNLOAD_LLAMA_TOKENIZER=True` _and_ supply an authorized HuggingFace access token via `HF_ACCESS_TOKEN=<your access token>`. If not, the ungated [e5-large-unsupervised](https://huggingface.co/intfloat/e5-large-unsupervised) tokenizer model will be downloaded instead. By default, the split task will use whichever model has been predownloaded. Refer to [Environment Configuration Variables](docs/docs/extraction/environment-config.md) for more info.

5. Start all services:
`docker compose --profile retrieval up`

> [!TIP]
> By default we have [configured log levels to be verbose](docker-compose.yaml).
>
> It's possible to observe service startup proceeding: you will notice many log messages. Disable verbose logging by configuring `NIM_TRITON_LOG_VERBOSE=0` for each NIM in [docker-compose.yaml](docker-compose.yaml).
>
> If you want to build from source, use `docker compose up --build` instead. This will build from your repo's code rather than from an already published container.

6. When all services have fully started, `nvidia-smi` should show processes like the following:
```
# If it's taking > 1m for `nvidia-smi` to return, it's likely the bus is still busy setting up the models.
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


Observe the started containers with `docker ps`:
```
CONTAINER ID   IMAGE                                                                      COMMAND                  CREATED          STATUS                    PORTS                                                                                                                                                                                                                                                                                NAMES
0f2f86615ea5   nvcr.io/nvidia/nemo-microservices/nv-ingest:24.12                       "/opt/conda/bin/tini…"   35 seconds ago   Up 33 seconds             0.0.0.0:7670->7670/tcp, :::7670->7670/tcp                                                                                                                                                                                                                                            nv-ingest-nv-ingest-ms-runtime-1
de44122c6ddc   otel/opentelemetry-collector-contrib:0.91.0                                "/otelcol-contrib --…"   14 hours ago     Up 24 seconds             0.0.0.0:4317-4318->4317-4318/tcp, :::4317-4318->4317-4318/tcp, 0.0.0.0:8888-8889->8888-8889/tcp, :::8888-8889->8888-8889/tcp, 0.0.0.0:13133->13133/tcp, :::13133->13133/tcp, 55678/tcp, 0.0.0.0:32849->9411/tcp, :::32848->9411/tcp, 0.0.0.0:55680->55679/tcp, :::55680->55679/tcp   nv-ingest-otel-collector-1
02c9ab8c6901   nvcr.io/nvidia/nemo-microservices/cached:0.2.0                          "/opt/nvidia/nvidia_…"   14 hours ago     Up 24 seconds             0.0.0.0:8006->8000/tcp, :::8006->8000/tcp, 0.0.0.0:8007->8001/tcp, :::8007->8001/tcp, 0.0.0.0:8008->8002/tcp, :::8008->8002/tcp                                                                                                                                                      nv-ingest-cached-1
d49369334398   nvcr.io/nim/nvidia/nv-embedqa-e5-v5:1.1.0                                  "/opt/nvidia/nvidia_…"   14 hours ago     Up 33 seconds             0.0.0.0:8012->8000/tcp, :::8012->8000/tcp, 0.0.0.0:8013->8001/tcp, :::8013->8001/tcp, 0.0.0.0:8014->8002/tcp, :::8014->8002/tcp                                                                                                                                                      nv-ingest-embedding-1
508715a24998   nvcr.io/nvidia/nemo-microservices/nv-yolox-structured-images-v1:0.2.0   "/opt/nvidia/nvidia_…"   14 hours ago     Up 33 seconds             0.0.0.0:8000-8002->8000-8002/tcp, :::8000-8002->8000-8002/tcp                                                                                                                                                                                                                        nv-ingest-yolox-1
5b7a174a0a85   nvcr.io/nvidia/nemo-microservices/deplot:1.0.0                          "/opt/nvidia/nvidia_…"   14 hours ago     Up 33 seconds             0.0.0.0:8003->8000/tcp, :::8003->8000/tcp, 0.0.0.0:8004->8001/tcp, :::8004->8001/tcp, 0.0.0.0:8005->8002/tcp, :::8005->8002/tcp                                                                                                                                                      nv-ingest-deplot-1
430045f98c02   nvcr.io/nvidia/nemo-microservices/paddleocr:0.2.0                       "/opt/nvidia/nvidia_…"   14 hours ago     Up 24 seconds             0.0.0.0:8009->8000/tcp, :::8009->8000/tcp, 0.0.0.0:8010->8001/tcp, :::8010->8001/tcp, 0.0.0.0:8011->8002/tcp, :::8011->8002/tcp                                                                                                                                                      nv-ingest-paddle-1
8e587b45821b   grafana/grafana                                                            "/run.sh"                14 hours ago     Up 33 seconds             0.0.0.0:3000->3000/tcp, :::3000->3000/tcp                                                                                                                                                                                                                                            grafana-service
aa2c0ec387e2   redis/redis-stack                                                          "/entrypoint.sh"         14 hours ago     Up 33 seconds             0.0.0.0:6379->6379/tcp, :::6379->6379/tcp, 8001/tcp                                                                                                                                                                                                                                  nv-ingest-redis-1
bda9a2a9c8b5   openzipkin/zipkin                                                          "start-zipkin"           14 hours ago     Up 33 seconds (healthy)   9410/tcp, 0.0.0.0:9411->9411/tcp, :::9411->9411/tcp                                                                                                                                                                                                                                  nv-ingest-zipkin-1
ac27e5297d57   prom/prometheus:latest                                                     "/bin/prometheus --w…"   14 hours ago     Up 33 seconds             0.0.0.0:9090->9090/tcp, :::9090->9090/tcp                                                                                                                                                                                                                                            nv-ingest-prometheus-1
```

> [!TIP]
> nv-ingest is in Early Access mode, meaning the codebase gets frequent updates. To build an updated nv-ingest service container with the latest changes you can:
> ```
> docker compose build
> ```
>
> After the image builds, run `docker compose --profile retrieval up` or `docker compose up --build` as explained in the previous step.

### Step 2: Install Python dependencies

To interact with the nv-ingest service, you can do so from the host, or by `docker exec`-ing into the nv-ingest container.

To interact from the host, you'll need a Python environment and install the client dependencies:

```bash
# conda not required, but makes it easy to create a fresh python environment
conda env create --name nv-ingest-dev --file ./conda/environments/nv_ingest_environment.yml
conda activate nv-ingest-dev

cd client
pip install .

# When not using Conda, pip dependencies for the client can be installed directly via pip. Pip based installation of
# the ingest service is not supported.
cd client
pip install .
```

> [!NOTE]
> Interacting from the host depends on the appropriate port being exposed from the nv-ingest container to the host as defined in [docker-compose.yaml](docker-compose.yaml).
>
> If you prefer, you can disable exposing that port, and interact with the nv-ingest service directly from within its container.
>
> To interact within the container:
> ```
> docker exec -it nv-ingest-nv-ingest-ms-runtime-1 bash
> ```
> You'll be in the `/workspace` directory, which has `DATASET_ROOT` from the .env file mounted at `./data`. The pre-activated `morpheus` conda environment has all the python client libraries pre-installed:
> ```
> (morpheus) root@aba77e2a4bde:/workspace#
> ```
>
> From the bash prompt above, you can run nv-ingest-cli and Python examples described below.

### Step 3: Ingesting Documents

You can submit jobs programmatically in Python or via the [NV-Ingest CLI](nv-ingest_cli.md).

In the below examples, we are doing text, chart, table, and image extraction:

- **extract_text** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to find and extract text from pages.
- **extract_images** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to extract images.
- **extract_tables** — Uses [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) to find tables and charts. Uses [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for table extraction, and [Deplot](https://huggingface.co/google/deplot) and CACHED for chart extraction.
- **extract_charts** — (Optional) Enables or disables Deplot and CACHED for chart extraction.

> [!IMPORTANT]
> `extract_tables` controls extraction for both tables and charts. You can optionally disable chart extraction by setting `extract_charts` to false.

#### In Python

> [!NOTE]
> You can find more examples in [the client examples folder](client/client_examples/examples/).


```python
import logging, time

from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.tasks import ExtractTask
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
result = client.fetch_job_result(job_id, timeout=60)
print(f"Got {len(result)} results")
```

#### Using the `nv-ingest-cli`

> [!NOTE]
> You can find more examples in [the client examples folder](client/client_examples/examples/).

```shell
nv-ingest-cli \
  --doc ./data/multimodal_test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_tables": "true", "extract_images": "true"}' \
  --client_host=localhost \
  --client_port=7670
```

You should notice output indicating document processing status, followed by a breakdown of time spent during job execution:
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

#### When processing has completed, you'll have separate result files for text and image data:
```shell
ls -R processed_docs/
```
```shell
processed_docs/:
image  structured  text

processed_docs/image:
multimodal_test.pdf.metadata.json

processed_docs/structured:
multimodal_test.pdf.metadata.json

processed_docs/text:
multimodal_test.pdf.metadata.json
```
For the full metadata definitions, refer to [Content Metadata](/docs/docs/extraction/content-metadata.md).

#### We also provide a script for inspecting [extracted images](src/util/image_viewer.py)

First, install `tkinter` by running the following code. Choose the code for your OS.

- For Ubuntu/Debian Linux:

  ```shell
  sudo apt-get update
  sudo apt-get install python3-tk
  ```

- For Fedora/RHEL Linux:

  ```shell
  sudo dnf install python3-tkinter
  ```

- For macOS using Homebrew:

  ```shell
  brew install python-tk
  ```

Then run the following command to execute the script for inspecting the extracted image:
```shell
python src/util/image_viewer.py --file_path ./processed_docs/image/multimodal_test.pdf.metadata.json
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
