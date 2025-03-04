# Quickstart Guide for NV-Ingest

To get started using NV-Ingest, you need to do a few things:

1. [Start supporting NIM microservices](#step-1-starting-containers) 🏗️
2. [Install the NVIDIA Ingest client dependencies in a Python environment](#step-2-install-python-dependencies) 🐍
3. [Submit ingestion job(s)](#step-3-ingesting-documents) 📓
4. [Inspect and consume results](#step-4-inspecting-and-consuming-results) 🔍

## Step 1: Starting Containers

This example demonstrates how to use the provided [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml) to start all needed services with a few commands.

!!! warning

    NIM containers on their first startup can take 10-15 minutes to pull and fully load models.


If you prefer, you can also [start services one by one](deployment.md) or run on Kubernetes by using [our Helm chart](https://github.com/NVIDIA/nv-ingest/blob/main/helm/README.md). Also, there are [additional environment variables](environment-config.md) you want to configure.

1. Git clone the repo:

    `git clone https://github.com/nvidia/nv-ingest`

2. Change the directory to the cloned repo
   
    `cd nv-ingest`.

3. [Generate API keys](ngc-api-key.md) and authenticate with NGC with the `docker login` command:

    ```shell
    # This is required to access pre-built containers and NIM microservices
    $ docker login nvcr.io
    Username: $oauthtoken
    Password: <Your Key>
    ```

    !!! note
   
        During the early access (EA) phase, you must apply for early access at [https://developer.nvidia.com/nemo-microservices-early-access/join](https://developer.nvidia.com/nemo-microservices-early-access/join). When your early access is approved, follow the instructions in the email to create an organization and team, link your profile, and generate your NGC API key.
   
4. Create a .env file containing your NGC API key and the following paths. For more information, refer to [Environment Configuration Variables](environment-config.md).

    ```
    # Container images must access resources from NGC.

    NGC_API_KEY=<key to download containers from NGC>
    NIM_NGC_API_KEY=<key to download model files after containers start>
    NVIDIA_BUILD_API_KEY=<key to use NIMs that are hosted on build.nvidia.com>
    
    DATASET_ROOT=<PATH_TO_THIS_REPO>/data
    NV_INGEST_ROOT=<PATH_TO_THIS_REPO>
    ```

    !!! note

        As configured by default in [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml#L52), the DePlot NIM is on a dedicated GPU. All other NIMs and the NV-Ingest container itself share a second. This avoids DePlot and other NIMs competing for VRAM on the same device. Change the `CUDA_VISIBLE_DEVICES` pinnings as desired for your system within [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml).
   
5. Make sure NVIDIA is set as your default container runtime before running the docker compose command with the command:

    `sudo nvidia-ctk runtime configure --runtime=docker --set-as-default`

6. Start all services:

    `docker compose --profile retrieval up`

    !!! tip

        By default, we have [configured log levels to be verbose](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml). It's possible to observe service startup proceeding. You will notice a lot of log messages. Disable verbose logging by configuring `NIM_TRITON_LOG_VERBOSE=0` for each NIM in [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml).
   
7. When all services have fully started, `nvidia-smi` should show processes like the following:

    ```
    # If it's taking > 1m for `nvidia-smi` to return, the bus will likely be busy setting up the models.
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

8. Observe the started containers with `docker ps`:

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

    !!! tip

        NV-Ingest is in early access (EA) mode, meaning the codebase gets frequent updates. To build an updated NV-Ingest service container with the latest changes, you can run `docker compose build`. After the image builds, run `docker compose --profile retrieval up` or `docker compose up --build` as explained in the previous step.


## Step 2: Install Python Dependencies

You can interact with the NV-Ingest service from the host or by `docker exec`-ing into the NV-Ingest container.

To interact from the host, you'll need a Python environment and install the client dependencies:
```
# conda not required but makes it easy to create a fresh Python environment
conda env create --name nv-ingest-dev python=3.10
conda activate nv-ingest-dev
cd client
pip install .
```

!!! note

    Interacting from the host depends on the appropriate port being exposed from the nv-ingest container to the host as defined in [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml#L141). If you prefer, you can disable exposing that port and interact with the NV-Ingest service directly from within its container. To interact within the container run `docker exec -it nv-ingest-nv-ingest-ms-runtime-1 bash`. You'll be in the `/workspace` directory with `DATASET_ROOT` from the .env file mounted at `./data`. The pre-activated `morpheus` conda environment has all the Python client libraries pre-installed and you should see `(morpheus) root@aba77e2a4bde:/workspace#`. From the bash prompt above, you can run the nv-ingest-cli and Python examples described following.


## Step 3: Ingesting Documents

You can submit jobs programmatically in Python or using the [NV-Ingest CLI](nv-ingest_cli.md).

In the below examples, we are doing text, chart, table, and image extraction:

- **extract_text** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to find and extract text from pages.
- **extract_images** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to extract images.
- **extract_tables** — Uses [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) to find tables and charts. Uses [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for table extraction, and [Deplot](https://huggingface.co/google/deplot) and CACHED for chart extraction.
- **extract_charts** — (Optional) Enables or disables Deplot and CACHED for chart extraction.


!!! tip

    `extract_tables` controls extraction for both tables and charts. You can optionally disable chart extraction by setting `extract_charts` to false.


### In Python

You can find more documentation and examples in [the client examples folder](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/).


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

### Using the `nv-ingest-cli`

You can find more nv-ingest-cli examples in [the client examples folder](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/).

```shell
nv-ingest-cli \
  --doc ./data/multimodal_test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_tables": "true", "extract_images": "true"}' \
  --client_host=localhost \
  --client_port=7670
```

You should notice output indicating document processing status followed by a breakdown of time spent during job execution:
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

## Step 4: Inspecting and Consuming Results

After the ingestion steps above have been completed, you should be able to find the `text` and `image` subfolders inside your processed docs folder. Each will contain JSON-formatted extracted content and metadata.

When processing has completed, you'll have separate result files for text and image data:
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

For the full metadata definitions, refer to [Content Metadata](content-metadata.md). 

We also provide a script for inspecting [extracted images](https://github.com/NVIDIA/nv-ingest/blob/main/src/util/image_viewer.py).

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

Then, run the following command to execute the script for inspecting the extracted image:

```shell
python src/util/image_viewer.py --file_path ./processed_docs/image/multimodal_test.pdf.metadata.json
```

!!! tip

    Beyond inspecting the results, you can read them into things like [llama-index](https://github.com/NVIDIA/nv-ingest/blob/main/examples/llama_index_multimodal_rag.ipynb) or [langchain](https://github.com/NVIDIA/nv-ingest/blob/main/examples/langchain_multimodal_rag.ipynb) retrieval pipelines. Also, checkout our [demo using a retrieval pipeline on build.nvidia.com](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag) to query over document content pre-extracted with NV-Ingest.


## Repo Structure

Beyond the relevant documentation, examples, and other links above, below is a description of the contents in this repo's folders:

- [.github](https://github.com/NVIDIA/nv-ingest/tree/main/.github): GitHub repo configuration files
- [ci](https://github.com/NVIDIA/nv-ingest/tree/main/ci): Scripts used to build the NV-Ingest container and other packages
- [client](https://github.com/NVIDIA/nv-ingest/tree/main/client): Docs and source code for the nv-ingest-cli utility
- [config](https://github.com/NVIDIA/nv-ingest/tree/main/config): Various .yaml files defining configuration for OTEL, Prometheus
- [data](https://github.com/NVIDIA/nv-ingest/tree/main/data): Sample PDFs provided for testing convenience
- [docker](https://github.com/NVIDIA/nv-ingest/tree/main/docker): Houses scripts used by the nv-ingest docker container
- [docs](https://github.com/NVIDIA/nv-ingest/tree/main/docs/docs): Various READMEs describing deployment, metadata schemas, auth and telemetry setup
- [examples](https://github.com/NVIDIA/nv-ingest/tree/main/examples): Example notebooks, scripts, and longer-form tutorial content
- [helm](https://github.com/NVIDIA/nv-ingest/tree/main/helm): Documentation for deploying NV-Ingest to a Kubernetes cluster via Helm chart
- [skaffold](https://github.com/NVIDIA/nv-ingest/tree/main/skaffold): Skaffold configuration
- [src](https://github.com/NVIDIA/nv-ingest/tree/main/src): Source code for the NV-Ingest pipelines and service
- [tests](https://github.com/NVIDIA/nv-ingest/tree/main/tests): Unit tests for NV-Ingest

## Notices

### Third-Party License Notice:

If configured to do so, this project will download and install additional third-party open-source software projects. Review the license terms of these open-source projects before use:

https://pypi.org/project/pdfservices-sdk/

- **`INSTALL_ADOBE_SDK`**:
  - **Description**: If set to `true`, the Adobe SDK will be installed in the container at launch time. This is
    required if you want to use the Adobe extraction service for PDF decomposition. Review the
    [license agreement](https://github.com/adobe/pdfservices-python-sdk?tab=License-1-ov-file) for the
    pdfservices-sdk before enabling this option.


### Contributing

We require that all contributors "sign off" on their commits. This certifies that the contribution is your original
work, or you have rights to submit it under the same license or a compatible license.

Any contribution that contains commits that aren't signed off won't be accepted.

To sign off on a commit, use the --signoff (or -s) option when committing your changes:

```
$ git commit -s -m "Add cool feature."
```

This appends the following to your commit message:

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

  (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open-source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
