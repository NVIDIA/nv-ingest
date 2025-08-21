# Deploy With Podman Compose (Self-Hosted) for NeMo Retriever Extraction

Use this documentation to get started using [NeMo Retriever extraction](overview.md) in self-hosted mode.

## Step 1: Set up Windows Subsystem for Linux 2 - WSL2 - Distro

1. Ensure virtualization is enabled in the system BIOS
    In Windows, open the Task Manager. Select the Performance tab and click on CPU. Check if Virtualization is enabled. If it is disabled, see [here](https://support.microsoft.com/en-us/windows/enable-virtualization-on-windows-c5578302-6e43-4b4b-a449-8ced115f58e1) to enable.

2. Open the NVIDIA-Workbench

    [Install WSL2](https://assets.ngc.nvidia.com/products/api-catalog/rtx/NIMSetup.exe). For additional instructions refer to the [documentation](https://docs.nvidia.com/nim/wsl2/latest/getting-started.html#installation).
    
    Once installed, open the NVIDIA-Workbench WSL2 distro using the following command in the Windows terminal.
    
    `wsl -d NVIDIA-Workbench` 

## Step 2: Starting Containers

This example demonstrates how to use the provided [podman-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/rtx/podman-compose.yaml) to start all needed services with a few commands. Podman is the recommended container platform for running this project on Windows as it offers a free and open-source container solution, whereas Docker requires a commercial license for certain use cases.


!!! warning

    NIM containers on their first startup can take 10-15 minutes to pull and fully load models.

1. [Generate API keys](ngc-api-key.md) and authenticate with NGC with the `podman login` command:

    ```shell
    echo "$NGC_API_KEY" | podman login nvcr.io --username '$oauthtoken' --password-stdin
    ```

2. Install conda; see [documentation](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2) for steps

3. Git clone the repo:

    `git clone -b release/25.6.3 https://github.com/nvidia/nv-ingest`

4. Change the directory to the cloned repo
   
    `cd nv-ingest`.

5. Create a .env file that contains your NVIDIA Build API key.

    !!! note

        If you use an NGC personal key, then you should provide the same value for all keys, but you must specify each environment variable individually. In the past, you could create an API key. If you have an API key, you can still use that. For more information, refer to [Generate Your NGC Keys](ngc-api-key.md) and [Environment Configuration Variables](environment-config.md).

    ```
    # Container images must access resources from NGC.

    NGC_API_KEY=<key to download containers from NGC>
    NIM_NGC_API_KEY=<key to download model files after containers start>
    ```

6. Install podman compose
    ```
    conda create --name podman python=3.12.11
    conda activate podman
    pip install podman-compose>=1.1.0
    ```

7. Make sure NVIDIA is set as your default container runtime before running the podman compose command. For Podman, NVIDIA recommends using [CDI](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-podman) for accessing NVIDIA devices in containers. Check the name of the generated devices by running

    `nvidia-ctk cdi list`
   
    The following example output is for a machine with a single GPU
    ```
    INFO[0000] Found 1 CDI devices
    nvidia.com/gpu=all
    ```
    
    If you see the following error when running `nvidia-ctk cdi list`, try again after removing `/etc/cdi/nvidia.json` file.
    
    ```
    WARN[0000] The following registry errors were reported:
    WARN[0000] /etc/cdi/nvidia.json: [conflicting device "nvidia.com/gpu=all" (specs "/etc/cdi/nvidia.yaml", "/etc/cdi/nvidia.json")]
    WARN[0000] /etc/cdi/nvidia.yaml: [conflicting device "nvidia.com/gpu=all" (specs "/etc/cdi/nvidia.yaml", "/etc/cdi/nvidia.json")]
    INFO[0000] Found 0 CDI devices
    ```


9. Start core services. This example uses the table-structure profile.  For more information about other profiles, see [Profile Information](#profile-information).

    `podman-compose -f rtx/podman-compose.yaml --env-file .env --profile retrieval --profile table-structure up`

    !!! tip

        By default, we have configured log levels to be verbose. It's possible to observe service startup proceeding. You will notice a lot of log messages. Disable verbose logging by configuring `NIM_TRITON_LOG_VERBOSE=0` for each NIM in [podman-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/rtx/podman-compose.yaml).

10. When core services have fully started, `nvidia-smi` should show processes like the following:

    ```
    # If it's taking > 1m for `nvidia-smi` to return, the bus will likely be busy setting up the models.
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |    0   N/A  N/A     80461      C   milvus                                     1438MiB |
    |    0   N/A  N/A     83791      C   tritonserver                               2492MiB |
    |    0   N/A  N/A     85605      C   tritonserver                               1896MiB |
    |    0   N/A  N/A     85889      C   tritonserver                               2824MiB |
    |    0   N/A  N/A     88253      C   tritonserver                               2824MiB |
    |    0   N/A  N/A     91194      C   tritonserver                               4546MiB |
    +---------------------------------------------------------------------------------------+
    ```

11. Observe the started containers with `podman ps`:

    ```
    CONTAINER ID  IMAGE                                                       COMMAND               CREATED         STATUS                   PORTS                                                                                                                                                          NAMES
    96486991ead2  docker.io/redis/redis-stack:latest                          /entrypoint.sh        12 minutes ago  Up 11 minutes            0.0.0.0:6379->6379/tcp                                                                                                                                         rtx_redis_1
    316215b3e265  nvcr.io/nim/nvidia/nemoretriever-page-elements-v2:1.3.0     /opt/nim/start_se...  12 minutes ago  Up 11 minutes            0.0.0.0:8000-8002->8000-8002/tcp                                                                                                                               rtx_page-elements_1
    cc65ee74e5c5  nvcr.io/nim/nvidia/nemoretriever-graphic-elements-v1:1.3.0  /opt/nim/start_se...  12 minutes ago  Up 11 minutes            0.0.0.0:8003-8005->8000-8002/tcp                                                                                                                               rtx_graphic-elements_1
    047307c55a0a  nvcr.io/nim/nvidia/nemoretriever-table-structure-v1:1.3.0   /opt/nim/start_se...  12 minutes ago  Up 11 minutes            0.0.0.0:8006-8008->8000-8002/tcp                                                                                                                               rtx_table-structure_1
    c60cc4fc01ad  nvcr.io/nim/baidu/paddleocr:1.3.0                           /opt/nim/start_se...  11 minutes ago  Up 11 minutes            0.0.0.0:8009-8011->8000-8002/tcp                                                                                                                               rtx_paddle_1
    d9f0b51716c3  nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.6.0         /opt/nim/start_se...  11 minutes ago  Up 11 minutes            0.0.0.0:8012-8014->8000-8002/tcp                                                                                                                               rtx_embedding_1
    82bb43bb182d  nvcr.io/nvidia/nemo-microservices/nv-ingest:25.6.2                                11 minutes ago  Up 11 minutes (healthy)  0.0.0.0:7670-7671->7670-7671/tcp, 0.0.0.0:8265->8265/tcp                                                                                                       rtx_nv-ingest-ms-runtime_1
    682f62dc4e07  docker.io/openzipkin/zipkin:latest                                                11 minutes ago  Up 11 minutes (healthy)  0.0.0.0:9411->9411/tcp                                                                                                                                         rtx_zipkin_1
    a39bde9a1937  docker.io/prom/prometheus:latest                            --web.console.tem...  11 minutes ago  Up 11 minutes            0.0.0.0:9090->9090/tcp                                                                                                                                         rtx_prometheus_1
    62dfcbb670d1  docker.io/grafana/grafana:latest                                                  11 minutes ago  Up 11 minutes            0.0.0.0:3000->3000/tcp                                                                                                                                         grafana-service
    9aa7ce0bf704  quay.io/coreos/etcd:v3.5.5                                  etcd -advertise-c...  11 minutes ago  Up 11 minutes (healthy)                                                                                                                                                                 milvus-etcd
    9aebdfdafb60  docker.io/minio/minio:RELEASE.2023-03-20T20-16-18Z          minio server /min...  11 minutes ago  Up 11 minutes (healthy)  0.0.0.0:9000-9001->9000-9001/tcp                                                                                                                               minio
    801cab09e86e  docker.io/otel/opentelemetry-collector-contrib:0.91.0       --config=/etc/ote...  11 minutes ago  Up 11 minutes            0.0.0.0:4317-4318->4317-4318/tcp, 0.0.0.0:8889->8889/tcp, 0.0.0.0:9988->9988/tcp, 0.0.0.0:13133->13133/tcp, 0.0.0.0:55680->55679/tcp, 0.0.0.0:39847->9411/tcp  rtx_otel-collector_1
    ee6a4c878401  docker.io/milvusdb/milvus:v2.5.3-gpu                        milvus run standa...  11 minutes ago  Up 11 minutes (healthy)  0.0.0.0:9091->9091/tcp, 0.0.0.0:19530->19530/tcp                                                                                                               milvus-standalone
    ceb5f134adcc  docker.io/zilliz/attu:v2.3.5                                /bin/bash -c /app...  11 minutes ago  Up 11 minutes            0.0.0.0:3001->3000/tcp                                                                                                                                         milvus-attu
    ```


## Step 3: Install Python Dependencies

You can interact with the NV-Ingest service from the host, or by using `podman exec` to run commands in the NV-Ingest container.

To interact from the host, you'll need a Python environment and install the client dependencies:

```
# conda not required but makes it easy to create a fresh Python environment
conda create --name nv-ingest-dev python=3.12.11
conda activate nv-ingest-dev
pip install nv-ingest==25.6.2 nv-ingest-api==25.6.2 nv-ingest-client==25.6.2
```

!!! tip

    To confirm that you have activated your Conda environment, run `which pip` and `which python`, and confirm that you see `nvingest` in the result. You can do this before any pip or python command that you run.


!!! note

    Interacting from the host depends on the appropriate port being exposed from the nv-ingest container to the host as defined in [podman-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/rtx/podman-compose.yaml#L141). If you prefer, you can disable exposing that port and interact with the NV-Ingest service directly from within its container. To interact within the container run `podman exec -it nv-ingest-nv-ingest-ms-runtime-1 bash`. You'll be in the `/workspace` directory with `DATASET_ROOT` from the .env file mounted at `./data`. The pre-activated `nv_ingest_runtime` conda environment has all the Python client libraries pre-installed and you should see `(nv_ingest_runtime) root@aba77e2a4bde:/workspace#`. From the bash prompt above, you can run the nv-ingest-cli and Python examples described following.


## Step 4: Ingesting Documents

You can submit jobs programmatically in Python or using the [NV-Ingest CLI](nv-ingest_cli.md).

In the below examples, we are doing text, chart, table, and image extraction:

- **extract_text** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to find and extract text from pages.
- **extract_images** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to extract images.
- **extract_tables** — Uses [object detection family of NIMs](https://docs.nvidia.com/nim/ingestion/object-detection/latest/overview.html) to find tables and charts, and [PaddleOCR NIM](https://build.nvidia.com/baidu/paddleocr/modelcard) for table extraction.
- **extract_charts** — Enables or disables chart extraction, also based on the object detection NIM family.


### In Python

!!! tip

    For more Python examples, refer to [NV-Ingest: Python Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/python_client_usage.ipynb).


```python
import logging, os, time

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
        # extract_method="nemoretriever_parse", #Slower, but maximally accurate, especially for PDFs with pages that are scanned images
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

# Return both successes and failures
# Use for large batches where you want successful chunks/pages to be committed, while collecting detailed diagnostics for failures.
results, failures = ingestor.ingest(show_progress=True, return_failures=True)

# Return only successes
# results = ingestor.ingest(show_progress=True)

t1 = time.time()
print(f"Total time: {t1 - t0} seconds")

# results blob is directly inspectable
print(ingest_json_results_to_blob(results[0]))

# (optional) Review any failures that were returned
if failures:
    print(f"There were {len(failures)} failures. Sample: {failures[0]}")
```

!!! note

    To use library mode with nemoretriever_parse, uncomment `extract_method="nemoretriever_parse"` in the previous code. For more information, refer to [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md).


```
Starting ingestion..
1 records to insert to milvus
logged 8 records
Time taken: 5.479151725769043 seconds
This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost   Chart 1 - Hammer - Powerdrill - Bluetooth speaker - Minifridge - Premium desk fan Dollars $- - $20.00 - $40.00 - $60.00 - $80.00 - $100.00 - $120.00 - $140.00 - $160.00 Cost
Table 1
| This table describes some animals, and some activities they might be doing in specific locations. | This table describes some animals, and some activities they might be doing in specific locations. | This table describes some animals, and some activities they might be doing in specific locations. |
| Animal | Activity | Place |
| Giraffe | Driving a car | At the beach |
| Lion | Putting on sunscreen | At the park |
| Cat | Jumping onto a laptop | In a home office |
| Dog | Chasing a squirrel | In the front yard |
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
image_caption:[]
image_caption:[]
Below,is a high-quality picture of some shapes          Picture
Table 2
| This table shows some popular colors that cars might come in | This table shows some popular colors that cars might come in | This table shows some popular colors that cars might come in | This table shows some popular colors that cars might come in |
| Car | Color1 | Color2 | Color3 |
| Coupe | White | Silver | Flat Gray |
| Sedan | White | Metallic Gray | Matte Gray |
| Minivan | Gray | Beige | Black |
| Truck | Dark Gray | Titanium Gray | Charcoal |
| Convertible | Light Gray | Graphite | Slate Gray |
Section One
This is the first section of the document. It has some more placeholder text to show how 
the document looks like. The text is not meant to be meaningful or informative, but rather to 
demonstrate the layout and formatting of the document.
• This is the first bullet point
• This is the second bullet point
• This is the third bullet point
Section Two
This is the second section of the document. It is more of the same as we’ve seen in the rest 
of the document. The content is meaningless, but the intent is to create a very simple 
smoke test to ensure extraction is working as intended. This will be used in CI as time goes 
on to ensure that changes we make to the library do not negatively impact our accuracy.
Table 2
This table shows some popular colors that cars might come in.
Car Color1 Color2 Color3
Coupe White Silver Flat Gray
Sedan White Metallic Gray Matte Gray
Minivan Gray Beige Black
Truck Dark Gray Titanium Gray Charcoal
Convertible Light Gray Graphite Slate Gray
Picture
Below, is a high-quality picture of some shapes.
image_caption:[]
image_caption:[]
This chart shows some average frequency ranges for speaker drivers. Frequency Ranges ofSpeaker Drivers   Tweeter - Midrange - Midwoofer - Subwoofer Chart2 Hertz (log scale) 1 - 10 - 100 - 1000 - 10000 - 100000 FrequencyRange Start (Hz) - Frequency Range End (Hz)
Chart 2
This chart shows some average frequency ranges for speaker drivers.
Conclusion
This is the conclusion of the document. It has some more placeholder text, but the most 
important thing is that this is the conclusion. As we end this document, we should have 
been able to extract 2 tables, 2 charts, and some text including 3 bullet points.
image_caption:[]

```

### Using the `nv-ingest-cli`

!!! tip

    There is a Jupyter notebook available to help you get started with the CLI. For more information, refer to [CLI Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/cli_client_usage.ipynb).

```shell
nv-ingest-cli \
  --doc ./data/multimodal_test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_tables": "true", "extract_images": "true", "extract_charts": "true"}' \
  --client_host=localhost \
  --client_port=7671
  --client_type=simple
```

You should notice output indicating document processing status followed by a breakdown of time spent during job execution:
```
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
[nltk_data] Downloading package punkt_tab to
[nltk_data]     /home/akberr/miniconda3/envs/nv-ingest-
[nltk_data]     dev/lib/python3.12/site-
[nltk_data]     packages/llama_index/core/_static/nltk_cache...
[nltk_data]   Package punkt_tab is already up-to-date!
INFO:nv_ingest_client.nv_ingest_cli:Processing 1 documents.
INFO:nv_ingest_client.nv_ingest_cli:Output will be written to: ./processed_docs
Processing files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:53<00:00, 53.10s/file, pages_per_sec=0.06]
INFO:nv_ingest_client.cli.util.processing:message_broker_task_source: Avg: 65.27 ms, Median: 65.27 ms, Total Time: 65.27 ms, Total % of Trace Computation: 0.11%
INFO:nv_ingest_client.cli.util.processing:broker_source_network_in: Avg: 27.57 ms, Median: 27.57 ms, Total Time: 27.57 ms, Total % of Trace Computation: 0.05%
INFO:nv_ingest_client.cli.util.processing:metadata_injector: Avg: 94.48 ms, Median: 94.48 ms, Total Time: 94.48 ms, Total % of Trace Computation: 0.16%
INFO:nv_ingest_client.cli.util.processing:metadata_injector_channel_in: Avg: 35061.50 ms, Median: 35061.50 ms, Total Time: 35061.50 ms, Total % of Trace Computation: 60.85%
INFO:nv_ingest_client.cli.util.processing:pdf_extraction: Avg: 897.85 ms, Median: 186.52 ms, Total Time: 4489.25 ms, Total % of Trace Computation: 7.79%
INFO:nv_ingest_client.cli.util.processing:pdf_extraction_channel_in: Avg: 4997.63 ms, Median: 4997.63 ms, Total Time: 4997.63 ms, Total % of Trace Computation: 8.67%
INFO:nv_ingest_client.cli.util.processing:audio_extractor: Avg: 0.07 ms, Median: 0.07 ms, Total Time: 0.07 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:audio_extractor_channel_in: Avg: 429.84 ms, Median: 429.84 ms, Total Time: 429.84 ms, Total % of Trace Computation: 0.75%
INFO:nv_ingest_client.cli.util.processing:docx_extractor: Avg: 0.08 ms, Median: 0.08 ms, Total Time: 0.08 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:docx_extractor_channel_in: Avg: 475.31 ms, Median: 475.31 ms, Total Time: 475.31 ms, Total % of Trace Computation: 0.82%
INFO:nv_ingest_client.cli.util.processing:pptx_extractor: Avg: 0.11 ms, Median: 0.11 ms, Total Time: 0.11 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:pptx_extractor_channel_in: Avg: 506.23 ms, Median: 506.23 ms, Total Time: 506.23 ms, Total % of Trace Computation: 0.88%
INFO:nv_ingest_client.cli.util.processing:image_extraction: Avg: 0.06 ms, Median: 0.06 ms, Total Time: 0.06 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:image_extraction_channel_in: Avg: 460.53 ms, Median: 460.53 ms, Total Time: 460.53 ms, Total % of Trace Computation: 0.80%
INFO:nv_ingest_client.cli.util.processing:html_extractor: Avg: 0.06 ms, Median: 0.06 ms, Total Time: 0.06 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:html_extractor_channel_in: Avg: 456.82 ms, Median: 456.82 ms, Total Time: 456.82 ms, Total % of Trace Computation: 0.79%
INFO:nv_ingest_client.cli.util.processing:infographic_extraction: Avg: 0.05 ms, Median: 0.05 ms, Total Time: 0.05 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:infographic_extraction_channel_in: Avg: 493.51 ms, Median: 493.51 ms, Total Time: 493.51 ms, Total % of Trace Computation: 0.86%
INFO:nv_ingest_client.cli.util.processing:table_extraction: Avg: 785.79 ms, Median: 785.79 ms, Total Time: 1571.57 ms, Total % of Trace Computation: 2.73%
INFO:nv_ingest_client.cli.util.processing:table_extraction_channel_in: Avg: 471.89 ms, Median: 471.89 ms, Total Time: 471.89 ms, Total % of Trace Computation: 0.82%
INFO:nv_ingest_client.cli.util.processing:chart_extraction: Avg: 1395.97 ms, Median: 1410.31 ms, Total Time: 4187.90 ms, Total % of Trace Computation: 7.27%
INFO:nv_ingest_client.cli.util.processing:chart_extraction_channel_in: Avg: 575.46 ms, Median: 575.46 ms, Total Time: 575.46 ms, Total % of Trace Computation: 1.00%
INFO:nv_ingest_client.cli.util.processing:image_filter: Avg: 0.03 ms, Median: 0.03 ms, Total Time: 0.03 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:image_filter_channel_in: Avg: 488.90 ms, Median: 488.90 ms, Total Time: 488.90 ms, Total % of Trace Computation: 0.85%
INFO:nv_ingest_client.cli.util.processing:image_deduplication: Avg: 0.03 ms, Median: 0.03 ms, Total Time: 0.03 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:image_deduplication_channel_in: Avg: 444.60 ms, Median: 444.60 ms, Total Time: 444.60 ms, Total % of Trace Computation: 0.77%
INFO:nv_ingest_client.cli.util.processing:text_splitter: Avg: 0.03 ms, Median: 0.03 ms, Total Time: 0.03 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:text_splitter_channel_in: Avg: 472.41 ms, Median: 472.41 ms, Total Time: 472.41 ms, Total % of Trace Computation: 0.82%
INFO:nv_ingest_client.cli.util.processing:text_embedding: Avg: 0.07 ms, Median: 0.07 ms, Total Time: 0.07 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:text_embedding_channel_in: Avg: 482.48 ms, Median: 482.48 ms, Total Time: 482.48 ms, Total % of Trace Computation: 0.84%
INFO:nv_ingest_client.cli.util.processing:image_captioning: Avg: 0.03 ms, Median: 0.03 ms, Total Time: 0.03 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:image_captioning_channel_in: Avg: 455.90 ms, Median: 455.90 ms, Total Time: 455.90 ms, Total % of Trace Computation: 0.79%
INFO:nv_ingest_client.cli.util.processing:image_storage: Avg: 0.04 ms, Median: 0.04 ms, Total Time: 0.04 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:image_storage_channel_in: Avg: 451.04 ms, Median: 451.04 ms, Total Time: 451.04 ms, Total % of Trace Computation: 0.78%
INFO:nv_ingest_client.cli.util.processing:embedding_storage: Avg: 0.04 ms, Median: 0.04 ms, Total Time: 0.04 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:embedding_storage_channel_in: Avg: 463.37 ms, Median: 463.37 ms, Total Time: 463.37 ms, Total % of Trace Computation: 0.80%
INFO:nv_ingest_client.cli.util.processing:No unresolved time detected. Trace times account for the entire elapsed duration.
INFO:nv_ingest_client.cli.util.processing:Processed 1 files in 53.11 seconds.
INFO:nv_ingest_client.cli.util.processing:Total pages processed: 3
INFO:nv_ingest_client.cli.util.processing:Throughput (Pages/sec): 0.06
INFO:nv_ingest_client.cli.util.processing:Throughput (Files/sec): 0.02
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



## Profile Information

The values that you specify in the `--profile` option of your `podman-compose up` command are explained in the following table. 
You can specify multiple `--profile` options.

| Profile               | Type     | Description                                                       | 
|-----------------------|----------|-------------------------------------------------------------------| 
| `retrieval`           | Core     | Enables the embedding NIM and (GPU accelerated) Milvus.           | 
| `table-structure`     | Core     | Enables the yolox table structure NIM which enhances markdown formatting of extracted table content. This benefits answer generation by downstream LLMs. | 
| `audio`               | Advanced | Use [Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html) for processing audio files. For more information, refer to [Audio Processing](nemoretriever-parse.md). | 
| `nemoretriever-parse` | Advanced | Use [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse), which adds state-of-the-art text and table extraction. For more information, refer to [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md). | 
| `vlm`                 | Advanced | Use [llama 3.2 11B Vision](https://build.nvidia.com/meta/llama-3.2-11b-vision-instruct/modelcard) for experimental image captioning of unstructured images. | 


!!! important

    Advanced features require additional GPU support and disk space. For more information, refer to [Support Matrix](support-matrix.md).



## Related Topics

- [Troubleshoot](troubleshoot.md)
- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Helm](helm.md)
- [Notebooks](notebooks.md)
- [Multimodal PDF Data Extraction](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
