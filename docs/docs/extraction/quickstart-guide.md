# Quickstart Guide for NeMo Retriever Extraction (Self-Hosted)

Use this documentation to get started using [NeMo Retriever extraction](overview.md) in self-hosted mode.


## Step 1: Starting Containers

This example demonstrates how to use the provided [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml) to start all needed services with a few commands.

!!! warning

    NIM containers on their first startup can take 10-15 minutes to pull and fully load models.


If you prefer, you can run on Kubernetes by using [our Helm chart](https://github.com/NVIDIA/nv-ingest/blob/main/helm/README.md). Also, there are [additional environment variables](environment-config.md) you might want to configure.

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
   
4. Create a .env file that contains your NVIDIA Build API key.

    !!! note

        If you use an NGC personal key, then you should provide the same value for all keys, but you must specify each environment variable individually. In the past, you could create an API key. If you have an API key, you can still use that. For more information, refer to [Generate Your NGC Keys](ngc-api-key.md) and [Environment Configuration Variables](environment-config.md).

    ```
    # Container images must access resources from NGC.

    NGC_API_KEY=<key to download containers from NGC>
    NIM_NGC_API_KEY=<key to download model files after containers start>
    ```
   
5. Make sure NVIDIA is set as your default container runtime before running the docker compose command with the command:

    `sudo nvidia-ctk runtime configure --runtime=docker --set-as-default`

6. Start core services. This example uses the table-structure profile.  For more information about other profiles, see [Profile Information](#profile-information).

    `docker compose --profile retrieval --profile table-structure up`

    !!! tip

        By default, we have [configured log levels to be verbose](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml). It's possible to observe service startup proceeding. You will notice a lot of log messages. Disable verbose logging by configuring `NIM_TRITON_LOG_VERBOSE=0` for each NIM in [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml).

7. When core services have fully started, `nvidia-smi` should show processes like the following:

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

8. Observe the started containers with `docker ps`:

    ```
    CONTAINER ID   IMAGE                                                                                                  COMMAND                  CREATED          STATUS                   PORTS                                                                                                                                                                                                                                                                                                       NAMES
    1b885f37c991   nvcr.io/nvidia/nemo-microservices/nv-ingest:25.03                                                      "/opt/conda/envs/nv_…"   3 minutes ago    Up 3 minutes (healthy)   0.0.0.0:7670-7671->7670-7671/tcp, :::7670-7671->7670-7671/tcp                                                                                                                                                                                                                                               nv-ingest-nv-ingest-ms-runtime-1
    62c6b999c413   zilliz/attu:v2.3.5                                                                                     "docker-entrypoint.s…"   13 minutes ago   Up 3 minutes             0.0.0.0:3001->3000/tcp, :::3001->3000/tcp                                                                                                                                                                                                                                                                   milvus-attu
    14ef31ed7f49   milvusdb/milvus:v2.5.3-gpu                                                                             "/tini -- milvus run…"   13 minutes ago   Up 3 minutes (healthy)   0.0.0.0:9091->9091/tcp, :::9091->9091/tcp, 0.0.0.0:19530->19530/tcp, :::19530->19530/tcp                                                                                                                                                                                                                    milvus-standalone
    dceaf36cc5df   otel/opentelemetry-collector-contrib:0.91.0                                                            "/otelcol-contrib --…"   13 minutes ago   Up 3 minutes             0.0.0.0:4317-4318->4317-4318/tcp, :::4317-4318->4317-4318/tcp, 0.0.0.0:8889->8889/tcp, :::8889->8889/tcp, 0.0.0.0:9988->9988/tcp, :::9988->9988/tcp, 0.0.0.0:13133->13133/tcp, :::13133->13133/tcp, 55678/tcp, 0.0.0.0:33249->9411/tcp, :::33247->9411/tcp, 0.0.0.0:55680->55679/tcp, :::55680->55679/tcp   nv-ingest-otel-collector-1
    fb252020e4d2   nvcr.io/nvidia/nim/nemoretriever-graphic-elements-v1:1.2.0-rc1-latest-datacenter-release-24734263   "/opt/nim/start_serv…"   13 minutes ago   Up 3 minutes             0.0.0.0:8003->8000/tcp, :::8003->8000/tcp, 0.0.0.0:8004->8001/tcp, :::8004->8001/tcp, 0.0.0.0:8005->8002/tcp, :::8005->8002/tcp                                                                                                                                                                             nv-ingest-graphic-elements-1
    c944a9d76831   nvcr.io/nvidia/nim/paddleocr:1.2.0-latest-datacenter-release-24685083                               "/opt/nim/start_serv…"   13 minutes ago   Up 3 minutes             0.0.0.0:8009->8000/tcp, :::8009->8000/tcp, 0.0.0.0:8010->8001/tcp, :::8010->8001/tcp, 0.0.0.0:8011->8002/tcp, :::8011->8002/tcp                                                                                                                                                                             nv-ingest-paddle-1
    5bea344526a2   nvcr.io/nvidia/nim/nemoretriever-page-elements-v2:1.2.0-rc0-latest-datacenter-release-24730057      "/opt/nim/start_serv…"   13 minutes ago   Up 3 minutes             0.0.0.0:8000-8002->8000-8002/tcp, :::8000-8002->8000-8002/tcp                                                                                                                                                                                                                                               nv-ingest-page-elements-1
    16dc2311a6cc   nvcr.io/nvidia/nim/llama-3.2-nv-embedqa-1b-v2:1.5.0-rc0-latest-datacenter-release-24738403          "/opt/nim/start_serv…"   13 minutes ago   Up 3 minutes             0.0.0.0:8012->8000/tcp, :::8012->8000/tcp, 0.0.0.0:8013->8001/tcp, :::8013->8001/tcp, 0.0.0.0:8014->8002/tcp, :::8014->8002/tcp                                                                                                                                                                             nv-ingest-embedding-1
    cea3ce001888   nvcr.io/nvidia/nim/nemoretriever-table-structure-v1:1.2.0-rc1-latest-datacenter-release-24826492    "/opt/nim/start_serv…"   13 minutes ago   Up 3 minutes             0.0.0.0:8006->8000/tcp, :::8006->8000/tcp, 0.0.0.0:8007->8001/tcp, :::8007->8001/tcp, 0.0.0.0:8008->8002/tcp, :::8008->8002/tcp                                                                                                                                                                             nv-ingest-table-structure-1
    7ddbf7690036   openzipkin/zipkin                                                                                      "start-zipkin"           13 minutes ago   Up 3 minutes (healthy)   9410/tcp, 0.0.0.0:9411->9411/tcp, :::9411->9411/tcp                                                                                                                                                                                                                                                         nv-ingest-zipkin-1
    b73bbe0c202d   minio/minio:RELEASE.2023-03-20T20-16-18Z                                                               "/usr/bin/docker-ent…"   13 minutes ago   Up 3 minutes (healthy)   0.0.0.0:9000-9001->9000-9001/tcp, :::9000-9001->9000-9001/tcp                                                                                                                                                                                                                                               minio
    97fa798dbe4f   prom/prometheus:latest                                                                                 "/bin/prometheus --w…"   13 minutes ago   Up 3 minutes             0.0.0.0:9090->9090/tcp, :::9090->9090/tcp                                                                                                                                                                                                                                                                   nv-ingest-prometheus-1
    f17cb556b086   grafana/grafana                                                                                        "/run.sh"                13 minutes ago   Up 3 minutes             0.0.0.0:3000->3000/tcp, :::3000->3000/tcp                                                                                                                                                                                                                                                                   grafana-service
    3403c5a0e7be   redis/redis-stack                                                                                      "/entrypoint.sh"         13 minutes ago   Up 3 minutes             0.0.0.0:6379->6379/tcp, :::6379->6379/tcp, 8001/tcp                                                                                                                                                                                                                                                         nv-ingest-redis-1
    ```


## Step 2: Install Python Dependencies

You can interact with the NV-Ingest service from the host, or by using `docker exec` to run commands in the NV-Ingest container.

To interact from the host, you'll need a Python environment and install the client dependencies:

```
# conda not required but makes it easy to create a fresh Python environment
conda create --name nv-ingest-dev python=3.10
conda activate nv-ingest-dev
pip install nv-ingest-client==2025.3.10.dev20250310
```

!!! tip

    To confirm that you have activated your Conda environment, run `which pip` and `which python`, and confirm that you see `nvingest` in the result. You can do this before any pip or python command that you run.


!!! note

    Interacting from the host depends on the appropriate port being exposed from the nv-ingest container to the host as defined in [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml#L141). If you prefer, you can disable exposing that port and interact with the NV-Ingest service directly from within its container. To interact within the container run `docker exec -it nv-ingest-nv-ingest-ms-runtime-1 bash`. You'll be in the `/workspace` directory with `DATASET_ROOT` from the .env file mounted at `./data`. The pre-activated `nv_ingest_runtime` conda environment has all the Python client libraries pre-installed and you should see `(morpheus) root@aba77e2a4bde:/workspace#`. From the bash prompt above, you can run the nv-ingest-cli and Python examples described following.


## Step 3: Ingesting Documents

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
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob
client = NvIngestClient(                                                                         
    message_client_port=7670,                                                               
    message_client_hostname="localhost"        
)                                                                 
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
    .vdb_upload(
        collection_name="test",
        sparse=False,
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
  --client_port=7670
```

You should notice output indicating document processing status followed by a breakdown of time spent during job execution:
```
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
[nltk_data] Downloading package punkt_tab to
[nltk_data]     /raid/jdyer/miniforge3/envs/nv-ingest-
[nltk_data]     dev/lib/python3.10/site-
[nltk_data]     packages/llama_index/core/_static/nltk_cache...
[nltk_data]   Package punkt_tab is already up-to-date!
INFO:nv_ingest_client.nv_ingest_cli:Processing 1 documents.
INFO:nv_ingest_client.nv_ingest_cli:Output will be written to: ./processed_docs
Processing files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.34s/file, pages_per_sec=1.28]
INFO:nv_ingest_client.cli.util.processing:message_broker_task_source: Avg: 2.39 ms, Median: 2.39 ms, Total Time: 2.39 ms, Total % of Trace Computation: 0.06%
INFO:nv_ingest_client.cli.util.processing:broker_source_network_in: Avg: 9.51 ms, Median: 9.51 ms, Total Time: 9.51 ms, Total % of Trace Computation: 0.25%
INFO:nv_ingest_client.cli.util.processing:job_counter: Avg: 1.47 ms, Median: 1.47 ms, Total Time: 1.47 ms, Total % of Trace Computation: 0.04%
INFO:nv_ingest_client.cli.util.processing:job_counter_channel_in: Avg: 0.46 ms, Median: 0.46 ms, Total Time: 0.46 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:metadata_injection: Avg: 3.52 ms, Median: 3.52 ms, Total Time: 3.52 ms, Total % of Trace Computation: 0.09%
INFO:nv_ingest_client.cli.util.processing:metadata_injection_channel_in: Avg: 0.16 ms, Median: 0.16 ms, Total Time: 0.16 ms, Total % of Trace Computation: 0.00%
INFO:nv_ingest_client.cli.util.processing:pdf_content_extractor: Avg: 475.64 ms, Median: 163.77 ms, Total Time: 2378.21 ms, Total % of Trace Computation: 62.73%
INFO:nv_ingest_client.cli.util.processing:pdf_content_extractor_channel_in: Avg: 0.31 ms, Median: 0.31 ms, Total Time: 0.31 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:image_content_extractor: Avg: 0.67 ms, Median: 0.67 ms, Total Time: 0.67 ms, Total % of Trace Computation: 0.02%
INFO:nv_ingest_client.cli.util.processing:image_content_extractor_channel_in: Avg: 0.21 ms, Median: 0.21 ms, Total Time: 0.21 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:docx_content_extractor: Avg: 0.46 ms, Median: 0.46 ms, Total Time: 0.46 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:docx_content_extractor_channel_in: Avg: 0.20 ms, Median: 0.20 ms, Total Time: 0.20 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:pptx_content_extractor: Avg: 0.68 ms, Median: 0.68 ms, Total Time: 0.68 ms, Total % of Trace Computation: 0.02%
INFO:nv_ingest_client.cli.util.processing:pptx_content_extractor_channel_in: Avg: 0.46 ms, Median: 0.46 ms, Total Time: 0.46 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:audio_data_extraction: Avg: 1.08 ms, Median: 1.08 ms, Total Time: 1.08 ms, Total % of Trace Computation: 0.03%
INFO:nv_ingest_client.cli.util.processing:audio_data_extraction_channel_in: Avg: 0.20 ms, Median: 0.20 ms, Total Time: 0.20 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:dedup_images: Avg: 0.42 ms, Median: 0.42 ms, Total Time: 0.42 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:dedup_images_channel_in: Avg: 0.42 ms, Median: 0.42 ms, Total Time: 0.42 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:filter_images: Avg: 0.59 ms, Median: 0.59 ms, Total Time: 0.59 ms, Total % of Trace Computation: 0.02%
INFO:nv_ingest_client.cli.util.processing:filter_images_channel_in: Avg: 0.57 ms, Median: 0.57 ms, Total Time: 0.57 ms, Total % of Trace Computation: 0.02%
INFO:nv_ingest_client.cli.util.processing:table_data_extraction: Avg: 240.75 ms, Median: 240.75 ms, Total Time: 481.49 ms, Total % of Trace Computation: 12.70%
INFO:nv_ingest_client.cli.util.processing:table_data_extraction_channel_in: Avg: 0.38 ms, Median: 0.38 ms, Total Time: 0.38 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:chart_data_extraction: Avg: 300.54 ms, Median: 299.94 ms, Total Time: 901.62 ms, Total % of Trace Computation: 23.78%
INFO:nv_ingest_client.cli.util.processing:chart_data_extraction_channel_in: Avg: 0.23 ms, Median: 0.23 ms, Total Time: 0.23 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:infographic_data_extraction: Avg: 0.77 ms, Median: 0.77 ms, Total Time: 0.77 ms, Total % of Trace Computation: 0.02%
INFO:nv_ingest_client.cli.util.processing:infographic_data_extraction_channel_in: Avg: 0.25 ms, Median: 0.25 ms, Total Time: 0.25 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:caption_ext: Avg: 0.55 ms, Median: 0.55 ms, Total Time: 0.55 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:caption_ext_channel_in: Avg: 0.51 ms, Median: 0.51 ms, Total Time: 0.51 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:embed_text: Avg: 1.21 ms, Median: 1.21 ms, Total Time: 1.21 ms, Total % of Trace Computation: 0.03%
INFO:nv_ingest_client.cli.util.processing:embed_text_channel_in: Avg: 0.21 ms, Median: 0.21 ms, Total Time: 0.21 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:store_embedding_minio: Avg: 0.32 ms, Median: 0.32 ms, Total Time: 0.32 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:store_embedding_minio_channel_in: Avg: 1.18 ms, Median: 1.18 ms, Total Time: 1.18 ms, Total % of Trace Computation: 0.03%
INFO:nv_ingest_client.cli.util.processing:message_broker_task_sink_channel_in: Avg: 0.42 ms, Median: 0.42 ms, Total Time: 0.42 ms, Total % of Trace Computation: 0.01%
INFO:nv_ingest_client.cli.util.processing:No unresolved time detected. Trace times account for the entire elapsed duration.
INFO:nv_ingest_client.cli.util.processing:Processed 1 files in 2.34 seconds.
INFO:nv_ingest_client.cli.util.processing:Total pages processed: 3
INFO:nv_ingest_client.cli.util.processing:Throughput (Pages/sec): 1.28
INFO:nv_ingest_client.cli.util.processing:Throughput (Files/sec): 0.43
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

The Nemo Retriever extraction core pipeline profiles run on a single A10G or better GPU. 
This includes text, table, chart, infographic extraction, embedding and indexing into Milvus. 
The advanced profiles require additional GPU support. 
This includes audio extraction and VLM integrations. 
For more information, refer to [Support Matrix](support-matrix.md).

The values that you specify in the `--profile` option of your `docker compose up` command are explained in the following table. 
You can specify multiple `--profile` options.


| Name                  | Type     | Description                                                       | GPU Requirements | Disk Space Requirements |
|-----------------------|----------|-------------------------------------------------------------------|-----------------------|-----------------------------------|
| `retrieval`           | Core     | Enables the embedding NIM and (GPU accelerated) Milvus. | [1 GPU](support-matrix.md) total for all core profiles. | ~150GB total for all core profiles. |
| `table-structure`     | Core     | Enables the yolox table structure NIM which enhances markdown formatting of extracted table content. This benefits answer generation by downstream LLMs. | [1 GPU](support-matrix.md) total for all core profiles. | ~150GB total for all core profiles. |
| `audio`               | Advanced | Use [Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html) for processing audio files. For more information, refer to [Audio Processing](nemoretriever-parse.md). | [1 additional dedicated GPU](support-matrix.md) | ~37GB additional space |
| `nemoretriever-parse` | Advanced | Use [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse). For more information, refer to [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md). | [1 additional dedicated GPU](support-matrix.md) | ~16 GB additional space |
| `vlm`                 | Advanced | Uses llama 3.2 11B VLM for experimental image captioning of unstructured images. | [1 additional dedicated GPU](support-matrix.md) | ~16GB additional space |



## Troubleshooting

In rare cases, when you run a job you might an see an error similar to `max open file descriptor`. 
This error is related to number of active jobs and cores available on your computer. 
To resolve this issue, raise the maximum number of open file descriptors by using the following [ulimit](https://ss64.com/bash/ulimit.html) command.

```bash
ulimit -n 10,000
```



## Related Topics

- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Quickstart (Library Mode)](quickstart-library-mode.md)
- [Notebooks](notebooks.md)
- [Multimodal PDF Data Extraction](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
