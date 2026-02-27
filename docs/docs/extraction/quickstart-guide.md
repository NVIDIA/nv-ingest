# Deploy With Docker Compose (Self-Hosted) for NeMo Retriever Extraction

This guide helps you get started using [NeMo Retriever extraction](overview.md) in self-hosted mode.


## Step 1: Start Containers

Use the provided [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml) to start all needed services with a few commands.

!!! warning

    NIM containers on their first startup can take 10-15 minutes to pull and fully load models.


If you prefer, you can run on Kubernetes by using [our Helm chart](https://github.com/NVIDIA/nv-ingest/blob/main/helm/README.md). Also, there are [additional environment variables](environment-config.md) you can configure.

a. Git clone the repo:

    `git clone https://github.com/nvidia/nv-ingest`

b. Change the directory to the cloned repo by running the following code.
   
    `cd nv-ingest`.

c. [Generate API keys](ngc-api-key.md) and authenticate with NGC with the `docker login` command.

    ```shell
    # This is required to access pre-built containers and NIM microservices
    $ docker login nvcr.io
    Username: $oauthtoken
    Password: <Your Key>
    ```
   
d. Create a .env file that contains your NVIDIA Build API key.

    !!! note

        If you use an NGC personal key, then you should provide the same value for all keys, but you must specify each environment variable individually. In the past, you could create an API key. If you have an API key, you can still use that. For more information, refer to [Generate Your NGC Keys](ngc-api-key.md) and [Environment Configuration Variables](environment-config.md).

    ```
    # Container images must access resources from NGC.

    NGC_API_KEY=<key to download containers from NGC>
    NIM_NGC_API_KEY=<key to download model files after containers start>
    ```
   
e. Make sure that NVIDIA is set as your default container runtime before you run the docker compose command by running the following code.

    `sudo nvidia-ctk runtime configure --runtime=docker --set-as-default`

f. Start core services. This example uses the retrieval profile.  For more information about other profiles, see [Profile Information](#profile-information).

    `docker compose --profile retrieval up`

    !!! tip

        By default, we have [configured log levels to be verbose](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml). It's possible to observe service startup proceeding. You will notice a lot of log messages. Disable verbose logging by configuring `NIM_TRITON_LOG_VERBOSE=0` for each NIM in [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml).

    !!! tip

        The default configuration might not fit on a single GPU for some hardware targets. Use a [docker compose override file](#docker-compose-override-files) to reduce VRAM usage. Override files typically lower per-service memory allocation, batch sizes, or concurrency, trading peak throughput for making the full pipeline runnable on the available GPU.

g. When core services have fully started, `nvidia-smi` should show processes like the following:

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

h. Run the command `docker ps`. You should see output similar to the following. Confirm that the status of the containers is `Up`.

    ```
    CONTAINER ID  IMAGE                                            COMMAND                 CREATED         STATUS                  PORTS            NAMES
    1b885f37c991  nvcr.io/nvidia/nemo-microservices/nv-ingest:...  "/usr/bin/tini -- /w…"  7 minutes ago   Up 7 minutes (healthy)  0.0.0.0:7670...  nv-ingest-nv-ingest-ms-runtime-1
    14ef31ed7f49  milvusdb/milvus:v2.5.3-gpu                       "/tini -- bash -c 's…"  7 minutes ago   Up 7 minutes (healthy)  0.0.0.0:9091...  milvus-standalone
    dceaf36cc5df  otel/opentelemetry-collector-contrib:...         "/otelcol-contrib --…"  7 minutes ago   Up 7 minutes            0.0.0.0:4317...  nv-ingest-otel-collector-1
    5bd0b48eb71b  nvcr.io/nim/nvidia/nemoretriever-graphic-ele...  "/opt/nvidia/nvidia_…"  7 minutes ago   Up 7 minutes            0.0.0.0:8003...  nv-ingest-graphic-elements-1
    daf878669036  nvcr.io/nim/nvidia/nemoretriever-ocr-v1:1.2.1    "/opt/nvidia/nvidia_…"  7 minutes ago   Up 7 minutes            0.0.0.0:8009...  nv-ingest-ocr-1
    216bdf11c566  nvcr.io/nim/nvidia/nemoretriever-page-elements-v3:1.7.0  "/opt/nvidia/nvidia_…"  7 minutes ago   Up 7 minutes            0.0.0.0:8000...  nv-ingest-page-elements-1
    aee9580b0b9a  nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.10.0  "/opt/nvidia/nvidia_…"  7 minutes ago   Up 7 minutes            0.0.0.0:8012...  nv-ingest-embedding-1
    178a92bf6f7f  nvcr.io/nim/nvidia/nemoretriever-table-struc...  "/opt/nvidia/nvidia_…"  7 minutes ago   Up 7 minutes            0.0.0.0:8006...  nv-ingest-table-structure-1
    7ddbf7690036  openzipkin/zipkin                                "start-zipkin"          7 minutes ago   Up 7 minutes (healthy)  9410/tcp...      nv-ingest-zipkin-1
    b73bbe0c202d  minio/minio:RELEASE.2023-03-20T20-16-18Z         "/usr/bin/docker-ent…"  7 minutes ago   Up 7 minutes (healthy)  0.0.0.0:9000...  minio
    97fa798dbe4f  prom/prometheus:latest                           "/bin/prometheus --w…"  7 minutes ago   Up 7 minutes            0.0.0.0:9090...  nv-ingest-prometheus-1
    f17cb556b086  grafana/grafana                                  "/run.sh"               7 minutes ago   Up 7 minutes            0.0.0.0:3000...  grafana-service
    3403c5a0e7be  redis/redis-stack                                "/entrypoint.sh"        7 minutes ago   Up 7 minutes            0.0.0.0:6379...  nv-ingest-redis-1
    ```


## Step 2: Install Python Dependencies

You can interact with the NV-Ingest service from the host, or by using `docker exec` to run commands in the NV-Ingest container.

To interact from the host, you'll need a Python environment that has the client dependencies installed.

```
uv venv --python 3.12 nv-ingest-dev
source nv-ingest-dev/bin/activate
uv pip install nv-ingest==26.1.2 nv-ingest-api==26.1.2 nv-ingest-client==26.1.2
```

!!! tip

    To confirm that you have activated your virtual environment, run `which pip` and `which python`, and confirm that you see `nvingest` in the result. You can do this before any pip or python command that you run.


!!! note

Interaction from the host requires the appropriate port to be exposed from the `nv-ingest` container, as defined in the `docker-compose.yaml` file. If you prefer, you can disable this port and interact directly with the NV-Ingest service from within its container.

To work inside the container, run the following code.

```bash
docker exec -it nv-ingest-nv-ingest-ms-runtime-1 bash
```
This command opens a shell in the `/workspace` directory, where the `DATASET_ROOT` from your `.env` file is mounted at `./data`. The pre-created `nv_ingest_runtime` virtual environment includes all necessary Python client libraries. You should see a prompt similar to the following.

```bash
(nv_ingest_runtime) root@your-computer-name:/workspace#
```
From this prompt, you can run the `nv-ingest` CLI and Python examples.

Because many service URIs default to localhost, running inside the `nv-ingest` container also requires that you specify URIs manually so that services can communicate across containers on the internal Docker network. See the example following for how to set the `milvus_uri`.

## Step 3: Ingest Documents

You can submit jobs programmatically in Python or using the [NV-Ingest CLI](nv-ingest_cli.md).

The following examples demonstrate how to extract text, charts, tables, and images:

- **extract_text** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to find and extract text from pages.
- **extract_images** — Uses [PDFium](https://github.com/pypdfium2-team/pypdfium2/) to extract images.
- **extract_tables** — Uses [object detection family of NIMs](https://docs.nvidia.com/nim/ingestion/object-detection/latest/overview.html) to find tables and charts, and NemoRetriever OCR for table extraction.
- **extract_charts** — Enables or disables chart extraction, also based on the object detection NIM family.


### In Python

!!! tip

    For more Python examples, refer to [NV-Ingest: Python Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/python_client_usage.ipynb).

<a id="ingest_python_example"></a>
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
        table_output_format="markdown",
        extract_infographics=True,
        # extract_method="nemotron_parse", # Slower, but maximally accurate, especially for PDFs with pages that are scanned images
        text_depth="page"
    ).embed()
    .vdb_upload(
        collection_name="test",
        sparse=False,
        # for llama-3.2 embedder, use 1024 for e5-v5
        dense_dim=2048,
        # milvus_uri="http://milvus:19530"  # When running from within a container, the URI to the Milvus service is specified using the internal Docker network.
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
print(f"Total time: {t1-t0} seconds")

# results blob is directly inspectable
print(ingest_json_results_to_blob(results[0]))

if failures:
    print(f"There were {len(failures)} failures. Sample: {failures[0]}")
```

!!! note

    For advanced visual parsing in self-hosted mode, uncomment `extract_method="nemotron_parse"` in the previous code. For more information, refer to [Advanced Visual Parsing](nemoretriever-parse.md).


The output looks similar to the following.

```
Starting ingestion..
1 records to insert to milvus
logged 8 records
Total time: 5.479151725769043 seconds
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

<a id="ingest_cli_example"></a>
```shell
nv-ingest-cli \
  --doc ./data/multimodal_test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_tables": "true", "extract_images": "true", "extract_charts": "true"}' \
  --client_host=localhost \
  --client_port=7670
```

You should see output that indicates the document processing status followed by a breakdown of time spent during job execution.

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

    Beyond inspecting the results, you can read them into things like [llama-index](https://github.com/NVIDIA/nv-ingest/blob/main/examples/llama_index_multimodal_rag.ipynb) or [langchain](https://github.com/NVIDIA/nv-ingest/blob/main/examples/langchain_multimodal_rag.ipynb) retrieval pipelines. Also, checkout our [Enterprise RAG Blueprint on build.nvidia.com](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag) to query over document content pre-extracted with NV-Ingest.



## Profile Information

The values that you specify in the `--profile` option of your `docker compose up` command are explained in the following table. 
You can specify multiple `--profile` options.

| Profile               | Type     | Description                                                       | 
|-----------------------|----------|-------------------------------------------------------------------| 
| `retrieval`           | Core     | Enables the embedding NIM and (GPU accelerated) Milvus.           | 
| `audio`               | Advanced | Use [Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html) for processing audio files. For more information, refer to [Audio Processing](audio.md). | 
| `nemotron-parse`      | Advanced | Use [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse), which adds state-of-the-art text and table extraction. For more information, refer to [Advanced Visual Parsing](nemoretriever-parse.md). | 
| `vlm`                 | Advanced | Use [llama 3.1 Nemotron 8B Vision](https://build.nvidia.com/nvidia/llama-3.1-nemotron-nano-vl-8b-v1/modelcard) for image captioning of unstructured images and infographics. This profile enables the `caption` method in the Python API to generate text descriptions of visual content. For more information, refer to [Use Multimodal Embedding](vlm-embed.md) and [Extract Captions from Images](nv-ingest-python-api.md#extract-captions-from-images). | 


## Docker Compose override files

The default [docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml) might exceed VRAM on a single GPU for some hardware. Override files reduce per-service memory, batch sizes, or concurrency so the full pipeline can run on the available GPU. To use an override, pass a second `-f` file after the base compose file; Docker Compose merges them and the override takes precedence.

| Override file | GPU target |
|---------------|------------|
| `docker-compose.a10g.yaml` | NVIDIA A10G |
| `docker-compose.a100-40gb.yaml` | NVIDIA A100-SXM4-40GB |
| `docker-compose.l40s.yaml` | NVIDIA L40S |

For RTX Pro 6000 Server Edition and other GPUs with limited VRAM, use the override that best matches your GPU memory (for example, `docker-compose.l40s.yaml` or `docker-compose.a10g.yaml`).

### Example: Using the VLM Profile for Infographic Captioning

Infographics often combine text, charts, and diagrams into complex visuals. Vision-language model (VLM) captioning generates natural language descriptions that capture this complexity, making the content searchable and more accessible for downstream applications.

To use VLM captioning for infographics, start NeMo Retriever extraction with both the `retrieval` and `vlm` profiles by running the following code.
```shell
docker compose \
  -f docker-compose.yaml \
  --profile retrieval \
  --profile vlm up
```

### Example with A100 40GB

The following example uses an override file for an A100 40GB GPU.

```shell
docker compose \
  -f docker-compose.yaml \
  -f docker-compose.a100-40gb.yaml \
  --profile retrieval up
```

### Example with A10G

```shell
docker compose \
  -f docker-compose.yaml \
  -f docker-compose.a10g.yaml \
  --profile retrieval up
```

### Example with L40S

```shell
docker compose \
  -f docker-compose.yaml \
  -f docker-compose.l40s.yaml \
  --profile retrieval up
```


## Specify MIG slices for NIM models

When you deploy NV-Ingest with NIM models on MIG‑enabled GPUs, MIG device slices are requested and scheduled through the `values.yaml` file for the corresponding NIM microservice. For IBM Content-Aware Storage (CAS) deployments, this allows NV-Ingest NIM pods to land only on nodes that expose the desired MIG profiles [raw.githubusercontent](https://raw.githubusercontent.com/NVIDIA/nv-ingest/main/helm/README.md%E2%80%8B).​

To target a specific MIG profile—for example, a 3g.20gb slice on an A100, which is a hardware-partitioned virtual GPU instance that gives your workload a fixed mid-sized share of the A100’s compute plus 20 GB of dedicated GPU memory and behaves like a smaller independent GPU—for a given NIM, configure the `resources` and `nodeSelector` under that NIM’s values path in `values.yaml`.

The following example shows the pattern. Paths vary by NIM, such as `nvingest.nvidiaNim.nemoretrieverPageElements` instead of the generic `nvingest.nim` placeholder. For details refer to [catalog.ngc.nvidia](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo-microservices/helm-charts/nv-ingest)​.
Set `resources.requests` and `resources.limits` to the name of the MIG resource that you want (for example, `nvidia.com/mig-3g.20gb`).
```shell
nvingest:
  nvidiaNim:
    nemoretrieverPageElements:
      modelName: "meta/llama3-8b-instruct"        # Example NIM model
      resources:
        limits:
          nvidia.com/mig-3g.20gb: 1               # MIG profile resource
        requests:
          nvidia.com/mig-3g.20gb: 1
      nodeSelector:
        nvidia.com/gpu.product: A100-SXM4-40GB-MIG-3g.20gb
```
Key points:
* Use the appropriate NIM‑specific values path (for example, `nvingest.nvidiaNim.nemoretrieverPageElements.resources`) rather than the generic `nvingest.nim` placeholder.
* Set `resources.requests` and `resources.limits` to the desired MIG resource name (for example, `nvidia.com/mig-3g.20gb`).
* Use `nodeSelector` (or tolerations/affinity, if you prefer) to target nodes labeled with the corresponding MIG‑enabled GPU product (for example, `nvidia.com/gpu.product: A100-SXM4-40GB-MIG-3g.20gb`).
This syntax and structure can be repeated for each NIM model used by CAS, ensuring that each NV-Ingest NIM pod is mapped to the correct MIG slice type and scheduled onto compatible nodes.

!!! important

    Advanced features require additional GPU support and disk space. For more information, refer to [Support Matrix](support-matrix.md).

## Related Topics

- [Troubleshoot](troubleshoot.md)
- [Prerequisites](prerequisites.md)
- [Support Matrix](support-matrix.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Helm](helm.md)
- [Notebooks](notebooks.md)
- [Enterprise RAG Blueprint](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
