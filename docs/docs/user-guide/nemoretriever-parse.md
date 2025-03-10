## How to Use the NV-Ingest Service with nemoretriever-parse

This guide outlines two methods for running NV-Ingest with nemoretriever-parse:

1. **Running the NIM locally using Docker Compose**
2. **Using NVCF Endpoints for Cloud-Based Inference**

---

## 1. Running the NIM Locally Using Docker Compose

### Step 1. Start the Services with the nemoretriever-parse Profile

When starting the NV-Ingest services, ensure that you enable the `nemoretriever-parse` profile. This profile includes the necessary components for extracting text and metadata from images.

Use the following command:

```shell
docker compose --profile nemoretriever-parse up --build
```

- The --profile nemoretriever-parse flag ensures that vision-language retrieval services are launched.
- The --build flag ensures that any changes to the container images are applied before starting.

### Step 2. Using the NV-Ingest API

Once the services are running, you can interact with NV-Ingest using the Python API.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract(
        document_type="pdf",  # Optional, Ingestor should detect automatically in most cases
        extract_method="nemoretriever_parse"
    )
)
```
- The `Ingestor()` object initializes the ingestion process.
- The `.files("./data/*.pdf")` method specifies the input files to process.
- The `.extract(document_type="image", extract_method="nemoretriever_parse")` method tells NV-Ingest to use nemoretriever-parse for extracting text and metadata from images.
- Note: The `document_type` parameter is optional, as Ingestor should detect the file type automatically.

## 2. Using NVCF Endpoints for Cloud-Based Inference

Instead of running NV-Ingest locally, you can use NVIDIA Cloud Functions (NVCF) to perform inference via remote endpoints.

### Step 1: Configure Authentication

Set the authentication token in the `.env` file:

```
NVIDIA_BUILD_API_KEY=nvapi-...
```

### Step 2: Configure the Endpoint in `docker-compose.yaml`

Modify `docker-compose.yaml` to use the hosted `nemoretriever-parse` service:

```yaml
  # build.nvidia.com hosted nemoretriever-parse
  - NEMORETRIEVER_PARSE_HTTP_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
  #- NEMORETRIEVER_PARSE_HTTP_ENDPOINT=http://nemoretriever-parse:8000/v1/chat/completions
```

### Step 3: Running Inference Using NVCF Endpoints

Once authentication and endpoint configuration are set, you can run inference via the Python API:

```python
ingestor = ( 
    Ingestor()
    .files("./data/*.pdf")
    .extract(
        document_type="pdf",
        extract_method="nemoretriever_parse"
    )
)
```
