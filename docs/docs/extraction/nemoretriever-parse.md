## Use Nemo Retriever Extraction with nemoretriever-parse

This documentation describes two methods to run NeMo Retriever Extraction 
with [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse).

- Run the NIM locally by using Docker Compose
- Use NVIDIA Cloud Functions (NVCF) endpoints for cloud-based inference

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.



## Run the NIM Locally by Using Docker Compose

Use the following procedure to run the NIM locally.

1. Start the nv-ingest services with the `nemoretriever-parse` profile. This profile includes the necessary components for extracting text and metadata from images. Use the following command.

    - The --profile nemoretriever-parse flag ensures that vision-language retrieval services are launched.
    - The --build flag ensures that any changes to the container images are applied before starting.

    ```shell
    docker compose --profile nemoretriever-parse up --build
    ```

2. After the services are running, you can interact with nv-ingest by using Python.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells nv-ingest to use `nemoretriever-parse` for extracting text and metadata from images.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.pdf")
        .extract(
            document_type="pdf",  # Ingestor should detect type automatically in most cases
            extract_method="nemoretriever_parse"
        )
    )
    ```



## Using NVCF Endpoints for Cloud-Based Inference

Instead of running NV-Ingest locally, you can use NVCF to perform inference by using remote endpoints.

1. Set the authentication token in the `.env` file.

    ```
    NVIDIA_BUILD_API_KEY=nvapi-...
    ```

2. Modify `docker-compose.yaml` to use the hosted `nemoretriever-parse` service.

    ```yaml
    # build.nvidia.com hosted nemoretriever-parse
    - NEMORETRIEVER_PARSE_HTTP_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
    #- NEMORETRIEVER_PARSE_HTTP_ENDPOINT=http://nemoretriever-parse:8000/v1/chat/completions
    ```

3. Run inference by using Python.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells nv-ingest to use `nemoretriever-parse` for extracting text and metadata from images.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.pdf")
        .extract(
            document_type="pdf",  # Ingestor should detect type automatically in most cases
            extract_method="nemoretriever_parse"
        )
    )
    ```
