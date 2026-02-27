# Advanced Visual Parsing in NeMo Retriever Extraction

For scanned documents, or documents with complex layouts, 
we recommend that you use [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse). 
Nemotron parse provides higher-accuracy text extraction. 

This documentation describes the following two methods 
to run [NeMo Retriever Library](overview.md) with nemotron-parse.

- Run the NIM locally by using Docker Compose
- Use NVIDIA Cloud Functions (NVCF) endpoints for cloud-based inference

!!! note

    This library is the NeMo Retriever Library.


## Limitations

Currently, the limitations to using `nemotron-parse` with NeMo Retriever Extraction are the following:

- Extraction with `nemotron-parse` only supports PDFs, not image files. For more information, refer to [Troubleshoot NeMo Retriever Library](troubleshoot.md).
- `nemotron-parse` is not supported on RTX Pro 6000, B200, or H200 NVL. For more information, refer to the [Nemotron Parse Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-parse).


## Run the NIM Locally by Using Docker Compose

Use the following procedure to run the NIM locally.

!!! important

    Due to limitations in available VRAM controls in the current release of nemotron-parse, it must run on a [dedicated additional GPU](support-matrix.md). Edit docker-compose.yaml to set nemotron-parse's device_id to a dedicated GPU: device_ids: ["1"] or higher.


1. Start the NeMo Retriever services with the `nemotron-parse` profile. This profile includes the necessary components for extracting text and metadata from images. Use the following command.

    - The --profile nemotron-parse flag ensures that vision-language retrieval services are launched.  For more information, refer to [Profile Information](quickstart-guide.md#profile-information).

    ```shell
    docker compose --profile nemotron-parse up
    ```

2. After the services are running, you can interact with NeMo Retriever by using Python.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells NeMo Retriever to use `nemotron-parse` for extracting text and metadata from images.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.pdf")
        .extract(
            document_type="pdf",  # Ingestor should detect type automatically in most cases
            extract_method="nemotron_parse"
        )
    )
    ```

    !!! tip

        For more Python examples, refer to [NeMo Retriever: Python Client Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/python_client_usage.ipynb).


## Using NVCF Endpoints for Cloud-Based Inference

Instead of running NeMo Retriever locally, you can use NVCF to perform inference by using remote endpoints.

1. Set the authentication token in the `.env` file.

    ```
    NVIDIA_API_KEY=nvapi-...
    ```

2. Modify `docker-compose.yaml` to use the hosted `nemotron-parse` service.

    ```yaml
    # build.nvidia.com hosted nemotron-parse
    - NEMOTRON_PARSE_HTTP_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
    #- NEMOTRON_PARSE_HTTP_ENDPOINT=http://nemotron-parse:8000/v1/chat/completions
    ```

3. Run inference by using Python.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells NeMo Retriever to use `nemotron-parse` for extracting text and metadata from images.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.pdf")
        .extract(
            document_type="pdf",  # Ingestor should detect type automatically in most cases
            extract_method="nemotron_parse"
        )
    )
    ```

    !!! tip

        For more Python examples, refer to [NeMo Retriever: Python Client Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/python_client_usage.ipynb).



## Related Topics

- [Support Matrix](support-matrix.md)
- [Troubleshoot NeMo Retriever Library](troubleshoot.md)
- [Use the NeMo Retriever Python API](python-api-reference.md)
