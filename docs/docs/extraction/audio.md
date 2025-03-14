## Use NeMo Retriever Extraction with Riva

This documentation describes two methods to run [NeMo Retriever extraction](overview.md) 
with [Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html) for processing audio files.

- Run the NIM locally by using Docker Compose
- Use NVIDIA Cloud Functions (NVCF) endpoints for cloud-based inference

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.



## Run the NIM Locally by Using Docker Compose

Use the following procedure to run the NIM locally.

!!! important

    Due to limitations in available VRAM controls in the current release of audio NIMs, it must run on a [dedicated additional GPU](support-matrix.md). Edit docker-compose.yaml to set the audio service's device_id to a dedicated GPU: device_ids: ["1"] or higher.


1. To access the required container images, log in to the NVIDIA Container Registry (nvcr.io). Use [your NGC key](ngc-api-key.md) as the password. Run the following command in your terminal.

    - Replace `<your-ngc-key>` with your actual NGC API key.
    - The username is always `$oauthtoken`.

    ```shell
    $ docker login nvcr.io
    Username: $oauthtoken
    Password: <your-ngc-key>
    ```

2. Store [your NGC key](ngc-api-key.md) in an environment variable file.

For convenience and security, store your NGC key in a .env file.
This enables services to access it without needing to enter the key manually each time.

Create a .env file in your working directory and add the following line:
```ini
NGC_API_KEY=<your-ngc-key>
```
Again, replace <your-ngc-key> with your actual NGC key.

3. Start the nv-ingest services with the `audio` profile. This profile includes the necessary components for audio processing. Use the following command.

    - The `--profile audio` flag ensures that audio-specific services are launched. For more information, refer to [Profile Information](quickstart-guide.md#profile-information).
    - The `--build` flag ensures that any changes to the container images are applied before starting.

    ```shell
    docker compose --profile retrieval --profile audio up --build
    ```

4. After the services are running, you can interact with nv-ingest by using Python.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells nv-ingest to extract information from WAV audio files.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.wav")
        .extract(
            document_type="wav",  # Ingestor should detect type automatically in most cases
            extract_method="audio",
        )
    )
    ```


    !!! tip

        For more Python examples, refer to [NV-Ingest: Python Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/python_client_usage.ipynb).


## Use NVCF Endpoints for Cloud-Based Inference

Instead of running NV-Ingest locally, you can use NVCF to perform inference by using remote endpoints.

1. NVCF requires an authentication token and a function ID for access. Ensure you have these credentials ready before making API calls.

2. Run inference by using Python. Provide an NVCF endpoint along with authentication details.

    - The `Ingestor` object initializes the ingestion process.
    - The `files` method specifies the input files to process.
    - The `extract` method tells nv-ingest to extract information from WAV audio files.
    - The `document_type` parameter is optional, because `Ingestor` should detect the file type automatically.

    ```python
    ingestor = (
        Ingestor()
        .files("./data/*.mp3")
        .extract(
            document_type="mp3",
            extract_method="audio",
            extract_audio_params={
                "grpc_endpoint": "grpc.nvcf.nvidia.com:443",
                "auth_token": "<API key>",
                "function_id": "<function ID>",
                "use_ssl": True,
            },
        )
    )
    ```

    !!! tip

        For more Python examples, refer to [NV-Ingest: Python Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/python_client_usage.ipynb).
