## How to Use the NV-Ingest Service with Riva

This guide outlines two methods for running NV-Ingest with Riva for processing audio files:

- Run the NIM locally using Docker Compose
- Use NVCF Endpoints for Cloud-Based Inference


## Run the NIM Locally Using Docker Compose

1. Log in to nvcr.io.

To access the required container images, log in to the NVIDIA Container Registry (nvcr.io). Use your NGC API key as the password.

Run the following command in your terminal:
```shell
$ docker login nvcr.io
Username: $oauthtoken
Password: <your-ngc-key>
```
- Replace `<your-ngc-key>` with your actual NGC API key.
- The username should always be `$oauthtoken`.
- Ensure that your NGC key belongs to the `nvidia/riva` org/team.

2. Store the NGC API Key in an Environment File

For convenience and security, store your NGC API key in a .env file.
This allows services to access it without needing to enter the key manually each time.

Create a .env file in your working directory and add the following line:
```ini
RIVA_NGC_API_KEY=<your-ngc-key>
```
Again, replace <your-ngc-key> with your actual API key.

3. Start the Services with the audio Profile

When starting the NV-Ingest services, ensure that you enable the audio profile. This profile includes the necessary components for audio processing.

Use the following command:
```shell
docker compose --profile audio up --build
```
- The `--profile audio` flag ensures that audio-specific services are launched.
- The `--build` flag ensures that any changes to the container images are applied before starting.

4. Using the NV-Ingest API

After the services are running, you can interact with NV-Ingest using the Python API.

```python
ingestor = (
    Ingestor()
    .files("./data/*.wav")
    .extract(
        document_type="wav"  # Optional, Ingestor should detect automatically in most cases
    )
)
```
- The `Ingestor()` object initializes the ingestion process.
- The `.files("./data/*.wav")` method specifies the input files to process.
- The `.extract(document_type="wav")` method tells NV-Ingest to extract information from WAV audio files.
- Note: The `document_type` parameter is optional, as Ingestor should detect the file type automatically.



## Use NVCF Endpoints for Cloud-Based Inference

Instead of running NV-Ingest locally, you can use NVIDIA Cloud Functions (NVCF) to perform inference via remote endpoints.

1. Authenticate with NVCF

NVCF requires an authentication token and a function ID for access. Ensure you have these credentials ready before making API calls.

2. Running Inference Using NVCF Endpoints

Instead of running NV-Ingest locally, you provide an NVCF endpoint along with authentication details.

```python
ingestor = (
    Ingestor()
    .files("./data/*.mp3")
    .extract(
        document_type="mp3",
        extract_audio_params={
            "grpc_endpoint": "grpc.nvcf.nvidia.com:443",
            "auth_token": "<API key>",
            "auth_metadata": [("function-id", "<function ID>")],
            "use_ssl": True,
        },
    )
)
```
