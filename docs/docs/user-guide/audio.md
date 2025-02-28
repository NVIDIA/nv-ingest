## How to Use the NV-Ingest Service with Riva

This guide outlines the steps required to set up and use NV-Ingest with Riva for processing audio files.

## 1. Log in to nvcr.io.

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

## 2. Store the NGC API Key in an Environment File

For convenience and security, store your NGC API key in a .env file.
This allows services to access it without needing to enter the key manually each time.

Create a .env file in your working directory and add the following line:
```ini
RIVA_NGC_API_KEY=<your-ngc-key>
```
Again, replace <your-ngc-key> with your actual API key.

## 3. Start the Services with the audio Profile

When starting the NV-Ingest services, ensure that you enable the audio profile. This profile includes the necessary components for audio processing.

Use the following command:
```shell
docker compose --profile audio up --build
```
- The `--profile audio` flag ensures that audio-specific services are launched.
- The `--build` flag ensures that any changes to the container images are applied before starting.

## 4. Using the NV-Ingest API

Once the services are running, you can interact with NV-Ingest using either the Python API or the Command-Line Interface (CLI).

### Python API Usage

You can use the Python API to process `.wav` audio files as follows:
```python
ingestor = (
    Ingestor()
    .files("./data/*.wav")
    .extract(
        document_type="wav"
    )
)
```
- The `Ingestor()` object initializes the ingestion process.
- The `.files("./data/*.wav")` method specifies the input files to process.
- The `.extract(document_type="wav")` method tells NV-Ingest to extract information from WAV audio files.
- Note: The `document_type` parameter is optional, as Ingestor should detect the file type automatically.

### Command-Line Interface (CLI) Usage

If you prefer the CLI, you can achieve the same functionality using the following command:
```shell
nv-ingest-cli \
  --task 'extract:{"document_type":"wav"}' \
  --doc "./data/*.wav"
```
- `--task 'extract:{"document_type":"wav"}'` specifies the extraction task for WAV files.
- `--doc "./data/*.wav"` points to the directory containing the audio files to be processed.
