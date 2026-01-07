# Environment Variables for NeMo Retriever Extraction

The following are the environment variables that you can use to configure [NeMo Retriever extraction](overview.md).
You can specify these in your .env file or directly in your environment.

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.


## General Environment Variables

| Name                             | Example                        | Description                                                           |
|----------------------------------|--------------------------------|-----------------------------------------------------------------------|
| `DOWNLOAD_LLAMA_TOKENIZER`       | `True` <br/>                                             | If `True`, the [llama-3.2 tokenizer](https://huggingface.co/meta-llama/Llama-3.2-1B) will be pre-dowloaded at build time. If not set to `True`, the (e5-large-unsupervised)[https://huggingface.co/intfloat/e5-large-unsupervised] tokenizer will be pre-downloaded. Note: setting this to `True` requires a HuggingFace access token with access to the gated Llama-3.2 models. See below for more info. |
| `HF_ACCESS_TOKEN`                | -                                                         | The HuggingFace access token used to pre-downlaod the Llama-3.2 tokenizer from HuggingFace (see above for more info). Llama 3.2 is a gated model, so you must [request access](https://huggingface.co/meta-llama/Llama-3.2-1B) to the Llama-3.2 models and then set this variable to a token that can access gated repositories on your behalf in order to use `DOWNLOAD_LLAMA_TOKENIZER=True`. |
| `INGEST_LOG_LEVEL`               | - `DEBUG` <br/> - `INFO` <br/> - `WARNING` <br/> - `ERROR` <br/> - `CRITICAL` <br/> | The log level for the ingest service, which controls the verbosity of the logging output. |
| `MESSAGE_CLIENT_HOST`            | - `redis` <br/> - `localhost` <br/> - `192.168.1.10` <br/> | Specifies the hostname or IP address of the message broker used for communication between services. |
| `MESSAGE_CLIENT_PORT`            | - `7670` <br/> - `6379` <br/>                              | Specifies the port number on which the message broker is listening. |
| `MINIO_BUCKET`                   | `nv-ingest` <br/>                                        | Name of MinIO bucket, used to store image, table, and chart extractions. |
| `NGC_API_KEY`                    | `nvapi-*************` <br/>                              | An authorized NGC API key, used to interact with hosted NIMs. To create an NGC key, go to [https://org.ngc.nvidia.com/setup/api-keys](https://org.ngc.nvidia.com/setup/api-keys). |
| `NIM_NGC_API_KEY`                | —                                                          | The key that NIM microservices inside docker containers use to access NGC resources. This is necessary only in some cases when it is different from `NGC_API_KEY`. If this is not specified, `NGC_API_KEY` is used to access NGC resources. |
| `OTEL_EXPORTER_OTLP_ENDPOINT`    | `http://otel-collector:4317` <br/>                       | The endpoint for the OpenTelemetry exporter, used for sending telemetry data. |
| `REDIS_INGEST_TASK_QUEUE`        | `ingest_task_queue` <br/>                              | The name of the task queue in Redis where tasks are stored and processed. |
| `IMAGE_STORAGE_URI`              | `s3://nv-ingest/artifacts/store/images` <br/>          | Default fsspec-compatible URI for the `store` task. Supports `s3://`, `file://`, `gs://`, etc. See [Store Extracted Images](nv-ingest-python-api.md#store-extracted-images). |
| `IMAGE_STORAGE_PUBLIC_BASE_URL`  | `https://assets.example.com/images` <br/>              | Optional HTTP(S) base URL for serving stored images. |


## Library Mode Environment Variables

These environment variables apply specifically when running NV-Ingest in library mode.

| Name                              | Example                                                 | Description |
|-----------------------------------|---------------------------------------------------------|-------------|
| `NVIDIA_API_KEY`                  | `nvapi-*************` <br/>                             | API key for NVIDIA-hosted NIM services. |



## Related Topics

- [Configure Ray Logging](https://docs.nvidia.com/nemo/retriever/latest/extraction/ray-logging/)
