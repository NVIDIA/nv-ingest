# Environment Configuration Variables for NV-Ingest

The following are the environment configuration variables that you can specify in your .env file.

## General Environment Variables

| Name                             | Example                        | Description                                                           |
|----------------------------------|--------------------------------|-----------------------------------------------------------------------|
| `INGEST_LOG_LEVEL`               | - `DEBUG` <br/> - `INFO` <br/> - `WARNING` <br/> - `ERROR` <br/> - `CRITICAL` <br/> | The log level for the ingest service, which controls the verbosity of the logging output. |
| `MESSAGE_CLIENT_HOST`            | - `redis` <br/> - `localhost` <br/> - `192.168.1.10` <br/> | Specifies the hostname or IP address of the message broker used for communication between services. |
| `MESSAGE_CLIENT_PORT`            | - `7670` <br/> - `6379` <br/>                              | Specifies the port number on which the message broker is listening. |
| `MINIO_BUCKET`                   | `nv-ingest` <br/>                                        | Name of MinIO bucket, used to store image, table, and chart extractions. |
| `NGC_API_KEY`                    | `nvapi-*************` <br/>                              | An authorized NGC API key, used to interact with hosted NIMs. To create an NGC key, go to [https://org.ngc.nvidia.com/setup/api-keys](https://org.ngc.nvidia.com/setup/api-keys). |
| `NIM_NGC_API_KEY`                | —                                                          | The key that NIM microservices inside docker containers use to access NGC resources. This is necessary only in some cases when it is different from `NGC_API_KEY`. If this is not specified, `NGC_API_KEY` is used to access NGC resources. |
| `NVIDIA_BUILD_API_KEY`           | —                                                          | The key to access NIMs that are hosted on build.nvidia.com instead of a self-hosted NIM. This is necessary only in some cases when it is different from `NGC_API_KEY`. If this is not specified, `NGC_API_KEY` is used for build.nvidia.com. |
| `OTEL_EXPORTER_OTLP_ENDPOINT`    | `http://otel-collector:4317` <br/>                       | The endpoint for the OpenTelemetry exporter, used for sending telemetry data. |
| `REDIS_MORPHEUS_TASK_QUEUE`      | `morpheus_task_queue` <br/>                              | The name of the task queue in Redis where tasks are stored and processed. |
| `DOWNLOAD_LLAMA_TOKENIZER`       | `True` <br/>                                             | If `True`, the [llama-3.2 tokenizer](https://huggingface.co/meta-llama/Llama-3.2-1B) will be pre-dowloaded at build time. If not set to `True`, the (e5-large-unsupervised)[https://huggingface.co/intfloat/e5-large-unsupervised] tokenizer will be pre-downloaded. Note: setting this to `True` requires a HuggingFace access token with access to the gated Llama-3.2 models. See below for more info. |
| `HF_ACCESS_TOKEN`                | -                                                         | The HuggingFace access token used to pre-downlaod the Llama-3.2 tokenizer from HuggingFace (see above for more info). Llama 3.2 is a gated model, so you must [request access](https://huggingface.co/meta-llama/Llama-3.2-1B) to the Llama-3.2 models and then set this variable to a token that can access gated repositories on your behalf in order to use `DOWNLOAD_LLAMA_TOKENIZER=True`. |


## Library Mode Environment Variables

These environment variables apply specifically when running NV-Ingest in library mode.

| Name                              | Example                                                 | Description |
|-----------------------------------|---------------------------------------------------------|-------------|
| `NVIDIA_BUILD_API_KEY`            | `nvapi-...`                                            | API key for NVIDIA-hosted NIM services. |
| `NVIDIA_API_KEY`                  | `nvapi-...`                                            | Alternative NVIDIA API key (used in specific cases). |
| `PADDLE_HTTP_ENDPOINT`            | `https://ai.api.nvidia.com/v1/cv/baidu/paddleocr`      | HTTP endpoint for PaddleOCR inference. |
| `PADDLE_INFER_PROTOCOL`           | `http`                                                 | Protocol used for PaddleOCR requests. |
| `YOLOX_HTTP_ENDPOINT`             | `https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2` | HTTP endpoint for YOLOX-based page element detection. |
| `YOLOX_INFER_PROTOCOL`            | `http`                                                 | Protocol used for YOLOX-based inference. |
| `YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT` | `https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1` | HTTP endpoint for detecting graphical elements. |
| `YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL` | `http`                                               | Protocol used for graphical elements inference. |
| `YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT` | `https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1` | HTTP endpoint for table structure analysis. |
| `YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL` | `http`                                               | Protocol used for table structure inference. |
| `EMBEDDING_NIM_ENDPOINT`          | `https://integrate.api.nvidia.com/v1`                  | The endpoint for NV-Ingest's embedding NIM. |
| `EMBEDDING_NIM_MODEL_NAME`        | `nvidia/llama-3.2-nv-embedqa-1b-v2`              | The model name used for embedding generation. |
| `VLM_CAPTION_ENDPOINT`            | `https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions` | Endpoint for vision-language caption generation. |
| `VLM_CAPTION_MODEL_NAME`          | `meta/llama-3.2-11b-vision-instruct`                   | The model name for vision-language caption generation. |
| `NEMORETRIEVER_PARSE_HTTP_ENDPOINT` | `https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-parse` | Endpoint for image-to-text retrieval using NemoRetriever. |
| `NEMORETRIEVER_PARSE_INFER_PROTOCOL` | `http`                                              | Protocol for NemoRetriever Parse requests. |
