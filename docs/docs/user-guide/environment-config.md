# Environment Configuration Variables for NV-Ingest

The following are the environment configuration variables that you can specify in your .env file.


| Name                             | Example                        | Description                                                           |
|----------------------------------|--------------------------------|-----------------------------------------------------------------------|
| `CAPTION_CLASSIFIER_GRPC_TRITON` | `triton:8001` <br/>                                      | The endpoint where the caption classifier model is hosted using gRPC for communication. This is used to send requests for caption classification. You must specify only ONE of an http or gRPC endpoint. If both are specified gRPC will take precedence. |
| `CAPTION_CLASSIFIER_MODEL_NAME`  | `deberta_large` <br/>                                    | The name of the caption classifier model. |
| `INGEST_LOG_LEVEL`               | - `DEBUG` <br/> - `INFO` <br/> - `WARNING` <br/> - `ERROR` <br/> - `CRITICAL` <br/> | The log level for the ingest service, which controls the verbosity of the logging output. |
| `MESSAGE_CLIENT_HOST`            | - `redis` <br/> - `localhost` <br/> - `192.168.1.10` <br/> | Specifies the hostname or IP address of the message broker used for communication between services. |
| `MESSAGE_CLIENT_PORT`            | - `7670` <br/> - `6379` <br/>                              | Specifies the port number on which the message broker is listening. |
| `MINIO_BUCKET`                   | `nv-ingest` <br/>                                        | Name of MinIO bucket, used to store image, table, and chart extractions. |
| `NGC_API_KEY`                    | `nvapi-*************` <br/>                              | An authorized NGC API key, used to interact with hosted NIMs. To create an NGC key, go to [https://org.ngc.nvidia.com/setup/personal-keys](https://org.ngc.nvidia.com/setup/personal-keys). |
| `NIM_NGC_API_KEY`                | —                                                          | The key that NIM microservices inside docker containers use to access NGC resources. This is necessary only in some cases when it is different from `NGC_API_KEY`. If this is not specified, `NGC_API_KEY` is used to access NGC resources. |
| `NVIDIA_BUILD_API_KEY`           | —                                                          | The key to access NIMs that are hosted on build.nvidia.com instead of a self-hosted NIM. This is necessary only in some cases when it is different from `NGC_API_KEY`. If this is not specified, `NGC_API_KEY` is used for build.nvidia.com. |
| `OTEL_EXPORTER_OTLP_ENDPOINT`    | `http://otel-collector:4317` <br/>                       | The endpoint for the OpenTelemetry exporter, used for sending telemetry data. |
| `REDIS_MORPHEUS_TASK_QUEUE`      | `morpheus_task_queue` <br/>                              | The name of the task queue in Redis where tasks are stored and processed. |
| `DOWNLOAD_LLAMA_TOKENIZER`       | `True` <br/>                                             | If `True`, the [llama-3.2 tokenizer](https://huggingface.co/meta-llama/Llama-3.2-1B) will be pre-dowloaded at build time. If not set to `True`, the (e5-large-unsupervised)[https://huggingface.co/intfloat/e5-large-unsupervised] tokenizer will be pre-downloaded. Note: setting this to `True` requires a HuggingFace access token with access to the gated Llama-3.2 models. See below for more info. |
| `HF_ACCESS_TOKEN`                | -                                                         | The HuggingFace access token used to pre-downlaod the Llama-3.2 tokenizer from HuggingFace (see above for more info). Llama 3.2 is a gated model, so you must [request access](https://huggingface.co/meta-llama/Llama-3.2-1B) to the Llama-3.2 models and then set this variable to a token that can access gated repositories on your behalf in order to use `DOWNLOAD_LLAMA_TOKENIZER=True`. |
