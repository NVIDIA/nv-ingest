<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

### **Environment Configuration Variables**

- **`MESSAGE_CLIENT_HOST`**:

  - **Description**: Specifies the hostname or IP address of the message broker used for communication between
    services.
  - **Example**: `redis`, `localhost`, `192.168.1.10`

- **`MESSAGE_CLIENT_PORT`**:

  - **Description**: Specifies the port number on which the message broker is listening.
  - **Example**: `7670`, `6379`

- **`CAPTION_CLASSIFIER_GRPC_TRITON`**:

  - **Description**: The endpoint where the caption classifier model is hosted using gRPC for communication. This is
    used to send requests for caption classification.
    You must specify only ONE of an http or gRPC endpoint. If both are specified gRPC will take precedence.
  - **Example**: `triton:8001`

- **`CAPTION_CLASSIFIER_MODEL_NAME`**:

  - **Description**: The name of the caption classifier model.
  - **Example**: `deberta_large`

- **`REDIS_MORPHEUS_TASK_QUEUE`**:

  - **Description**: The name of the task queue in Redis where tasks are stored and processed.
  - **Example**: `morpheus_task_queue`

- **`DOUGHNUT_TRITON_HOST`**:

  - **Description**: The hostname or IP address of the DOUGHNUT model service.
  - **Example**: `triton-doughnut`

- **`DOUGHNUT_TRITON_PORT`**:

  - **Description**: The port number on which the DOUGHNUT model service is listening.
  - **Example**: `8001`

- **`OTEL_EXPORTER_OTLP_ENDPOINT`**:

  - **Description**: The endpoint for the OpenTelemetry exporter, used for sending telemetry data.
  - **Example**: `http://otel-collector:4317`

- **`NGC_API_KEY`**:

  - **Description**: An authorized NGC API key, used to interact with hosted NIMs and can be generated here: https://org.ngc.nvidia.com/setup/personal-keys.
  - **Example**: `nvapi-*************`

- **`MINIO_BUCKET`**:

  - **Description**: Name of MinIO bucket, used to store image, table, and chart extractions.
  - **Example**: `nv-ingest`

- **`INGEST_LOG_LEVEL`**:

  - **Description**: The log level for the ingest service, which controls the verbosity of the logging output.
  - **Example**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

- **`NVIDIA_BUILD_API_KEY`**:
  - **Description**: This key is for when you are using the build.nvidia.com endpoint instead of a self hosted Deplot NIM.
    This is necessary only in some cases when it is different from `NGC_API_KEY`. If this is not specified, `NGC_API_KEY` will be used for bulid.nvidia.com.

- **`NIM_NGC_API_KEY`**:
  - **Description**: This key is by NIM microservices inside docker containers to access NGC resources.
    This is necessary only in some cases when it is different from `NGC_API_KEY`. If this is not specified, `NGC_API_KEY` will be used to access NGC resources.
