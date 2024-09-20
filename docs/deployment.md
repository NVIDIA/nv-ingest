<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

### Launch nv-ingest micro-service(s)

```bash
# Redis is our message broker for the ingest service, always required.
docker compose up -d redis

# `yolox`, `deplot`, `cached`, and `paddle` are NIMs used to perform table and chart extraction.
docker compose up -d yolox deplot cached paddle

# Optional (MinIO) is an object store to store extracted images, tables, and charts, by default it is commented out in the docker compose file.
# The `store` task will not be functional without this service or external s3 compliant object store.
docker compose up -d minio

# Optional (Milvus) is a vector database to embeddings for multi-model extractions, by default it is commented out in the docker compose file.
# The `vdb_upload` task will not be functional without this serivce or external Milvus database.
docker compose up -d etcd minio milvus attu

# Optional (Telemetry services)
# TODO: Add examples for telemetry services
docker compose up -d otel-collector prometheus grafana zipkin

# Optional (Embedding NIM) Stand up `nv-embedqa-e5-v5` NIM to calculate embeddings for extracted content.
# The `embed` task will not be functional without this service.
docker compose up -d embedding

# Optional (Triton) See below for Triton setup we need Triton for any model inference
# This is only needed for captioning or DOUGHNUT based extraction.
docker compose up -d triton

# Ingest service
docker compose up -d nv-ingest-ms-runtime
```

You should see something like this:

```bash
CONTAINER ID   IMAGE                                        COMMAND                 CREATED        STATUS                PORTS                              NAMES
6065c12d6034   .../nv-ingest:2024.6.3.dev0                 "/opt/conda/bin/tini…"   6 hours ago    Up 6 hours                                               nv-ingest-ms-runtime-1
c1f1f6b9cc8c   .../tritonserver:24.05-py3       "/opt/nvidia/nvidia_…"   5 days ago     Up 8 hours            0.0.0.0:8000-8002->8000-8002/tcp   devin-nv-ingest-triton-1
d277cf2c2703   redis/redis-stack                           "/entrypoint.sh"         2 weeks ago    Up 8 hours            0.0.0.0:6379->6379/tcp, 8001/tcp   devin-nv-ingest-redis-1
```
