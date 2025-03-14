# nv-ingest

![Version: 0.4.0](https://img.shields.io/badge/Version-0.4.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square)

NV-Ingest Microservice

## Maintainers

| Name | Email | Url |
| ---- | ------ | --- |
| NVIDIA Corporation |  | <https://www.nvidia.com/> |

## Requirements

| Repository | Name | Version |
|------------|------|---------|
| alias:nemo-microservices | nvidia-nim-nemoretriever-graphic-elements-v1 | 1.2.0 |
| alias:nemo-microservices | paddleocr-nim | 0.2.0 |
| alias:nemo-microservices | yolox-nim | 0.2.0 |
| alias:nvidia-nim | nvidia-nim-llama-32-nv-embedqa-1b-v2 | 1.3.0 |
| alias:nvidia-nim | text-embedding-nim | 1.1.0 |
| https://open-telemetry.github.io/opentelemetry-helm-charts | opentelemetry-collector | 0.78.1 |
| https://zilliztech.github.io/milvus-helm | milvus | 4.1.11 |
| https://zipkin.io/zipkin-helm | zipkin | 0.1.2 |
| oci://registry-1.docker.io/bitnamicharts | common | 2.x.x |
| oci://registry-1.docker.io/bitnamicharts | redis | 19.1.3 |

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| affinity | object | `{}` |  |
| autoscaling.enabled | bool | `false` |  |
| autoscaling.maxReplicas | int | `100` |  |
| autoscaling.metrics | list | `[]` |  |
| autoscaling.minReplicas | int | `1` |  |
| containerArgs | list | `[]` |  |
| containerSecurityContext | object | `{}` |  |
| embedqaDeployed | bool | `false` |  |
| envVars.EMBEDDING_NIM_ENDPOINT | string | `"http://nv-ingest-embedqa:8000/v1"` |  |
| envVars.EMBEDDING_NIM_MODEL_NAME | string | `"nvidia/llama-3.2-nv-embedqa-1b-v2"` |  |
| envVars.MESSAGE_CLIENT_HOST | string | `"nv-ingest-redis-master"` |  |
| envVars.MESSAGE_CLIENT_PORT | string | `"6379"` |  |
| envVars.MILVUS_ENDPOINT | string | `"http://nv-ingest-milvus:19530"` |  |
| envVars.MINIO_BUCKET | string | `"nv-ingest"` |  |
| envVars.MINIO_INTERNAL_ADDRESS | string | `"nv-ingest-minio:9000"` |  |
| envVars.MINIO_PUBLIC_ADDRESS | string | `"http://localhost:9000"` |  |
| envVars.NEMORETRIEVER_PARSE_HTTP_ENDPOINT | string | `"http://nim-vlm:8000/v1/chat/completions"` |  |
| envVars.NEMORETRIEVER_PARSE_INFER_PROTOCOL | string | `"http"` |  |
| envVars.NV_INGEST_DEFAULT_TIMEOUT_MS | string | `"1234"` |  |
| envVars.PADDLE_GRPC_ENDPOINT | string | `"nv-ingest-paddle:8001"` |  |
| envVars.PADDLE_HTTP_ENDPOINT | string | `"http://nv-ingest-paddle:8000/v1/infer"` |  |
| envVars.PADDLE_INFER_PROTOCOL | string | `"grpc"` |  |
| envVars.REDIS_MORPHEUS_TASK_QUEUE | string | `"morpheus_task_queue"` |  |
| envVars.VLM_CAPTION_ENDPOINT | string | `"https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"` |  |
| envVars.VLM_CAPTION_MODEL_NAME | string | `"meta/llama-3.2-11b-vision-instruct"` |  |
| envVars.YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT | string | `"nv-ingest-nvidia-nim-nemoretriever-graphic-elements-v1:8001"` |  |
| envVars.YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT | string | `"http://nv-ingest-nvidia-nim-nemoretriever-graphic-elements-v1:8000/v1/infer"` |  |
| envVars.YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL | string | `"http"` |  |
| envVars.YOLOX_GRPC_ENDPOINT | string | `"nv-ingest-yolox:8001"` |  |
| envVars.YOLOX_HTTP_ENDPOINT | string | `"http://nv-ingest-yolox:8000/v1/infer"` |  |
| envVars.YOLOX_INFER_PROTOCOL | string | `"grpc"` |  |
| envVars.YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT | string | `"nv-ingest-nvidia-nim-nemoretriever-table-structure-v1:8001"` |  |
| envVars.YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT | string | `"http://nv-ingest-nvidia-nim-nemoretriever-table-structure-v1:8000/v1/infer"` |  |
| envVars.YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL | string | `"http"` |  |
| extraEnvVarsCM | string | `""` |  |
| extraEnvVarsSecret | string | `""` |  |
| extraVolumeMounts | object | `{}` |  |
| extraVolumes | object | `{}` |  |
| fullnameOverride | string | `""` |  |
| image.pullPolicy | string | `"IfNotPresent"` |  |
| image.repository | string | `"nvcr.io/nvidia/nemo-microservices/nv-ingest"` |  |
| image.tag | string | `"24.12"` |  |
| imagePullSecret.create | bool | `false` |  |
| imagePullSecret.name | string | `"nvcrimagepullsecret"` |  |
| imagePullSecret.password | string | `""` |  |
| imagePullSecret.registry | string | `"nvcr.io"` |  |
| imagePullSecret.username | string | `"$oauthtoken"` |  |
| imagePullSecrets[0].name | string | `"nvcrimagepullsecret"` |  |
| imagePullSecrets[1].name | string | `"nemoMicroservicesPullSecret"` |  |
| ingress.annotations | object | `{}` |  |
| ingress.className | string | `""` |  |
| ingress.enabled | bool | `false` |  |
| ingress.hosts[0].host | string | `"chart-example.local"` |  |
| ingress.hosts[0].paths[0].path | string | `"/"` |  |
| ingress.hosts[0].paths[0].pathType | string | `"ImplementationSpecific"` |  |
| ingress.tls | list | `[]` |  |
| livenessProbe.enabled | bool | `false` |  |
| livenessProbe.enabled | bool | `true` |  |
| livenessProbe.failureThreshold | int | `20` |  |
| livenessProbe.httpGet.path | string | `"/v1/health/live"` |  |
| livenessProbe.httpGet.path | string | `"/health"` |  |
| livenessProbe.httpGet.port | int | `7670` |  |
| livenessProbe.httpGet.port | string | `"http"` |  |
| livenessProbe.initialDelaySeconds | int | `15` |  |
| livenessProbe.initialDelaySeconds | int | `120` |  |
| livenessProbe.periodSeconds | int | `10` |  |
| livenessProbe.periodSeconds | int | `20` |  |
| livenessProbe.successThreshold | int | `1` |  |
| livenessProbe.timeoutSeconds | int | `20` |  |
| logLevel | string | `"DEBUG"` |  |
| milvus.cluster.enabled | bool | `false` |  |
| milvus.etcd.persistence.storageClass | string | `nil` |  |
| milvus.etcd.replicaCount | int | `1` |  |
| milvus.minio.mode | string | `"standalone"` |  |
| milvus.minio.persistence.size | string | `"10Gi"` |  |
| milvus.minio.persistence.storageClass | string | `nil` |  |
| milvus.pulsar.enabled | bool | `false` |  |
| milvus.standalone.extraEnv[0].name | string | `"LOG_LEVEL"` |  |
| milvus.standalone.extraEnv[0].value | string | `"error"` |  |
| milvus.standalone.persistence.persistentVolumeClaim.size | string | `"10Gi"` |  |
| milvus.standalone.persistence.persistentVolumeClaim.storageClass | string | `nil` |  |
| milvusDeployed | bool | `true` |  |
| nameOverride | string | `""` |  |
| nemo.groupID | string | `"1000"` |  |
| nemo.userID | string | `"1000"` |  |
| ngcSecret.create | bool | `false` |  |
| ngcSecret.password | string | `""` |  |
| nodeSelector | object | `{}` |  |
| nvEmbedqaDeployed | bool | `true` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.env[0].value | string | `"8000"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.fullnameOverride | string | `"nv-ingest-embedqa"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.image.repository | string | `"nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.image.tag | string | `"1.3.1"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.nim.grpcPort | int | `8001` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.service.grpcPort | int | `8001` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.service.name | string | `"nv-ingest-embedqa"` |  |
| opentelemetry-collector.config.exporters.debug.verbosity | string | `"detailed"` |  |
| opentelemetry-collector.config.exporters.zipkin.endpoint | string | `"http://nv-ingest-zipkin:9411/api/v2/spans"` |  |
| opentelemetry-collector.config.extensions.health_check | object | `{}` |  |
| opentelemetry-collector.config.extensions.zpages.endpoint | string | `"0.0.0.0:55679"` |  |
| opentelemetry-collector.config.processors.batch | object | `{}` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].name | string | `"drop_noisy_traces_url"` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].string_attribute.enabled_regex_matching | bool | `true` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].string_attribute.invert_match | bool | `true` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].string_attribute.key | string | `"http.target"` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].string_attribute.values[0] | string | `"\\/health"` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].type | string | `"string_attribute"` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].context | string | `"span"` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].statements[0] | string | `"set(status.code, 1) where attributes[\"http.path\"] == \"/health\""` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].statements[1] | string | `"replace_match(attributes[\"http.route\"], \"/v1\", attributes[\"http.target\"]) where attributes[\"http.target\"] != nil"` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].statements[2] | string | `"replace_pattern(name, \"/v1\", attributes[\"http.route\"]) where attributes[\"http.route\"] != nil"` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].statements[3] | string | `"set(name, Concat([name, attributes[\"http.url\"]], \" \")) where name == \"POST\""` |  |
| opentelemetry-collector.config.receivers.otlp.protocols.grpc.endpoint | string | `"${env:MY_POD_IP}:4317"` |  |
| opentelemetry-collector.config.receivers.otlp.protocols.http.cors.allowed_origins[0] | string | `"*"` |  |
| opentelemetry-collector.config.service.extensions[0] | string | `"zpages"` |  |
| opentelemetry-collector.config.service.extensions[1] | string | `"health_check"` |  |
| opentelemetry-collector.config.service.pipelines.logs.exporters[0] | string | `"debug"` |  |
| opentelemetry-collector.config.service.pipelines.logs.processors[0] | string | `"batch"` |  |
| opentelemetry-collector.config.service.pipelines.logs.receivers[0] | string | `"otlp"` |  |
| opentelemetry-collector.config.service.pipelines.metrics.exporters[0] | string | `"debug"` |  |
| opentelemetry-collector.config.service.pipelines.metrics.processors[0] | string | `"batch"` |  |
| opentelemetry-collector.config.service.pipelines.metrics.receivers[0] | string | `"otlp"` |  |
| opentelemetry-collector.config.service.pipelines.traces.exporters[0] | string | `"debug"` |  |
| opentelemetry-collector.config.service.pipelines.traces.exporters[1] | string | `"zipkin"` |  |
| opentelemetry-collector.config.service.pipelines.traces.processors[0] | string | `"tail_sampling"` |  |
| opentelemetry-collector.config.service.pipelines.traces.processors[1] | string | `"transform"` |  |
| opentelemetry-collector.config.service.pipelines.traces.receivers[0] | string | `"otlp"` |  |
| opentelemetry-collector.mode | string | `"deployment"` |  |
| otelDeployed | bool | `true` |  |
| otelEnabled | bool | `true` |  |
| otelEnvVars.OTEL_LOGS_EXPORTER | string | `"none"` |  |
| otelEnvVars.OTEL_METRICS_EXPORTER | string | `"otlp"` |  |
| otelEnvVars.OTEL_PROPAGATORS | string | `"tracecontext,baggage"` |  |
| otelEnvVars.OTEL_PYTHON_EXCLUDED_URLS | string | `"health"` |  |
| otelEnvVars.OTEL_RESOURCE_ATTRIBUTES | string | `"deployment.environment=$(NAMESPACE)"` |  |
| otelEnvVars.OTEL_SERVICE_NAME | string | `"nemo-retrieval-service"` |  |
| otelEnvVars.OTEL_TRACES_EXPORTER | string | `"otlp"` |  |
| paddleocr-nim.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| paddleocr-nim.env[0].value | string | `"8000"` |  |
| paddleocr-nim.fullnameOverride | string | `"nv-ingest-paddle"` |  |
| paddleocr-nim.image.repository | string | `"nvcr.io/nvidia/nemo-microservices/paddleocr"` |  |
| paddleocr-nim.image.tag | string | `"1.0.0"` |  |
| paddleocr-nim.nim.grpcPort | int | `8001` |  |
| paddleocr-nim.service.grpcPort | int | `8001` |  |
| paddleocr-nim.service.name | string | `"nv-ingest-paddle"` |  |
| paddleocrDeployed | bool | `true` |  |
| podAnnotations."traffic.sidecar.istio.io/excludeOutboundPorts" | string | `"8007"` |  |
| podLabels | object | `{}` |  |
| podSecurityContext.fsGroup | int | `1000` |  |
| readinessProbe.enabled | bool | `false` |  |
| readinessProbe.enabled | bool | `true` |  |
| readinessProbe.failureThreshold | int | `220` |  |
| readinessProbe.httpGet.path | string | `"/health"` |  |
| readinessProbe.httpGet.path | string | `"/v1/health/ready"` |  |
| readinessProbe.httpGet.port | string | `"http"` |  |
| readinessProbe.httpGet.port | int | `7670` |  |
| readinessProbe.initialDelaySeconds | int | `120` |  |
| readinessProbe.initialDelaySeconds | int | `30` |  |
| readinessProbe.periodSeconds | int | `30` |  |
| readinessProbe.periodSeconds | int | `30` |  |
| readinessProbe.successThreshold | int | `1` |  |
| readinessProbe.timeoutSeconds | int | `10` |  |
| redis.auth.enabled | bool | `false` |  |
| redis.master.persistence.size | string | `"50Gi"` |  |
| redis.master.resources.limits.memory | string | `"12Gi"` |  |
| redis.master.resources.requests.memory | string | `"6Gi"` |  |
| redis.replica.persistence.size | string | `"50Gi"` |  |
| redis.replica.replicaCount | int | `1` |  |
| redis.replica.resources.limits.memory | string | `"12Gi"` |  |
| redis.replica.resources.requests.memory | string | `"6Gi"` |  |
| redisDeployed | bool | `true` |  |
| replicaCount | int | `1` |  |
| resources.limits."nvidia.com/gpu" | int | `1` |  |
| resources.limits.cpu | string | `"48000m"` |  |
| resources.limits.memory | string | `"90Gi"` |  |
| resources.requests.cpu | string | `"24000m"` |  |
| resources.requests.memory | string | `"24Gi"` |  |
| service.annotations | object | `{}` |  |
| service.labels | object | `{}` |  |
| service.name | string | `""` |  |
| service.nodePort | string | `nil` |  |
| service.port | int | `7670` |  |
| service.type | string | `"ClusterIP"` |  |
| serviceAccount.annotations | object | `{}` |  |
| serviceAccount.automount | bool | `true` |  |
| serviceAccount.create | bool | `true` |  |
| serviceAccount.name | string | `""` |  |
| startupProbe.enabled | bool | `false` |  |
| startupProbe.failureThreshold | int | `220` |  |
| startupProbe.httpGet.path | string | `"/health"` |  |
| startupProbe.httpGet.port | string | `"http"` |  |
| startupProbe.initialDelaySeconds | int | `120` |  |
| startupProbe.periodSeconds | int | `30` |  |
| startupProbe.successThreshold | int | `1` |  |
| startupProbe.timeoutSeconds | int | `10` |  |
| text-embedding-nim.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| text-embedding-nim.env[0].value | string | `"8000"` |  |
| text-embedding-nim.fullnameOverride | string | `"nv-ingest-embedqa"` |  |
| text-embedding-nim.image.repository | string | `"nvcr.io/nim/nvidia/nv-embedqa-e5-v5"` |  |
| text-embedding-nim.image.tag | string | `"1.1.0"` |  |
| text-embedding-nim.nim.grpcPort | int | `8001` |  |
| text-embedding-nim.service.grpcPort | int | `8001` |  |
| text-embedding-nim.service.name | string | `"nv-ingest-embedqa"` |  |
| tmpDirSize | string | `"16Gi"` |  |
| tolerations | list | `[]` |  |
| triton.enabled | bool | `false` |  |
| yolox-nim.deployed | bool | `true` |  |
| yolox-nim.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| yolox-nim.env[0].value | string | `"8000"` |  |
| yolox-nim.fullnameOverride | string | `"nv-ingest-yolox"` |  |
| yolox-nim.image.repository | string | `"nvcr.io/nvidia/nemo-microservices/nv-yolox-page-elements-v1"` |  |
| yolox-nim.image.tag | string | `"1.0.0"` |  |
| yolox-nim.nim.grpcPort | int | `8001` |  |
| yolox-nim.service.grpcPort | int | `8001` |  |
| yolox-nim.service.name | string | `"nv-ingest-yolox"` |  |
| yoloxDeployed | bool | `true` |  |
| zipkinDeployed | bool | `true` |  |

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.14.2](https://github.com/norwoodj/helm-docs/releases/v1.14.2)
