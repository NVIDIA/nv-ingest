# NV-Ingest Helm Charts

This documentation contains documentation for the NV-Ingest Helm charts.

> [!Note]
> NV-Ingest is also known as NVIDIA Ingest and NeMo Retriever extraction.

## Prerequisites

Before you install the Helm charts, be sure you meet the hardware and software prerequisites. Refer to the [supported configurations](https://github.com/NVIDIA/nv-ingest?tab=readme-ov-file#hardware).

> Starting with version 26.0.0, the [NVIDIA NIM Operator](https://docs.nvidia.com/nim-operator/latest/install.html) is **required**. All NIM services are now deployed via NIM Operator CRDs (NIMCache and NIMService), not Helm subcharts.
>
> **Upgrading from 25.9.0:**
> 1. Install NIM Operator before upgrading
> 2. Update your values file with the new configuration keys:
>
> | 25.9.0 | 26.x |
> |--------|------|
> | `nim-vlm-image-captioning.deployed=true` | `nimOperator.nemotron_nano_12b_v2_vl.enabled=true` |
> | `paddleocr-nim.deployed=true` | `nimOperator.paddleocr.enabled=true` |
> | `riva-nim.deployed=true` | Not yet available |
> | `nim-vlm-text-extraction.deployed=true` | Not yet available |

The [Nvidia GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) must also be installed and configured in your cluster.

## Initial Environment Setup

1. Create your namespace by running the following code.

```bash
NAMESPACE=nv-ingest
kubectl create namespace ${NAMESPACE}
```

2. Add the Helm repos by running the following code.

```bash
# NVIDIA nemo-microservices NGC repository
helm repo add nemo-microservices https://helm.ngc.nvidia.com/nvidia/nemo-microservices --username='$oauthtoken' --password=<NGC_API_KEY>

# NVIDIA NIM NGC repository
helm repo add nvidia-nim https://helm.ngc.nvidia.com/nim/nvidia --username='$oauthtoken' --password=<NGC_API_KEY>

# NVIDIA NIM Base repository
helm repo add nim https://helm.ngc.nvidia.com/nim --username='$oauthtoken' --password=<NGC_API_KEY>

# NVIDIA NIM baidu NGC repository
helm repo add baidu-nim https://helm.ngc.nvidia.com/nim/baidu --username='$oauthtoken' --password=<YOUR API KEY>
```

## Install or Upgrade the Helm Chart

To install or upgrade the Helm chart, run the following code.

```bash
helm upgrade \
    --install \
    nv-ingest \
    https://helm.ngc.nvidia.com/nvidia/nemo-microservices/charts/nv-ingest-25.9.0.tgz \
    -n ${NAMESPACE} \
    --username '$oauthtoken' \
    --password "${NGC_API_KEY}" \
    --set ngcImagePullSecret.create=true \
    --set ngcImagePullSecret.password="${NGC_API_KEY}" \
    --set ngcApiSecret.create=true \
    --set ngcApiSecret.password="${NGC_API_KEY}" \
    --set image.repository="nvcr.io/nvidia/nemo-microservices/nv-ingest" \
    --set image.tag="25.9.0"
```

Optionally you can create your own versions of the `Secrets` if you do not want to use the creation via the helm chart.

```bash

NAMESPACE=nv-ingest
DOCKER_CONFIG='{"auths":{"nvcr.io":{"username":"$oauthtoken", "password":"'${NGC_API_KEY}'" }}}'
echo -n $DOCKER_CONFIG | base64 -w0
NGC_REGISTRY_PASSWORD=$(echo -n $DOCKER_CONFIG | base64 -w0 )

kubectl apply -n ${NAMESPACE} -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: ngcImagePullSecret
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: ${NGC_REGISTRY_PASSWORD}
EOF
kubectl create -n ${NAMESPACE} secret generic ngc-api --from-literal=NGC_API_KEY=${NGC_API_KEY}

```

Alternatively, you can also use an External Secret Store like Vault, the name of the secret name expected for the NGC API is `ngc-api` and the secret name expected for NVCR is `ngcImagePullSecret`.

In this case, make sure to remove the following from your helm command:

```bash
    --set ngcImagePullSecret.create=true \
    --set ngcImagePullSecret.password="${NGC_API_KEY}" \
    --set ngcApiSecret.create=true \
    --set ngcApiSecret.password="${NGC_API_KEY}" \
```

## Usage

Jobs are submitted via the `nv-ingest-cli` command.

#### NV-Ingest CLI Installation

NV-Ingest uses a HTTP/Rest based submission method. By default the Rest service runs on port `7670`.

First, build `nv-ingest-cli` from the source to ensure you have the latest code.
For more information, refer to [NV-Ingest-Client](https://github.com/NVIDIA/nv-ingest/tree/main/client).

```bash
# Just to be cautious we remove any existing installation
pip uninstall nv-ingest-client

pip install nv-ingest-client==25.9.0
```

#### Rest Endpoint Ingress

It is recommended that the end user provide a mechanism for [`Ingress`](https://kubernetes.io/docs/concepts/services-networking/ingress/) for the NV-Ingest pod.
You can test outside of your Kubernetes cluster by [port-forwarding](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_port-forward/) the NV-Ingest pod to your local environment.

Example:

You can find the name of your NV-Ingest pod you want to forward traffic to by running:

```bash
kubectl get pods -n <namespace> --no-headers -o custom-columns=":metadata.name"
kubectl port-forward -n ${NAMESPACE} service/nv-ingest 7670:7670
```

The output will look similar to the following but with different auto-generated sequences.

```
nv-ingest-674f6b7477-65nvm
nv-ingest-etcd-0
nv-ingest-milvus-standalone-7f8ffbdfbc-jpmlj
nv-ingest-minio-7cbd4f5b9d-99hl4
nv-ingest-opentelemetry-collector-7bb59d57fc-4h59q
nv-ingest-ocr-0
nv-ingest-redis-master-0
nv-ingest-redis-replicas-0
nv-ingest-yolox-0
nv-ingest-zipkin-77b5fc459f-ptsj6
```

```bash
kubectl port-forward -n ${NAMESPACE} nv-ingest-674f6b7477-65nvm 7670:7670
```

#### Enabling GPU time-slicing

##### Configure the NVIDIA Operator

The NVIDIA GPU operator is used to make the Kubernetes cluster aware of the GPUs.

```bash
helm install --wait --generate-name --repo https://helm.ngc.nvidia.com/nvidia \
    --namespace gpu-operator --create-namespace \
    gpu-operator --set driver.enabled=false
```

Once this is installed we can check the number of available GPUs. Your ouput might differ depending on the number
of GPUs on your nodes.

```console
kubectl get nodes -o json | jq -r '.items[] | select(.metadata.name | test("-worker[0-9]*$")) | {name: .metadata.name, "nvidia.com/gpu": .status.allocatable["nvidia.com/gpu"]}'
{
  "name": "one-worker-one-gpu-worker",
  "nvidia.com/gpu": "1"
}
```

##### Enable time-slicing

With the NVIDIA GPU operator properly installed we want to enable time-slicing to allow more than one Pod to use this GPU. We do this by creating a time-slicing config file using [`time-slicing-config.yaml`](time-slicing/time-slicing-config.yaml).

```bash
kubectl apply -n gpu-operator -f time-slicing/time-slicing-config.yaml
```

Then update the cluster policy to use this new config.

```bash
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    -n gpu-operator --type merge \
    -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config", "default": "any"}}}}'
```

Now if we check the number of GPUs available we should see `16`, but note this is still just GPU `0` which we exposed to the node, just sliced into `16`.

```console
kubectl get nodes -o json | jq -r '.items[] | select(.metadata.name | test("-worker[0-9]*$")) | {name: .metadata.name, "nvidia.com/gpu": .status.allocatable["nvidia.com/gpu"]}'
{
  "name": "one-worker-one-gpu-worker",
  "nvidia.com/gpu": "16"
}
```

#### Enable NVIDIA GPU MIG

[NVIDIA MIG](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html) is a technology that enables a specific GPU to be sliced into individual virtual GPUs.
This approach is considered better for production environments than time-slicing, because it isolates it process to a pre-allocated amount of compute and memory.

Use this section to learn how to enable NVIDIA GPU MIG.

##### Compatible GPUs

For the list of GPUs that are compatible with MIG, refer to [the MIG support matrix](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#supported-gpus).

##### Understanding GPU profiles

You can think of a GPU profile like a traditional virtual computer (vm), where each vm has a predefined set of compute and memory.

Each NVIDIA GPU has different valid profiles. To see the profiles that are available for your GPU, run the following code.

```bash
nvidia-smi mig -lgip
```

The complete matrix of available profiles for your GPU appears. To understand the output, refer to [Supported MIG Profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#supported-mig-profiles).

##### MIG Profile configuration

An example MIG profile for a DGX H100 can be found in mig/nv-ingest-mig-config.yaml. This profile demonstrates mixed MIG modes across different GPUs
on the DGX machine.
You should edit the file for your scenario, and include only the profiles supported by your GPU.

To install the MIG profile on the Kubernetes cluster, use the following code.

```bash
kubectl apply -n gpu-operator -f mig/nv-ingest-mig-config.yaml
```

After the configmap is installed, you can adjust the MIG profile to be `mixed`. We do this because our configuration
file specifies different profiles across the GPUs. This is not required if you are not using a `mixed` MIG mode.

```bash
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    --type='json' \
    -p='[{"op":"replace", "path":"/spec/mig/strategy", "value":"mixed"}]'
```

Patch the cluster so MIG manager uses the custom config map by using the following code.

```bash
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    --type='json' \
    -p='[{"op":"replace", "path":"/spec/migManager/config/name", "value":"nv-ingest-mig-config"}]'
```

Label the nodes with which MIG profile you would like for them to use by using the following code.

```bash
kubectl label nodes <node-name> nvidia.com/mig.config=single-gpu-nv-ingest --overwrite
```

Validate that the configuration was applied by running the following code.

```bash
kubectl logs -n gpu-operator -l app=nvidia-mig-manager -c nvidia-mig-manager
```

You can adjust Kubernetes request and limit resources for MIG by using a Helm values file. To use a MIG values file in conjunction with other values files, append `-f mig/nv-ingest-mig-values.yaml` to your Helm command. For an example Helm values file for MIG settings, refer to [mig/nv-ingest-mig-config.yaml](mig/nv-ingest-mig-config.yaml). This file is only an example, and you should adjust the values for your environment and specific needs.

#### Running Jobs

Here is a sample invocation of a PDF extraction task using the port forward above:

```bash
mkdir -p ./processed_docs

nv-ingest-cli \
  --doc /path/to/your/unique.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_text": true, "extract_images": true, "extract_tables": true, "extract_charts": true}' \
  --client_host=localhost \
  --client_port=7670
```

You can also use NV-Ingest's Python client API to interact with the service running in the cluster. Use the same host and port as in the previous nv-ingest-cli example.

## Requirements

| Repository | Name | Version |
|------------|------|---------|
| https://open-telemetry.github.io/opentelemetry-helm-charts | opentelemetry-collector | 0.133.0 |
| https://prometheus-community.github.io/helm-charts | prometheus | 27.12.1 |
| https://zilliztech.github.io/milvus-helm | milvus | 4.1.11 |
| https://zipkin.io/zipkin-helm | zipkin | 0.4.0 |
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
| envVars.ARROW_DEFAULT_MEMORY_POOL | string | `"system"` |  |
| envVars.AUDIO_GRPC_ENDPOINT | string | `"nv-ingest-riva-nim:50051"` |  |
| envVars.AUDIO_INFER_PROTOCOL | string | `"grpc"` |  |
| envVars.COMPONENTS_TO_READY_CHECK | string | `"ALL"` |  |
| envVars.EMBEDDING_NIM_ENDPOINT | string | `"http://llama-32-nv-embedqa-1b-v2:8000/v1"` |  |
| envVars.EMBEDDING_NIM_MODEL_NAME | string | `"nvidia/llama-3.2-nv-embedqa-1b-v2"` |  |
| envVars.IMAGE_STORAGE_PUBLIC_BASE_URL | string | `""` |  |
| envVars.IMAGE_STORAGE_URI | string | `"s3://nv-ingest/artifacts/store/images"` |  |
| envVars.INGEST_DISABLE_DYNAMIC_SCALING | bool | `true` |  |
| envVars.INGEST_DYNAMIC_MEMORY_THRESHOLD | float | `0.8` |  |
| envVars.INGEST_EDGE_BUFFER_SIZE | int | `64` |  |
| envVars.INGEST_LOG_LEVEL | string | `"DEFAULT"` |  |
| envVars.MAX_INGEST_PROCESS_WORKERS | int | `16` |  |
| envVars.MESSAGE_CLIENT_HOST | string | `"nv-ingest-redis-master"` |  |
| envVars.MESSAGE_CLIENT_PORT | string | `"6379"` |  |
| envVars.MESSAGE_CLIENT_TYPE | string | `"redis"` |  |
| envVars.MILVUS_ENDPOINT | string | `"http://nv-ingest-milvus:19530"` |  |
| envVars.MINIO_ACCESS_KEY | string | `"minioadmin"` |  |
| envVars.MINIO_BUCKET | string | `"nv-ingest"` |  |
| envVars.MINIO_INTERNAL_ADDRESS | string | `"nv-ingest-minio:9000"` |  |
| envVars.MINIO_PUBLIC_ADDRESS | string | `"http://localhost:9000"` |  |
| envVars.MINIO_SECRET_KEY | string | `"minioadmin"` |  |
| envVars.MODEL_PREDOWNLOAD_PATH | string | `"/workspace/models/"` |  |
| envVars.NEMOTRON_PARSE_HTTP_ENDPOINT | string | `"http://nim-vlm-text-extraction-nemotron-parse:8000/v1/chat/completions"` |  |
| envVars.NEMOTRON_PARSE_INFER_PROTOCOL | string | `"http"` |  |
| envVars.NEMOTRON_PARSE_MODEL_NAME | string | `"nvidia/nemotron-parse"` |  |
| envVars.NV_INGEST_MAX_UTIL | int | `48` |  |
| envVars.OCR_GRPC_ENDPOINT | string | `"nemoretriever-ocr-v1:8001"` |  |
| envVars.OCR_HTTP_ENDPOINT | string | `"http://nemoretriever-ocr-v1:8000/v1/infer"` |  |
| envVars.OCR_INFER_PROTOCOL | string | `"grpc"` |  |
| envVars.OCR_MODEL_NAME | string | `"scene_text_ensemble"` |  |
| envVars.OMP_NUM_THREADS | int | `1` |  |
| envVars.PADDLE_GRPC_ENDPOINT | string | `"nv-ingest-paddle:8001"` |  |
| envVars.PADDLE_HTTP_ENDPOINT | string | `"http://nv-ingest-paddle:8000/v1/infer"` |  |
| envVars.PADDLE_INFER_PROTOCOL | string | `"grpc"` |  |
| envVars.RAY_num_grpc_threads | int | `1` |  |
| envVars.REDIS_INGEST_TASK_QUEUE | string | `"ingest_task_queue"` |  |
| envVars.VLM_CAPTION_MODEL_NAME | string | `"nvidia/nemotron-nano-12b-v2-vl"` |  |
| envVars.VLM_CAPTION_PROMPT | string | `"Caption the content of this image:"` |  |
| envVars.VLM_CAPTION_SYSTEM_PROMPT | string | `"/no_think"` |  |
| envVars.YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT | string | `"nemoretriever-graphic-elements-v1:8001"` |  |
| envVars.YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT | string | `"http://nemoretriever-graphic-elements-v1:8000/v1/infer"` |  |
| envVars.YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL | string | `"grpc"` |  |
| envVars.YOLOX_GRPC_ENDPOINT | string | `"nemoretriever-page-elements-v3:8001"` |  |
| envVars.YOLOX_HTTP_ENDPOINT | string | `"http://nemoretriever-page-elements-v3:8000/v1/infer"` |  |
| envVars.YOLOX_INFER_PROTOCOL | string | `"grpc"` |  |
| envVars.YOLOX_PAGE_IMAGE_FORMAT | string | `"JPEG"` |  |
| envVars.YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT | string | `"nemoretriever-table-structure-v1:8001"` |  |
| envVars.YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT | string | `"http://nemoretriever-table-structure-v1:8000/v1/infer"` |  |
| envVars.YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL | string | `"grpc"` |  |
| extraEnvVarsCM | string | `""` |  |
| extraEnvVarsSecret | string | `""` |  |
| extraVolumeMounts | object | `{}` |  |
| extraVolumes | object | `{}` |  |
| fullnameOverride | string | `""` |  |
| image.pullPolicy | string | `"IfNotPresent"` |  |
| image.repository | string | `"nvcr.io/nvidia/nemo-microservices/nv-ingest"` |  |
| image.tag | string | `"25.9.0"` |  |
| imagePullSecrets[0].name | string | `"ngc-api"` |  |
| imagePullSecrets[1].name | string | `"ngc-secret"` |  |
| ingress.annotations | object | `{}` |  |
| ingress.className | string | `""` |  |
| ingress.enabled | bool | `false` |  |
| ingress.hosts[0].host | string | `"chart-example.local"` |  |
| ingress.hosts[0].paths[0].path | string | `"/"` |  |
| ingress.hosts[0].paths[0].pathType | string | `"ImplementationSpecific"` |  |
| ingress.tls | list | `[]` |  |
| livenessProbe.enabled | bool | `false` |  |
| livenessProbe.failureThreshold | int | `20` |  |
| livenessProbe.httpGet.path | string | `"/v1/health/live"` |  |
| livenessProbe.httpGet.port | string | `"http"` |  |
| livenessProbe.initialDelaySeconds | int | `120` |  |
| livenessProbe.periodSeconds | int | `10` |  |
| livenessProbe.successThreshold | int | `1` |  |
| livenessProbe.timeoutSeconds | int | `20` |  |
| logLevel | string | `"DEFAULT"` |  |
| milvus.cluster.enabled | bool | `false` |  |
| milvus.etcd.extraVolumeMounts | list | `[]` |  |
| milvus.etcd.extraVolumes | list | `[]` |  |
| milvus.etcd.image.repository | string | `"milvusdb/etcd"` |  |
| milvus.etcd.image.tag | string | `"3.5.23-r2"` |  |
| milvus.etcd.persistence.storageClass | string | `nil` |  |
| milvus.etcd.replicaCount | int | `1` |  |
| milvus.image.all.repository | string | `"milvusdb/milvus"` |  |
| milvus.image.all.tag | string | `"v2.6.5-gpu"` |  |
| milvus.minio.bucketName | string | `"nv-ingest"` |  |
| milvus.minio.enabled | bool | `true` |  |
| milvus.minio.image.tag | string | `"RELEASE.2025-09-07T16-13-09Z"` |  |
| milvus.minio.mode | string | `"standalone"` |  |
| milvus.minio.persistence.size | string | `"50Gi"` |  |
| milvus.minio.persistence.storageClass | string | `nil` |  |
| milvus.pulsar.enabled | bool | `false` |  |
| milvus.pulsarv3.enabled | bool | `false` |  |
| milvus.standalone.extraEnv[0].name | string | `"LOG_LEVEL"` |  |
| milvus.standalone.extraEnv[0].value | string | `"error"` |  |
| milvus.standalone.persistence.persistentVolumeClaim.size | string | `"50Gi"` |  |
| milvus.standalone.persistence.persistentVolumeClaim.storageClass | string | `nil` |  |
| milvus.standalone.resources.limits."nvidia.com/gpu" | int | `1` |  |
| milvusDeployed | bool | `true` |  |
| nameOverride | string | `""` |  |
| nemo.groupID | string | `"1000"` |  |
| nemo.userID | string | `"1000"` |  |
| ngcApiSecret.create | bool | `false` |  |
| ngcApiSecret.password | string | `""` |  |
| ngcImagePullSecret.create | bool | `false` |  |
| ngcImagePullSecret.name | string | `"ngcImagePullSecret"` |  |
| ngcImagePullSecret.password | string | `""` |  |
| ngcImagePullSecret.registry | string | `"nvcr.io"` |  |
| ngcImagePullSecret.username | string | `"$oauthtoken"` |  |
| nimOperator.embedqa.authSecret | string | `"ngc-api"` |  |
| nimOperator.embedqa.enabled | bool | `true` |  |
| nimOperator.embedqa.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nimOperator.embedqa.env[0].value | string | `"8000"` |  |
| nimOperator.embedqa.env[1].name | string | `"NIM_TRITON_LOG_VERBOSE"` |  |
| nimOperator.embedqa.env[1].value | string | `"1"` |  |
| nimOperator.embedqa.env[2].name | string | `"OMP_NUM_THREADS"` |  |
| nimOperator.embedqa.env[2].value | string | `"1"` |  |
| nimOperator.embedqa.env[3].name | string | `"NIM_TRITON_PERFORMANCE_MODE"` |  |
| nimOperator.embedqa.env[3].value | string | `"throughput"` |  |
| nimOperator.embedqa.expose.service.grpcPort | int | `8001` |  |
| nimOperator.embedqa.expose.service.port | int | `8000` |  |
| nimOperator.embedqa.expose.service.type | string | `"ClusterIP"` |  |
| nimOperator.embedqa.image.pullPolicy | string | `"IfNotPresent"` |  |
| nimOperator.embedqa.image.pullSecrets[0] | string | `"ngc-secret"` |  |
| nimOperator.embedqa.image.repository | string | `"nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2"` |  |
| nimOperator.embedqa.image.tag | string | `"1.10.0"` |  |
| nimOperator.embedqa.replicas | int | `1` |  |
| nimOperator.embedqa.resources.limits."nvidia.com/gpu" | int | `1` |  |
| nimOperator.embedqa.storage.pvc.create | bool | `true` |  |
| nimOperator.embedqa.storage.pvc.size | string | `"50Gi"` |  |
| nimOperator.embedqa.storage.pvc.volumeAccessMode | string | `"ReadWriteMany"` |  |
| nimOperator.graphic_elements.authSecret | string | `"ngc-api"` |  |
| nimOperator.graphic_elements.enabled | bool | `true` |  |
| nimOperator.graphic_elements.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nimOperator.graphic_elements.env[0].value | string | `"8000"` |  |
| nimOperator.graphic_elements.env[1].name | string | `"NIM_TRITON_LOG_VERBOSE"` |  |
| nimOperator.graphic_elements.env[1].value | string | `"1"` |  |
| nimOperator.graphic_elements.env[2].name | string | `"NIM_TRITON_RATE_LIMIT"` |  |
| nimOperator.graphic_elements.env[2].value | string | `"3"` |  |
| nimOperator.graphic_elements.env[3].name | string | `"NIM_TRITON_MAX_BATCH_SIZE"` |  |
| nimOperator.graphic_elements.env[3].value | string | `"32"` |  |
| nimOperator.graphic_elements.env[4].name | string | `"NIM_TRITON_CUDA_MEMORY_POOL_MB"` |  |
| nimOperator.graphic_elements.env[4].value | string | `"2048"` |  |
| nimOperator.graphic_elements.env[5].name | string | `"OMP_NUM_THREADS"` |  |
| nimOperator.graphic_elements.env[5].value | string | `"1"` |  |
| nimOperator.graphic_elements.expose.service.grpcPort | int | `8001` |  |
| nimOperator.graphic_elements.expose.service.port | int | `8000` |  |
| nimOperator.graphic_elements.expose.service.type | string | `"ClusterIP"` |  |
| nimOperator.graphic_elements.image.pullPolicy | string | `"IfNotPresent"` |  |
| nimOperator.graphic_elements.image.pullSecrets[0] | string | `"ngc-secret"` |  |
| nimOperator.graphic_elements.image.repository | string | `"nvcr.io/nim/nvidia/nemoretriever-graphic-elements-v1"` |  |
| nimOperator.graphic_elements.image.tag | string | `"1.6.0"` |  |
| nimOperator.graphic_elements.replicas | int | `1` |  |
| nimOperator.graphic_elements.resources.limits."nvidia.com/gpu" | int | `1` |  |
| nimOperator.graphic_elements.storage.pvc.create | bool | `true` |  |
| nimOperator.graphic_elements.storage.pvc.size | string | `"25Gi"` |  |
| nimOperator.graphic_elements.storage.pvc.volumeAccessMode | string | `"ReadWriteMany"` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.authSecret | string | `"ngc-api"` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.enabled | bool | `false` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.env | list | `[]` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.expose.service.grpcPort | int | `8001` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.expose.service.port | int | `8000` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.expose.service.type | string | `"ClusterIP"` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.image.pullPolicy | string | `"IfNotPresent"` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.image.pullSecrets[0] | string | `"ngc-secret"` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.image.repository | string | `"nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2"` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.image.tag | string | `"1.8.0"` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.replicas | int | `1` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.resources.limits."nvidia.com/gpu" | int | `1` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.storage.pvc.create | bool | `true` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.storage.pvc.size | string | `"50Gi"` |  |
| nimOperator.llama_3_2_nv_rerankqa_1b_v2.storage.pvc.volumeAccessMode | string | `"ReadWriteMany"` |  |
| nimOperator.nemoretriever_ocr_v1.authSecret | string | `"ngc-api"` |  |
| nimOperator.nemoretriever_ocr_v1.enabled | bool | `true` |  |
| nimOperator.nemoretriever_ocr_v1.env[0].name | string | `"OMP_NUM_THREADS"` |  |
| nimOperator.nemoretriever_ocr_v1.env[0].value | string | `"8"` |  |
| nimOperator.nemoretriever_ocr_v1.env[1].name | string | `"NIM_HTTP_API_PORT"` |  |
| nimOperator.nemoretriever_ocr_v1.env[1].value | string | `"8000"` |  |
| nimOperator.nemoretriever_ocr_v1.env[2].name | string | `"NIM_TRITON_LOG_VERBOSE"` |  |
| nimOperator.nemoretriever_ocr_v1.env[2].value | string | `"1"` |  |
| nimOperator.nemoretriever_ocr_v1.env[3].name | string | `"NIM_TRITON_RATE_LIMIT"` |  |
| nimOperator.nemoretriever_ocr_v1.env[3].value | string | `"3"` |  |
| nimOperator.nemoretriever_ocr_v1.env[4].name | string | `"NIM_TRITON_CUDA_MEMORY_POOL_MB"` |  |
| nimOperator.nemoretriever_ocr_v1.env[4].value | string | `"8192"` |  |
| nimOperator.nemoretriever_ocr_v1.env[5].name | string | `"NIM_TRITON_MAX_BATCH_SIZE"` |  |
| nimOperator.nemoretriever_ocr_v1.env[5].value | string | `"32"` |  |
| nimOperator.nemoretriever_ocr_v1.env[6].name | string | `"NIM_TRITON_ENABLE_MODEL_CONTROL"` |  |
| nimOperator.nemoretriever_ocr_v1.env[6].value | string | `"1"` |  |
| nimOperator.nemoretriever_ocr_v1.expose.service.grpcPort | int | `8001` |  |
| nimOperator.nemoretriever_ocr_v1.expose.service.port | int | `8000` |  |
| nimOperator.nemoretriever_ocr_v1.expose.service.type | string | `"ClusterIP"` |  |
| nimOperator.nemoretriever_ocr_v1.image.pullPolicy | string | `"IfNotPresent"` |  |
| nimOperator.nemoretriever_ocr_v1.image.pullSecrets[0] | string | `"ngc-secret"` |  |
| nimOperator.nemoretriever_ocr_v1.image.repository | string | `"nvcr.io/nvidia/nemo-microservices/nemoretriever-ocr-v1"` |  |
| nimOperator.nemoretriever_ocr_v1.image.tag | string | `"1.1.0"` |  |
| nimOperator.nemoretriever_ocr_v1.replicas | int | `1` |  |
| nimOperator.nemoretriever_ocr_v1.resources.limits."nvidia.com/gpu" | int | `1` |  |
| nimOperator.nemoretriever_ocr_v1.storage.pvc.create | bool | `true` |  |
| nimOperator.nemoretriever_ocr_v1.storage.pvc.size | string | `"25Gi"` |  |
| nimOperator.nemoretriever_ocr_v1.storage.pvc.volumeAccessMode | string | `"ReadWriteMany"` |  |
| nimOperator.nemotron_nano_12b_v2_vl.authSecret | string | `"ngc-api"` |  |
| nimOperator.nemotron_nano_12b_v2_vl.enabled | bool | `false` |  |
| nimOperator.nemotron_nano_12b_v2_vl.expose.service.grpcPort | int | `8001` |  |
| nimOperator.nemotron_nano_12b_v2_vl.expose.service.port | int | `8000` |  |
| nimOperator.nemotron_nano_12b_v2_vl.expose.service.type | string | `"ClusterIP"` |  |
| nimOperator.nemotron_nano_12b_v2_vl.image.pullPolicy | string | `"IfNotPresent"` |  |
| nimOperator.nemotron_nano_12b_v2_vl.image.pullSecrets[0] | string | `"ngc-secret"` |  |
| nimOperator.nemotron_nano_12b_v2_vl.image.repository | string | `"nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl"` |  |
| nimOperator.nemotron_nano_12b_v2_vl.image.tag | string | `"1.5.0"` |  |
| nimOperator.nemotron_nano_12b_v2_vl.replicas | int | `1` |  |
| nimOperator.nemotron_nano_12b_v2_vl.resources.limits."nvidia.com/gpu" | int | `1` |  |
| nimOperator.nemotron_nano_12b_v2_vl.storage.pvc.create | bool | `true` |  |
| nimOperator.nemotron_nano_12b_v2_vl.storage.pvc.size | string | `"300Gi"` |  |
| nimOperator.nemotron_nano_12b_v2_vl.storage.pvc.volumeAccessMode | string | `"ReadWriteMany"` |  |
| nimOperator.nimCache.pvc.create | bool | `true` |  |
| nimOperator.nimCache.pvc.size | string | `"25Gi"` |  |
| nimOperator.nimCache.pvc.storageClass | string | `"default"` |  |
| nimOperator.nimCache.pvc.volumeAccessMode | string | `"ReadWriteMany"` |  |
| nimOperator.nimService.namespaces | list | `[]` |  |
| nimOperator.nimService.resources | object | `{}` |  |
| nimOperator.paddleocr.authSecret | string | `"ngc-api"` |  |
| nimOperator.paddleocr.enabled | bool | `false` |  |
| nimOperator.paddleocr.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nimOperator.paddleocr.env[0].value | string | `"8000"` |  |
| nimOperator.paddleocr.env[1].name | string | `"NIM_TRITON_LOG_VERBOSE"` |  |
| nimOperator.paddleocr.env[1].value | string | `"1"` |  |
| nimOperator.paddleocr.env[2].name | string | `"NIM_TRITON_RATE_LIMIT"` |  |
| nimOperator.paddleocr.env[2].value | string | `"3"` |  |
| nimOperator.paddleocr.env[3].name | string | `"NIM_TRITON_CUDA_MEMORY_POOL_MB"` |  |
| nimOperator.paddleocr.env[3].value | string | `"3072"` |  |
| nimOperator.paddleocr.env[4].name | string | `"NIM_TRITON_CPU_THREADS_PRE_PROCESSOR"` |  |
| nimOperator.paddleocr.env[4].value | string | `"2"` |  |
| nimOperator.paddleocr.env[5].name | string | `"OMP_NUM_THREADS"` |  |
| nimOperator.paddleocr.env[5].value | string | `"8"` |  |
| nimOperator.paddleocr.env[6].name | string | `"NIM_TRITON_CPU_THREADS_POST_PROCESSOR"` |  |
| nimOperator.paddleocr.env[6].value | string | `"1"` |  |
| nimOperator.paddleocr.expose.service.grpcPort | int | `8001` |  |
| nimOperator.paddleocr.expose.service.port | int | `8000` |  |
| nimOperator.paddleocr.expose.service.type | string | `"ClusterIP"` |  |
| nimOperator.paddleocr.image.pullPolicy | string | `"IfNotPresent"` |  |
| nimOperator.paddleocr.image.pullSecrets[0] | string | `"ngc-secret"` |  |
| nimOperator.paddleocr.image.repository | string | `"nvcr.io/nim/baidu/paddleocr"` |  |
| nimOperator.paddleocr.image.tag | string | `"1.5.0"` |  |
| nimOperator.paddleocr.replicas | int | `1` |  |
| nimOperator.paddleocr.resources.limits."nvidia.com/gpu" | int | `1` |  |
| nimOperator.paddleocr.storage.pvc.create | bool | `true` |  |
| nimOperator.paddleocr.storage.pvc.size | string | `"25Gi"` |  |
| nimOperator.paddleocr.storage.pvc.volumeAccessMode | string | `"ReadWriteMany"` |  |
| nimOperator.page_elements.authSecret | string | `"ngc-api"` |  |
| nimOperator.page_elements.enabled | bool | `true` |  |
| nimOperator.page_elements.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nimOperator.page_elements.env[0].value | string | `"8000"` |  |
| nimOperator.page_elements.env[10].name | string | `"NIM_OTEL_SERVICE_NAME"` |  |
| nimOperator.page_elements.env[10].value | string | `"page-elements"` |  |
| nimOperator.page_elements.env[11].name | string | `"NIM_OTEL_TRACES_EXPORTER"` |  |
| nimOperator.page_elements.env[11].value | string | `"otlp"` |  |
| nimOperator.page_elements.env[12].name | string | `"NIM_OTEL_METRICS_EXPORTER"` |  |
| nimOperator.page_elements.env[12].value | string | `"console"` |  |
| nimOperator.page_elements.env[13].name | string | `"NIM_OTEL_EXPORTER_OTLP_ENDPOINT"` |  |
| nimOperator.page_elements.env[13].value | string | `"http://otel-collector:4318"` |  |
| nimOperator.page_elements.env[14].name | string | `"TRITON_OTEL_URL"` |  |
| nimOperator.page_elements.env[14].value | string | `"http://otel-collector:4318/v1/traces"` |  |
| nimOperator.page_elements.env[15].name | string | `"TRITON_OTEL_RATE"` |  |
| nimOperator.page_elements.env[15].value | string | `"1"` |  |
| nimOperator.page_elements.env[1].name | string | `"NIM_TRITON_LOG_VERBOSE"` |  |
| nimOperator.page_elements.env[1].value | string | `"1"` |  |
| nimOperator.page_elements.env[2].name | string | `"NIM_TRITON_RATE_LIMIT"` |  |
| nimOperator.page_elements.env[2].value | string | `"3"` |  |
| nimOperator.page_elements.env[3].name | string | `"NIM_TRITON_MAX_BATCH_SIZE"` |  |
| nimOperator.page_elements.env[3].value | string | `"32"` |  |
| nimOperator.page_elements.env[4].name | string | `"NIM_TRITON_CUDA_MEMORY_POOL_MB"` |  |
| nimOperator.page_elements.env[4].value | string | `"2048"` |  |
| nimOperator.page_elements.env[5].name | string | `"NIM_TRITON_CPU_THREADS_PRE_PROCESSOR"` |  |
| nimOperator.page_elements.env[5].value | string | `"2"` |  |
| nimOperator.page_elements.env[6].name | string | `"OMP_NUM_THREADS"` |  |
| nimOperator.page_elements.env[6].value | string | `"2"` |  |
| nimOperator.page_elements.env[7].name | string | `"NIM_TRITON_CPU_THREADS_PRE_PROCESSOR"` |  |
| nimOperator.page_elements.env[7].value | string | `"2"` |  |
| nimOperator.page_elements.env[8].name | string | `"NIM_TRITON_CPU_THREADS_POST_PROCESSOR"` |  |
| nimOperator.page_elements.env[8].value | string | `"1"` |  |
| nimOperator.page_elements.env[9].name | string | `"NIM_ENABLE_OTEL"` |  |
| nimOperator.page_elements.env[9].value | string | `"true"` |  |
| nimOperator.page_elements.expose.service.grpcPort | int | `8001` |  |
| nimOperator.page_elements.expose.service.port | int | `8000` |  |
| nimOperator.page_elements.expose.service.type | string | `"ClusterIP"` |  |
| nimOperator.page_elements.image.pullPolicy | string | `"IfNotPresent"` |  |
| nimOperator.page_elements.image.pullSecrets[0] | string | `"ngc-secret"` |  |
| nimOperator.page_elements.image.repository | string | `"nvcr.io/nim/nvidia/nemoretriever-page-elements-v3"` |  |
| nimOperator.page_elements.image.tag | string | `"1.7.0"` |  |
| nimOperator.page_elements.replicas | int | `1` |  |
| nimOperator.page_elements.resources.limits."nvidia.com/gpu" | int | `1` |  |
| nimOperator.page_elements.storage.pvc.create | bool | `true` |  |
| nimOperator.page_elements.storage.pvc.size | string | `"25Gi"` |  |
| nimOperator.page_elements.storage.pvc.volumeAccessMode | string | `"ReadWriteMany"` |  |
| nimOperator.table_structure.authSecret | string | `"ngc-api"` |  |
| nimOperator.table_structure.enabled | bool | `true` |  |
| nimOperator.table_structure.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nimOperator.table_structure.env[0].value | string | `"8000"` |  |
| nimOperator.table_structure.env[1].name | string | `"NIM_TRITON_LOG_VERBOSE"` |  |
| nimOperator.table_structure.env[1].value | string | `"1"` |  |
| nimOperator.table_structure.env[2].name | string | `"NIM_TRITON_RATE_LIMIT"` |  |
| nimOperator.table_structure.env[2].value | string | `"3"` |  |
| nimOperator.table_structure.env[3].name | string | `"NIM_TRITON_MAX_BATCH_SIZE"` |  |
| nimOperator.table_structure.env[3].value | string | `"32"` |  |
| nimOperator.table_structure.env[4].name | string | `"NIM_TRITON_CUDA_MEMORY_POOL_MB"` |  |
| nimOperator.table_structure.env[4].value | string | `"3072"` |  |
| nimOperator.table_structure.env[5].name | string | `"OMP_NUM_THREADS"` |  |
| nimOperator.table_structure.env[5].value | string | `"8"` |  |
| nimOperator.table_structure.expose.service.grpcPort | int | `8001` |  |
| nimOperator.table_structure.expose.service.port | int | `8000` |  |
| nimOperator.table_structure.expose.service.type | string | `"ClusterIP"` |  |
| nimOperator.table_structure.image.pullPolicy | string | `"IfNotPresent"` |  |
| nimOperator.table_structure.image.pullSecrets[0] | string | `"ngc-secret"` |  |
| nimOperator.table_structure.image.repository | string | `"nvcr.io/nim/nvidia/nemoretriever-table-structure-v1"` |  |
| nimOperator.table_structure.image.tag | string | `"1.6.0"` |  |
| nimOperator.table_structure.replicas | int | `1` |  |
| nimOperator.table_structure.resources.limits."nvidia.com/gpu" | int | `1` |  |
| nimOperator.table_structure.storage.pvc.create | bool | `true` |  |
| nimOperator.table_structure.storage.pvc.size | string | `"25Gi"` |  |
| nimOperator.table_structure.storage.pvc.volumeAccessMode | string | `"ReadWriteMany"` |  |
| nodeSelector | object | `{}` |  |
| opentelemetry-collector.config.exporters.debug.verbosity | string | `"detailed"` |  |
| opentelemetry-collector.config.exporters.zipkin.endpoint | string | `"http://nv-ingest-zipkin:9411/api/v2/spans"` |  |
| opentelemetry-collector.config.extensions.health_check | object | `{}` |  |
| opentelemetry-collector.config.extensions.zpages.endpoint | string | `"0.0.0.0:55679"` |  |
| opentelemetry-collector.config.processors.batch | object | `{}` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].name | string | `"drop_noisy_traces_url"` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].string_attribute.enabled_regex_matching | bool | `true` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].string_attribute.invert_match | bool | `true` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].string_attribute.key | string | `"http.target"` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].string_attribute.values[0] | string | `"/health"` |  |
| opentelemetry-collector.config.processors.tail_sampling.policies[0].type | string | `"string_attribute"` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].context | string | `"span"` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].statements[0] | string | `"set(status.code, 1) where attributes[\"http.path\"] == \"/health\""` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].statements[1] | string | `"replace_match(attributes[\"http.route\"], \"/v1\", attributes[\"http.target\"]) where attributes[\"http.target\"] != nil"` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].statements[2] | string | `"replace_pattern(name, \"/v1\", attributes[\"http.route\"]) where attributes[\"http.route\"] != nil"` |  |
| opentelemetry-collector.config.processors.transform.trace_statements[0].statements[3] | string | `"set(name, Concat([name, attributes[\"http.url\"]], \" \")) where name == \"POST\""` |  |
| opentelemetry-collector.config.receivers.otlp.protocols.grpc.endpoint | string | `"${env:MY_POD_IP}:4317"` |  |
| opentelemetry-collector.config.receivers.otlp.protocols.http.cors.allowed_origins[0] | string | `"*"` |  |
| opentelemetry-collector.config.receivers.otlp.protocols.http.endpoint | string | `"${env:MY_POD_IP}:4318"` |  |
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
| opentelemetry-collector.config.service.pipelines.traces.processors[0] | string | `"batch"` |  |
| opentelemetry-collector.config.service.pipelines.traces.processors[1] | string | `"tail_sampling"` |  |
| opentelemetry-collector.config.service.pipelines.traces.processors[2] | string | `"transform"` |  |
| opentelemetry-collector.config.service.pipelines.traces.receivers[0] | string | `"otlp"` |  |
| opentelemetry-collector.image.repository | string | `"otel/opentelemetry-collector-contrib"` |  |
| opentelemetry-collector.image.tag | string | `"0.140.0"` |  |
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
| podAnnotations."traffic.sidecar.istio.io/excludeOutboundPorts" | string | `"8007"` |  |
| podLabels | object | `{}` |  |
| podSecurityContext.fsGroup | int | `1000` |  |
| prometheus.alertmanager.enabled | bool | `false` |  |
| prometheus.enabled | bool | `false` |  |
| prometheus.server.enabled | bool | `false` |  |
| readinessProbe.enabled | bool | `false` |  |
| readinessProbe.failureThreshold | int | `220` |  |
| readinessProbe.httpGet.path | string | `"/v1/health/ready"` |  |
| readinessProbe.httpGet.port | string | `"http"` |  |
| readinessProbe.initialDelaySeconds | int | `120` |  |
| readinessProbe.periodSeconds | int | `30` |  |
| readinessProbe.successThreshold | int | `1` |  |
| readinessProbe.timeoutSeconds | int | `10` |  |
| redis.auth.enabled | bool | `false` |  |
| redis.image.repository | string | `"redis"` |  |
| redis.image.tag | string | `"8.2.3"` |  |
| redis.master.configmap | string | `"protected-mode no"` |  |
| redis.master.persistence.size | string | `"50Gi"` |  |
| redis.master.resources.limits.memory | string | `"12Gi"` |  |
| redis.master.resources.requests.memory | string | `"6Gi"` |  |
| redis.replica.persistence.size | string | `"50Gi"` |  |
| redis.replica.replicaCount | int | `1` |  |
| redis.replica.resources.limits.memory | string | `"12Gi"` |  |
| redis.replica.resources.requests.memory | string | `"6Gi"` |  |
| redisDeployed | bool | `true` |  |
| replicaCount | int | `1` |  |
| resources.limits.cpu | string | `"48000m"` |  |
| resources.limits.memory | string | `"200Gi"` |  |
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
| tmpDirSize | string | `"50Gi"` |  |
| tolerations | list | `[]` |  |
| zipkin.image.repository | string | `"ghcr.io/openzipkin/alpine"` |  |
| zipkin.image.tag | string | `"3.21.3"` |  |
| zipkin.resources.limits.cpu | string | `"500m"` |  |
| zipkin.resources.limits.memory | string | `"4.5Gi"` |  |
| zipkin.resources.requests.cpu | string | `"100m"` |  |
| zipkin.resources.requests.memory | string | `"2.5Gi"` |  |
| zipkin.zipkin.extraEnv.JAVA_OPTS | string | `"-Xms2g -Xmx4g -XX:+ExitOnOutOfMemoryError"` |  |
| zipkinDeployed | bool | `true` |  |
