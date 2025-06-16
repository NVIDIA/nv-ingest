# NV-Ingest Helm Charts

This documentation contains documentation for the NV-Ingest Helm charts.

> [!Note]
> NV-Ingest is also known as NVIDIA Ingest and NeMo Retriever extraction.

## Prerequisites

Before you install the Helm charts, be sure you meet the hardware and software prerequisites. Refer to the [supported configurations](https://github.com/NVIDIA/nv-ingest?tab=readme-ov-file#hardware).

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
    https://helm.ngc.nvidia.com/nvidia/nemo-microservices/charts/nv-ingest-25.6.2.tgz \
    -n ${NAMESPACE} \
    --username '$oauthtoken' \
    --password "${NGC_API_KEY}" \
    --set ngcImagePullSecret.create=true \
    --set ngcImagePullSecret.password="${NGC_API_KEY}" \
    --set ngcApiSecret.create=true \
    --set ngcApiSecret.password="${NGC_API_KEY}" \
    --set image.repository="nvcr.io/nvidia/nemo-microservices/nv-ingest" \
    --set image.tag="25.6.2"
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

pip install nv-ingest-client==25.6.2
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
nv-ingest-paddle-0
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
| alias:baidu-nim | paddleocr-nim(nvidia-nim-paddleocr) | 1.3.0 |
| alias:nim | nim-vlm-image-captioning(nim-vlm) | 1.2.1 |
| alias:nim | nim-vlm-text-extraction(nim-vlm) | 1.2.1 |
| alias:nvidia-nim | nvidia-nim-llama-32-nv-embedqa-1b-v2 | 1.6.0 |
| alias:nvidia-nim | llama-32-nv-rerankqa-1b-v2(nvidia-nim-llama-32-nv-rerankqa-1b-v2) | 1.5.0 |
| alias:nvidia-nim | nemoretriever-graphic-elements-v1(nvidia-nim-nemoretriever-graphic-elements-v1) | 1.3.0 |
| alias:nvidia-nim | nemoretriever-page-elements-v2(nvidia-nim-nemoretriever-page-elements-v2) | 1.3.0 |
| alias:nvidia-nim | nemoretriever-table-structure-v1(nvidia-nim-nemoretriever-table-structure-v1) | 1.3.0 |
| alias:nvidia-nim | text-embedding-nim(nvidia-nim-nv-embedqa-e5-v5) | 1.6.0 |
| alias:nvidia-nim | riva-nim | 1.0.0 |
| https://open-telemetry.github.io/opentelemetry-helm-charts | opentelemetry-collector | 0.78.1 |
| https://prometheus-community.github.io/helm-charts | prometheus | 27.12.1 |
| https://zilliztech.github.io/milvus-helm | milvus | 4.1.11 |
| https://zipkin.io/zipkin-helm | zipkin | 0.1.2 |
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
| envVars.EMBEDDING_NIM_ENDPOINT | string | `"http://nv-ingest-embedqa:8000/v1"` |  |
| envVars.EMBEDDING_NIM_MODEL_NAME | string | `"nvidia/llama-3.2-nv-embedqa-1b-v2"` |  |
| envVars.INGEST_DYNAMIC_MEMORY_THRESHOLD | float | `0.8` |  |
| envVars.INGEST_EDGE_BUFFER_SIZE | int | `64` |  |
| envVars.INGEST_LOG_LEVEL | string | `"DEFAULT"` |  |
| envVars.INSTALL_AUDIO_EXTRACTION_DEPS | string | `"true"` |  |
| envVars.MAX_INGEST_PROCESS_WORKERS | int | `16` |  |
| envVars.MESSAGE_CLIENT_HOST | string | `"nv-ingest-redis-master"` |  |
| envVars.MESSAGE_CLIENT_PORT | string | `"6379"` |  |
| envVars.MESSAGE_CLIENT_TYPE | string | `"redis"` |  |
| envVars.MILVUS_ENDPOINT | string | `"http://nv-ingest-milvus:19530"` |  |
| envVars.MINIO_BUCKET | string | `"nv-ingest"` |  |
| envVars.MINIO_INTERNAL_ADDRESS | string | `"nv-ingest-minio:9000"` |  |
| envVars.MINIO_PUBLIC_ADDRESS | string | `"http://localhost:9000"` |  |
| envVars.MODEL_PREDOWNLOAD_PATH | string | `"/workspace/models/"` |  |
| envVars.NEMORETRIEVER_PARSE_HTTP_ENDPOINT | string | `"http://nim-vlm-text-extraction-nemoretriever-parse:8000/v1/chat/completions"` |  |
| envVars.NEMORETRIEVER_PARSE_INFER_PROTOCOL | string | `"http"` |  |
| envVars.NEMORETRIEVER_PARSE_MODEL_NAME | string | `"nvidia/nemoretriever-parse"` |  |
| envVars.NV_INGEST_DEFAULT_TIMEOUT_MS | string | `"1234"` |  |
| envVars.NV_INGEST_MAX_UTIL | int | `48` |  |
| envVars.PADDLE_GRPC_ENDPOINT | string | `"nv-ingest-paddle:8001"` |  |
| envVars.PADDLE_HTTP_ENDPOINT | string | `"http://nv-ingest-paddle:8000/v1/infer"` |  |
| envVars.PADDLE_INFER_PROTOCOL | string | `"grpc"` |  |
| envVars.REDIS_INGEST_TASK_QUEUE | string | `"ingest_task_queue"` |  |
| envVars.VLM_CAPTION_ENDPOINT | string | `"https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"` |  |
| envVars.VLM_CAPTION_MODEL_NAME | string | `"meta/llama-3.2-11b-vision-instruct"` |  |
| envVars.YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT | string | `"nemoretriever-graphic-elements-v1:8001"` |  |
| envVars.YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT | string | `"http://nemoretriever-graphic-elements-v1:8000/v1/infer"` |  |
| envVars.YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL | string | `"grpc"` |  |
| envVars.YOLOX_GRPC_ENDPOINT | string | `"nemoretriever-page-elements-v2:8001"` |  |
| envVars.YOLOX_HTTP_ENDPOINT | string | `"http://nemoretriever-page-elements-v2:8000/v1/infer"` |  |
| envVars.YOLOX_INFER_PROTOCOL | string | `"grpc"` |  |
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
| image.tag | string | `"25.6.2"` |  |
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
| llama-32-nv-rerankqa-1b-v2.autoscaling.enabled | bool | `false` |  |
| llama-32-nv-rerankqa-1b-v2.autoscaling.maxReplicas | int | `10` |  |
| llama-32-nv-rerankqa-1b-v2.autoscaling.metrics | list | `[]` |  |
| llama-32-nv-rerankqa-1b-v2.autoscaling.minReplicas | int | `1` |  |
| llama-32-nv-rerankqa-1b-v2.customArgs | list | `[]` |  |
| llama-32-nv-rerankqa-1b-v2.customCommand | list | `[]` |  |
| llama-32-nv-rerankqa-1b-v2.deployed | bool | `false` |  |
| llama-32-nv-rerankqa-1b-v2.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| llama-32-nv-rerankqa-1b-v2.env[0].value | string | `"8000"` |  |
| llama-32-nv-rerankqa-1b-v2.env[1].name | string | `"NIM_TRITON_MODEL_BATCH_SIZE"` |  |
| llama-32-nv-rerankqa-1b-v2.env[1].value | string | `"1"` |  |
| llama-32-nv-rerankqa-1b-v2.image.repository | string | `"nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2"` |  |
| llama-32-nv-rerankqa-1b-v2.image.tag | string | `"1.5.0"` |  |
| llama-32-nv-rerankqa-1b-v2.nim.grpcPort | int | `8001` |  |
| llama-32-nv-rerankqa-1b-v2.nim.logLevel | string | `"INFO"` |  |
| llama-32-nv-rerankqa-1b-v2.podSecurityContext.fsGroup | int | `1000` |  |
| llama-32-nv-rerankqa-1b-v2.podSecurityContext.runAsGroup | int | `1000` |  |
| llama-32-nv-rerankqa-1b-v2.podSecurityContext.runAsUser | int | `1000` |  |
| llama-32-nv-rerankqa-1b-v2.replicaCount | int | `1` |  |
| llama-32-nv-rerankqa-1b-v2.service.grpcPort | int | `8001` |  |
| llama-32-nv-rerankqa-1b-v2.service.httpPort | int | `8000` |  |
| llama-32-nv-rerankqa-1b-v2.service.metricsPort | int | `0` |  |
| llama-32-nv-rerankqa-1b-v2.service.name | string | `"llama-32-nv-rerankqa-1b-v2"` |  |
| llama-32-nv-rerankqa-1b-v2.service.type | string | `"ClusterIP"` |  |
| llama-32-nv-rerankqa-1b-v2.serviceAccount.create | bool | `false` |  |
| llama-32-nv-rerankqa-1b-v2.serviceAccount.name | string | `""` |  |
| llama-32-nv-rerankqa-1b-v2.statefuleSet.enabled | bool | `false` |  |
| logLevel | string | `"DEFAULT"` |  |
| milvus.cluster.enabled | bool | `false` |  |
| milvus.etcd.persistence.storageClass | string | `nil` |  |
| milvus.etcd.replicaCount | int | `1` |  |
| milvus.image.all.repository | string | `"milvusdb/milvus"` |  |
| milvus.image.all.tag | string | `"v2.5.3-gpu"` |  |
| milvus.minio.bucketName | string | `"a-bucket"` |  |
| milvus.minio.mode | string | `"standalone"` |  |
| milvus.minio.persistence.size | string | `"50Gi"` |  |
| milvus.minio.persistence.storageClass | string | `nil` |  |
| milvus.pulsar.enabled | bool | `false` |  |
| milvus.standalone.extraEnv[0].name | string | `"LOG_LEVEL"` |  |
| milvus.standalone.extraEnv[0].value | string | `"error"` |  |
| milvus.standalone.persistence.persistentVolumeClaim.size | string | `"50Gi"` |  |
| milvus.standalone.persistence.persistentVolumeClaim.storageClass | string | `nil` |  |
| milvus.standalone.resources.limits."nvidia.com/gpu" | int | `1` |  |
| milvusDeployed | bool | `true` |  |
| nameOverride | string | `""` |  |
| nemo.groupID | string | `"1000"` |  |
| nemo.userID | string | `"1000"` |  |
| nemoretriever-graphic-elements-v1.autoscaling.enabled | bool | `false` |  |
| nemoretriever-graphic-elements-v1.autoscaling.maxReplicas | int | `10` |  |
| nemoretriever-graphic-elements-v1.autoscaling.metrics | list | `[]` |  |
| nemoretriever-graphic-elements-v1.autoscaling.minReplicas | int | `1` |  |
| nemoretriever-graphic-elements-v1.customArgs | list | `[]` |  |
| nemoretriever-graphic-elements-v1.customCommand | list | `[]` |  |
| nemoretriever-graphic-elements-v1.deployed | bool | `true` |  |
| nemoretriever-graphic-elements-v1.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nemoretriever-graphic-elements-v1.env[0].value | string | `"8000"` |  |
| nemoretriever-graphic-elements-v1.env[1].name | string | `"NIM_TRITON_MODEL_BATCH_SIZE"` |  |
| nemoretriever-graphic-elements-v1.env[1].value | string | `"1"` |  |
| nemoretriever-graphic-elements-v1.image.repository | string | `"nvcr.io/nim/nvidia/nemoretriever-graphic-elements-v1"` |  |
| nemoretriever-graphic-elements-v1.image.tag | string | `"1.3.0"` |  |
| nemoretriever-graphic-elements-v1.nim.grpcPort | int | `8001` |  |
| nemoretriever-graphic-elements-v1.nim.logLevel | string | `"INFO"` |  |
| nemoretriever-graphic-elements-v1.podSecurityContext.fsGroup | int | `1000` |  |
| nemoretriever-graphic-elements-v1.podSecurityContext.runAsGroup | int | `1000` |  |
| nemoretriever-graphic-elements-v1.podSecurityContext.runAsUser | int | `1000` |  |
| nemoretriever-graphic-elements-v1.replicaCount | int | `1` |  |
| nemoretriever-graphic-elements-v1.service.grpcPort | int | `8001` |  |
| nemoretriever-graphic-elements-v1.service.httpPort | int | `8000` |  |
| nemoretriever-graphic-elements-v1.service.metricsPort | int | `0` |  |
| nemoretriever-graphic-elements-v1.service.name | string | `"nemoretriever-graphic-elements-v1"` |  |
| nemoretriever-graphic-elements-v1.service.type | string | `"ClusterIP"` |  |
| nemoretriever-graphic-elements-v1.serviceAccount.create | bool | `false` |  |
| nemoretriever-graphic-elements-v1.serviceAccount.name | string | `""` |  |
| nemoretriever-graphic-elements-v1.statefuleSet.enabled | bool | `false` |  |
| nemoretriever-page-elements-v2.autoscaling.enabled | bool | `false` |  |
| nemoretriever-page-elements-v2.autoscaling.maxReplicas | int | `10` |  |
| nemoretriever-page-elements-v2.autoscaling.metrics | list | `[]` |  |
| nemoretriever-page-elements-v2.autoscaling.minReplicas | int | `1` |  |
| nemoretriever-page-elements-v2.deployed | bool | `true` |  |
| nemoretriever-page-elements-v2.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nemoretriever-page-elements-v2.env[0].value | string | `"8000"` |  |
| nemoretriever-page-elements-v2.env[1].name | string | `"NIM_TRITON_MODEL_BATCH_SIZE"` |  |
| nemoretriever-page-elements-v2.env[1].value | string | `"1"` |  |
| nemoretriever-page-elements-v2.image.pullPolicy | string | `"IfNotPresent"` |  |
| nemoretriever-page-elements-v2.image.repository | string | `"nvcr.io/nim/nvidia/nemoretriever-page-elements-v2"` |  |
| nemoretriever-page-elements-v2.image.tag | string | `"1.3.0"` |  |
| nemoretriever-page-elements-v2.nim.grpcPort | int | `8001` |  |
| nemoretriever-page-elements-v2.nim.logLevel | string | `"INFO"` |  |
| nemoretriever-page-elements-v2.podSecurityContext.fsGroup | int | `1000` |  |
| nemoretriever-page-elements-v2.podSecurityContext.runAsGroup | int | `1000` |  |
| nemoretriever-page-elements-v2.podSecurityContext.runAsUser | int | `1000` |  |
| nemoretriever-page-elements-v2.replicaCount | int | `1` |  |
| nemoretriever-page-elements-v2.service.grpcPort | int | `8001` |  |
| nemoretriever-page-elements-v2.service.httpPort | int | `8000` |  |
| nemoretriever-page-elements-v2.service.metricsPort | int | `0` |  |
| nemoretriever-page-elements-v2.service.name | string | `"nemoretriever-page-elements-v2"` |  |
| nemoretriever-page-elements-v2.service.type | string | `"ClusterIP"` |  |
| nemoretriever-page-elements-v2.serviceAccount.create | bool | `false` |  |
| nemoretriever-page-elements-v2.serviceAccount.name | string | `""` |  |
| nemoretriever-page-elements-v2.statefuleSet.enabled | bool | `false` |  |
| nemoretriever-table-structure-v1.autoscaling.enabled | bool | `false` |  |
| nemoretriever-table-structure-v1.autoscaling.maxReplicas | int | `10` |  |
| nemoretriever-table-structure-v1.autoscaling.metrics | list | `[]` |  |
| nemoretriever-table-structure-v1.autoscaling.minReplicas | int | `1` |  |
| nemoretriever-table-structure-v1.customArgs | list | `[]` |  |
| nemoretriever-table-structure-v1.customCommand | list | `[]` |  |
| nemoretriever-table-structure-v1.deployed | bool | `true` |  |
| nemoretriever-table-structure-v1.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nemoretriever-table-structure-v1.env[0].value | string | `"8000"` |  |
| nemoretriever-table-structure-v1.env[1].name | string | `"NIM_TRITON_MODEL_BATCH_SIZE"` |  |
| nemoretriever-table-structure-v1.env[1].value | string | `"1"` |  |
| nemoretriever-table-structure-v1.image.repository | string | `"nvcr.io/nim/nvidia/nemoretriever-table-structure-v1"` |  |
| nemoretriever-table-structure-v1.image.tag | string | `"1.3.0"` |  |
| nemoretriever-table-structure-v1.nim.grpcPort | int | `8001` |  |
| nemoretriever-table-structure-v1.nim.logLevel | string | `"INFO"` |  |
| nemoretriever-table-structure-v1.podSecurityContext.fsGroup | int | `1000` |  |
| nemoretriever-table-structure-v1.podSecurityContext.runAsGroup | int | `1000` |  |
| nemoretriever-table-structure-v1.podSecurityContext.runAsUser | int | `1000` |  |
| nemoretriever-table-structure-v1.replicaCount | int | `1` |  |
| nemoretriever-table-structure-v1.service.grpcPort | int | `8001` |  |
| nemoretriever-table-structure-v1.service.httpPort | int | `8000` |  |
| nemoretriever-table-structure-v1.service.metricsPort | int | `0` |  |
| nemoretriever-table-structure-v1.service.name | string | `"nemoretriever-table-structure-v1"` |  |
| nemoretriever-table-structure-v1.service.type | string | `"ClusterIP"` |  |
| nemoretriever-table-structure-v1.serviceAccount.create | bool | `false` |  |
| nemoretriever-table-structure-v1.serviceAccount.name | string | `""` |  |
| nemoretriever-table-structure-v1.statefuleSet.enabled | bool | `false` |  |
| ngcApiSecret.create | bool | `false` |  |
| ngcApiSecret.password | string | `""` |  |
| ngcImagePullSecret.create | bool | `false` |  |
| ngcImagePullSecret.name | string | `"ngcImagePullSecret"` |  |
| ngcImagePullSecret.password | string | `""` |  |
| ngcImagePullSecret.registry | string | `"nvcr.io"` |  |
| ngcImagePullSecret.username | string | `"$oauthtoken"` |  |
| nim-vlm-image-captioning.customArgs | list | `[]` |  |
| nim-vlm-image-captioning.customCommand | list | `[]` |  |
| nim-vlm-image-captioning.deployed | bool | `false` |  |
| nim-vlm-image-captioning.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nim-vlm-image-captioning.env[0].value | string | `"8000"` |  |
| nim-vlm-image-captioning.env[1].name | string | `"NIM_TRITON_MODEL_BATCH_SIZE"` |  |
| nim-vlm-image-captioning.env[1].value | string | `"1"` |  |
| nim-vlm-image-captioning.fullnameOverride | string | `"nim-vlm-image-captioning"` |  |
| nim-vlm-image-captioning.image.repository | string | `"nvcr.io/nim/meta/llama-3.2-11b-vision-instruct"` |  |
| nim-vlm-image-captioning.image.tag | string | `"1.1.1"` |  |
| nim-vlm-image-captioning.nim.grpcPort | int | `8001` |  |
| nim-vlm-image-captioning.nim.logLevel | string | `"INFO"` |  |
| nim-vlm-image-captioning.podSecurityContext.fsGroup | int | `1000` |  |
| nim-vlm-image-captioning.podSecurityContext.runAsGroup | int | `1000` |  |
| nim-vlm-image-captioning.podSecurityContext.runAsUser | int | `1000` |  |
| nim-vlm-image-captioning.replicaCount | int | `1` |  |
| nim-vlm-image-captioning.service.grpcPort | int | `8001` |  |
| nim-vlm-image-captioning.service.httpPort | int | `8000` |  |
| nim-vlm-image-captioning.service.metricsPort | int | `0` |  |
| nim-vlm-image-captioning.service.name | string | `"nim-vlm-image-captioning"` |  |
| nim-vlm-image-captioning.service.type | string | `"ClusterIP"` |  |
| nim-vlm-text-extraction.customArgs | list | `[]` |  |
| nim-vlm-text-extraction.customCommand | list | `[]` |  |
| nim-vlm-text-extraction.deployed | bool | `false` |  |
| nim-vlm-text-extraction.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nim-vlm-text-extraction.env[0].value | string | `"8000"` |  |
| nim-vlm-text-extraction.env[1].name | string | `"NIM_TRITON_MODEL_BATCH_SIZE"` |  |
| nim-vlm-text-extraction.env[1].value | string | `"1"` |  |
| nim-vlm-text-extraction.fullnameOverride | string | `"nim-vlm-text-extraction-nemoretriever-parse"` |  |
| nim-vlm-text-extraction.image.repository | string | `"nvcr.io/nvidia/nemo-microservices/nemoretriever-parse"` |  |
| nim-vlm-text-extraction.image.tag | string | `"1.2.0ea"` |  |
| nim-vlm-text-extraction.nim.grpcPort | int | `8001` |  |
| nim-vlm-text-extraction.nim.logLevel | string | `"INFO"` |  |
| nim-vlm-text-extraction.podSecurityContext.fsGroup | int | `1000` |  |
| nim-vlm-text-extraction.podSecurityContext.runAsGroup | int | `1000` |  |
| nim-vlm-text-extraction.podSecurityContext.runAsUser | int | `1000` |  |
| nim-vlm-text-extraction.replicaCount | int | `1` |  |
| nim-vlm-text-extraction.service.grpcPort | int | `8001` |  |
| nim-vlm-text-extraction.service.httpPort | int | `8000` |  |
| nim-vlm-text-extraction.service.metricsPort | int | `0` |  |
| nim-vlm-text-extraction.service.name | string | `"nim-vlm-text-extraction-nemoretriever-parse"` |  |
| nim-vlm-text-extraction.service.type | string | `"ClusterIP"` |  |
| nodeSelector | object | `{}` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.autoscaling.enabled | bool | `false` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.autoscaling.maxReplicas | int | `10` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.autoscaling.metrics | list | `[]` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.autoscaling.minReplicas | int | `1` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.customArgs | list | `[]` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.customCommand | list | `[]` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.deployed | bool | `true` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.env[0].value | string | `"8000"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.env[1].name | string | `"NIM_TRITON_MAX_BATCH_SIZE"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.env[1].value | string | `"1"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.fullnameOverride | string | `"nv-ingest-embedqa"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.image.repository | string | `"nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.image.tag | string | `"1.6.0"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.nim.grpcPort | int | `8001` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.nim.logLevel | string | `"INFO"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.podSecurityContext.fsGroup | int | `1000` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.podSecurityContext.runAsGroup | int | `1000` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.podSecurityContext.runAsUser | int | `1000` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.replicaCount | int | `1` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.service.grpcPort | int | `8001` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.service.httpPort | int | `8000` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.service.metricsPort | int | `0` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.service.name | string | `"nv-ingest-embedqa"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.service.type | string | `"ClusterIP"` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.serviceAccount.create | bool | `false` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.serviceAccount.name | string | `""` |  |
| nvidia-nim-llama-32-nv-embedqa-1b-v2.statefuleSet.enabled | bool | `false` |  |
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
| paddleocr-nim."paddleocr-nim.deployed" | bool | `true` |  |
| paddleocr-nim.autoscaling.enabled | bool | `false` |  |
| paddleocr-nim.autoscaling.maxReplicas | int | `10` |  |
| paddleocr-nim.autoscaling.metrics | list | `[]` |  |
| paddleocr-nim.autoscaling.minReplicas | int | `1` |  |
| paddleocr-nim.customArgs | list | `[]` |  |
| paddleocr-nim.customCommand | list | `[]` |  |
| paddleocr-nim.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| paddleocr-nim.env[0].value | string | `"8000"` |  |
| paddleocr-nim.env[1].name | string | `"NIM_TRITON_MODEL_BATCH_SIZE"` |  |
| paddleocr-nim.env[1].value | string | `"1"` |  |
| paddleocr-nim.fullnameOverride | string | `"nv-ingest-paddle"` |  |
| paddleocr-nim.image.repository | string | `"nvcr.io/nim/baidu/paddleocr"` |  |
| paddleocr-nim.image.tag | string | `"1.3.0"` |  |
| paddleocr-nim.nim.grpcPort | int | `8001` |  |
| paddleocr-nim.nim.logLevel | string | `"INFO"` |  |
| paddleocr-nim.podSecurityContext.fsGroup | int | `1000` |  |
| paddleocr-nim.podSecurityContext.runAsGroup | int | `1000` |  |
| paddleocr-nim.podSecurityContext.runAsUser | int | `1000` |  |
| paddleocr-nim.replicaCount | int | `1` |  |
| paddleocr-nim.service.grpcPort | int | `8001` |  |
| paddleocr-nim.service.httpPort | int | `8000` |  |
| paddleocr-nim.service.metricsPort | int | `0` |  |
| paddleocr-nim.service.name | string | `"nv-ingest-paddle"` |  |
| paddleocr-nim.service.type | string | `"ClusterIP"` |  |
| paddleocr-nim.serviceAccount.create | bool | `false` |  |
| paddleocr-nim.serviceAccount.name | string | `""` |  |
| paddleocr-nim.statefuleSet.enabled | bool | `false` |  |
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
| riva-nim.autoscaling.enabled | bool | `false` |  |
| riva-nim.autoscaling.maxReplicas | int | `10` |  |
| riva-nim.autoscaling.metrics | list | `[]` |  |
| riva-nim.autoscaling.minReplicas | int | `1` |  |
| riva-nim.customArgs | list | `[]` |  |
| riva-nim.customCommand | list | `[]` |  |
| riva-nim.deployed | bool | `false` |  |
| riva-nim.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| riva-nim.env[0].value | string | `"9000"` |  |
| riva-nim.env[1].name | string | `"NIM_TAGS_SELECTOR"` |  |
| riva-nim.env[1].value | string | `"name=parakeet-1-1b-ctc-riva-en-us,mode=ofl"` |  |
| riva-nim.fullnameOverride | string | `"nv-ingest-riva-nim"` |  |
| riva-nim.image.repository | string | `"nvcr.io/nim/nvidia/riva-asr"` |  |
| riva-nim.image.tag | string | `"1.3.0"` |  |
| riva-nim.nim.grpcPort | int | `50051` |  |
| riva-nim.nim.logLevel | string | `"INFO"` |  |
| riva-nim.podSecurityContext.fsGroup | int | `1000` |  |
| riva-nim.podSecurityContext.runAsGroup | int | `1000` |  |
| riva-nim.podSecurityContext.runAsUser | int | `1000` |  |
| riva-nim.replicaCount | int | `1` |  |
| riva-nim.service.grpcPort | int | `50051` |  |
| riva-nim.service.httpPort | int | `9000` |  |
| riva-nim.service.metricsPort | int | `0` |  |
| riva-nim.service.name | string | `"nv-ingest-riva-nim"` |  |
| riva-nim.service.type | string | `"ClusterIP"` |  |
| riva-nim.serviceAccount.create | bool | `false` |  |
| riva-nim.serviceAccount.name | string | `""` |  |
| riva-nim.statefuleSet.enabled | bool | `false` |  |
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
| text-embedding-nim.autoscaling.enabled | bool | `false` |  |
| text-embedding-nim.autoscaling.maxReplicas | int | `10` |  |
| text-embedding-nim.autoscaling.metrics | list | `[]` |  |
| text-embedding-nim.autoscaling.minReplicas | int | `1` |  |
| text-embedding-nim.customArgs | list | `[]` |  |
| text-embedding-nim.customCommand | list | `[]` |  |
| text-embedding-nim.deployed | bool | `false` |  |
| text-embedding-nim.env[0].name | string | `"NIM_HTTP_API_PORT"` |  |
| text-embedding-nim.env[0].value | string | `"8000"` |  |
| text-embedding-nim.env[1].name | string | `"NIM_TRITON_MODEL_BATCH_SIZE"` |  |
| text-embedding-nim.env[1].value | string | `"1"` |  |
| text-embedding-nim.fullnameOverride | string | `"nv-ingest-embedqa"` |  |
| text-embedding-nim.image.repository | string | `"nvcr.io/nim/nvidia/nv-embedqa-e5-v5"` |  |
| text-embedding-nim.image.tag | string | `"1.6.0"` |  |
| text-embedding-nim.nim.grpcPort | int | `8001` |  |
| text-embedding-nim.nim.logLevel | string | `"INFO"` |  |
| text-embedding-nim.podSecurityContext.fsGroup | int | `1000` |  |
| text-embedding-nim.podSecurityContext.runAsGroup | int | `1000` |  |
| text-embedding-nim.podSecurityContext.runAsUser | int | `1000` |  |
| text-embedding-nim.replicaCount | int | `1` |  |
| text-embedding-nim.service.grpcPort | int | `8001` |  |
| text-embedding-nim.service.httpPort | int | `8000` |  |
| text-embedding-nim.service.metricsPort | int | `0` |  |
| text-embedding-nim.service.name | string | `"nv-ingest-embedqa"` |  |
| text-embedding-nim.service.type | string | `"ClusterIP"` |  |
| text-embedding-nim.serviceAccount.create | bool | `false` |  |
| text-embedding-nim.serviceAccount.name | string | `""` |  |
| text-embedding-nim.statefuleSet.enabled | bool | `false` |  |
| tmpDirSize | string | `"50Gi"` |  |
| tolerations | list | `[]` |  |
| zipkin.resources.limits.cpu | string | `"500m"` |  |
| zipkin.resources.limits.memory | string | `"4.5Gi"` |  |
| zipkin.resources.requests.cpu | string | `"100m"` |  |
| zipkin.resources.requests.memory | string | `"2.5Gi"` |  |
| zipkin.zipkin.extraEnv.JAVA_OPTS | string | `"-Xms2g -Xmx4g -XX:+ExitOnOutOfMemoryError"` |  |
| zipkinDeployed | bool | `true` |  |
