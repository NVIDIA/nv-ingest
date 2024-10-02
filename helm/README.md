# NVIDIA NVIngest

## Prerequisites

### Hardware

| GPU | Family | Memory | # of GPUs |
| ------ | ------ | ------ | ------ |
| H100 | SXM/NVLink or PCIe | 80GB | 2 |
| A100 | SXM/NVLink or PCIe | 80GB | 2 |

### Software

- Linux operating systems (Ubuntu 20.04 or later recommended)
- [Docker](https://docs.docker.com/engine/install/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA Driver >= 535)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Setup Environment

- First create your namespace

```bash
NAMESPACE=nv-ingest
kubectl create namespace ${NAMESPACE}
```

- Install the chart

```bash
helm upgrade \
    --install \
    --username '$oauthtoken' \
    --password "${NGC_API_KEY}" \
    -n ${NAMESPACE} \
    nv-ingest \
    --set imagePullSecret.create=true \
    --set imagePullSecret.password="${NGC_API_KEY}" \
    --set ngcSecret.create=true \
    --set ngcSecret.password="${NGC_API_KEY}" \
    --set image.repository="nvcr.io/ohlfw0olaadg/ea-participants/nv-ingest" \
    --set image.tag="24.08" \
    https://helm.ngc.nvidia.com/ohlfw0olaadg/ea-participants/charts/nv-ingest-0.3.5.tgz

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
  name: nvcrimagepullsecret
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: ${NGC_REGISTRY_PASSWORD}
EOF
kubectl create -n ${NAMESPACE} secret generic ngc-api --from-literal=NGC_API_KEY=${NGC_API_KEY}

```

Alternatively, you can also use an External Secret Store like Vault, the name of the secret name expected for the NGC API is `ngc-api` and the secret name expected for NVCR is `nvcrimagepullsecret`.

In this case, make sure to remove the following from your helm commmand:

```bash
    --set imagePullSecret.create=true \
    --set imagePullSecret.password="${NGC_API_KEY}" \
    --set ngcSecret.create=true \
    --set ngcSecret.password="${NGC_API_KEY}" \
```

### Minikube Setup

The PVC setup for minikube requires a little bit more configuraiton. Please follow the steps below if you are using minikube.

```bash
minikube start --driver docker --container-runtime docker --gpus all --nodes 3
minikube addons enable nvidia-gpu-device-plugin
minikube addons enable storage-provisioner-rancher
```

## Usage

Jobs are submitted via the `nv-ingest-cli` command. Installation methods differ depending on your version of nv-ingest. Please see more detailed instructions in either the <= 24.08 or > 24.08 subsections below.

### Versions <= 24.08

Nv-Ingest versions <= 24.08 exposed Redis directly to the client. This means that the setup for the `nv-ingest-cli` differs slightly than
in newer versions as a specific version of the `nv-ingest-cli` must be used and different ports must be opened.

#### Nv-Ingest CLI Installation <= 24.08

You can find the Python wheel for the `nv-ingest-cli` located in our [nv-ingest 24.08 release artifacts](https://github.com/NVIDIA/nv-ingest/releases/tag/24.08). Installation of the `nv-ingest-cli` goes as follows.

```shell
# Just to be cautious we remove any existing installation
pip uninstall nv-ingest-cli

# Download the 24.08 .whl
wget https://github.com/NVIDIA/nv-ingest/releases/download/24.08/nv_ingest_client-24.08-py3-none-any.whl

# Pip install that .whl
pip install nv_ingest_client-24.08-py3-none-any.whl
```

#### Access To Redis <= 24.08

It is recommended that the end user provide a mechanism for [`Ingress`](https://kubernetes.io/docs/concepts/services-networking/ingress/) for the Redis pod.
You can test outside of your Kuberenetes cluster by [port-forwarding](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_port-forward/) the Redis pod to your local environment.

Example:

```bash
kubectl port-forward -n ${NAMESPACE} nv-ingest-redis-master-0 6379:6379
```

#### Executing jobs <= 24.08

Here is a sample invocation of a PDF extraction task using the port forward above:

```bash
mkdir -p ./processed_docs

nv-ingest-cli \
  --doc /path/to/your/unique.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_text": true, "extract_images": true, "extract_tables": true}' \
  --client_host=localhost \
  --client_port=6379
```

### Versions > 24.08

Nv-Ingest versions <= 24.08 exposed Redis directly to the client. This means that the setup for the `nv-ingest-cli` differs slightly than
in newer versions as a specific version of the `nv-ingest-cli` must be used and different ports must be opened.

#### Nv-Ingest CLI Installation > 24.08

For versions > 24.08 we recommend building `nv-ingest-cli` from source to ensure you have the latest code. See installation [here](https://github.com/NVIDIA/nv-ingest/tree/main/client)

```bash
# Just to be cautious we remove any existing installation
pip uninstall nv-ingest-cli

# Download the 24.08 .whl
wget https://github.com/NVIDIA/nv-ingest/releases/download/24.08/nv_ingest_client-24.08-py3-none-any.whl

# Pip install that .whl
pip install nv_ingest_client-24.08-py3-none-any.whl
```

#### Rest Endpoint Ingress > 24.08

It is recommended that the end user provide a mechanism for [`Ingress`](https://kubernetes.io/docs/concepts/services-networking/ingress/) for the nv-ingest pod.
You can test outside of your Kuberenetes cluster by [port-forwarding](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_port-forward/) the nv-ingest pod to your local environment.

Example:

You can find the name of your nv-ingest pod that you want to forward traffic to by running.

```bash
kubectl get pods -n <namespace> --no-headers -o custom-columns=":metadata.name"
```

The output will look something like this with different auto generated sequences.

```
nv-ingest-674f6b7477-65nvm
nv-ingest-cached-0
nv-ingest-deplot-0
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

#### Executing jobs > 24.08

Here is a sample invocation of a PDF extraction task using the port forward above:

```bash
mkdir -p ./processed_docs

nv-ingest-cli \
  --doc /path/to/your/unique.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_text": true, "extract_images": true, "extract_tables": true}' \
  --client_host=localhost \
  --client_port=7670
```

You can also use nv-ingest's python client API to interact with the service running in the cluster. Use the same host and port as in the above nv-ingest-cli example.

## Parameters

### Deployment parameters

| Name                                | Description                                                                                                      | Value   |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------- |
| `affinity`                          | [default: {}] Affinity settings for deployment.                                                                  | `{}`    |
| `nodeSelector`                      | Sets node selectors for the NIM -- for example `nvidia.com/gpu.present: "true"`                                  | `{}`    |
| `logLevel`                          | Log level of NVIngest service. Possible values of the variable are TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL. | `DEBUG` |
| `extraEnvVarsCM`                    | [default: ""] A Config map holding Enviroment variables to include in the NVIngest containerextraEnvVarsCM: ""   | `""`    |
| `extraEnvVarsSecret`                | [default: ""] A K8S Secret to map to Enviroment variables to include in the NVIngest container                   | `""`    |
| `fullnameOverride`                  | [default: ""] A name to force the fullname of the NVIngest container to have, defaults to the Helm Release Name  | `""`    |
| `nameOverride`                      | [default: ""] A name to base the objects created by this helm chart                                              | `""`    |
| `image.repository`                  | NIM Image Repository                                                                                             | `""`    |
| `image.tag`                         | Image tag or version                                                                                             | `""`    |
| `image.pullPolicy`                  | Image pull policy                                                                                                | `""`    |
| `podAnnotations`                    | Sets additional annotations on the main deployment pods                                                          | `{}`    |
| `podLabels`                         | Specify extra labels to be add to on deployed pods.                                                              | `{}`    |
| `podSecurityContext`                | Specify privilege and access control settings for pod                                                            |         |
| `podSecurityContext.fsGroup`        | Specify file system owner group id.                                                                              | `1000`  |
| `extraVolumes`                      | Adds arbitrary additional volumes to the deployment set definition                                               | `{}`    |
| `extraVolumeMounts`                 | Specify volume mounts to the main container from `extraVolumes`                                                  | `{}`    |
| `imagePullSecrets`                  | Specify list of secret names that are needed for the main container and any init containers.                     |         |
| `containerSecurityContext`          | Sets privilege and access control settings for container (Only affects the main container, not pod-level)        | `{}`    |
| `tolerations`                       | Specify tolerations for pod assignment. Allows the scheduler to schedule pods with matching taints.              |         |
| `replicaCount`                      | The number of replicas for NVIngest when autoscaling is disabled                                                 | `1`     |
| `resources.limits."nvidia.com/gpu"` | Specify number of GPUs to present to the running service.                                                        |         |
| `resources.limits.memory`           | Specify limit for memory                                                                                         | `32Gi`  |
| `resources.requests.memory`         | Specify request for memory                                                                                       | `16Gi`  |
| `tmpDirSize`                        | Specify the amount of space to reserve for temporary storage                                                     | `8Gi`   |

### NIM Configuration

Define additional values to the dependent NIM helm charts by updating the "yolox-nim", "cached-nim", "deplot-nim", and "paddleocr-nim"
values. A sane set of configurations are already included in this value file and only the "image.repository" and "image.tag" fields are
explicitly called out here.

| Name                             | Description                                                     | Value |
| -------------------------------- | --------------------------------------------------------------- | ----- |
| `yolox-nim.image.repository`     | The repository to override the location of the YOLOX            |       |
| `yolox-nim.image.tag`            | The tag override for YOLOX                                      |       |
| `cached-nim.image.repository`    | The repository to override the location of the Cached Model NIM |       |
| `cached-nim.image.tag`           | The tag override for Cached Model NIM                           |       |
| `paddleocr-nim.image.repository` | The repository to override the location of the Paddle OCR NIM   |       |
| `paddleocr-nim.image.tag`        | The tag override for Paddle OCR NIM                             |       |
| `deplot-nim.image.repository`    | The repository to override the location of the Deplot NIM       |       |
| `deplot-nim.image.tag`           | The tag override for Deplot NIM                                 |       |

### Milvus Deployment parameters

NVIngest uses Milvus and Minio to store extracted images from a document
This chart by default sets up a Milvus standalone instance in the namespace using the
Helm chart at found https://artifacthub.io/packages/helm/milvus-helm/milvus

| Name             | Description                                                            | Value     |
| ---------------- | ---------------------------------------------------------------------- | --------- |
| `milvusDeployed` | Whether to deploy Milvus and Minio from this helm chart                | `true`    |
| `milvus`         | Find values at https://artifacthub.io/packages/helm/milvus-helm/milvus | `sane {}` |

### Autoscaling parameters

Values used for creating a `Horizontal Pod Autoscaler`. If autoscaling is not enabled, the rest are ignored.
NVIDIA recommends usage of the custom metrics API, commonly implemented with the prometheus-adapter.
Standard metrics of CPU and memory are of limited use in scaling NIM.

| Name                      | Description                               | Value   |
| ------------------------- | ----------------------------------------- | ------- |
| `autoscaling.enabled`     | Enables horizontal pod autoscaler.        | `false` |
| `autoscaling.minReplicas` | Specify minimum replicas for autoscaling. | `1`     |
| `autoscaling.maxReplicas` | Specify maximum replicas for autoscaling. | `100`   |
| `autoscaling.metrics`     | Array of metrics for autoscaling.         | `[]`    |

### Redis configurations

Include any redis configuration that you'd like with the deployed Redis
Find values at https://github.com/bitnami/charts/tree/main/bitnami/redis

| Name            | Description                                                              | Value     |
| --------------- | ------------------------------------------------------------------------ | --------- |
| `redisDeployed` | Whether to deploy Redis from this helm chart                             | `true`    |
| `redis`         | Find values at https://github.com/bitnami/charts/tree/main/bitnami/redis | `sane {}` |

### Environment Variables

Define environment variables as key/value dictionary pairs

| Name                                   | Description                                                                                               | Value                      |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------- |
| `envVars`                              | Adds arbitrary environment variables to the main container using key-value pairs, for example NAME: value | `sane {}`                  |
| `envVars.MESSAGE_CLIENT_HOST`          | Override this value if disabling Redis deployment in this chart.                                          | `"nv-ingest-redis-master"` |
| `envVars.MESSAGE_CLIENT_PORT`          | Override this value if disabling Redis deployment in this chart.                                          | `"7670"`                   |
| `envVars.NV_INGEST_DEFAULT_TIMEOUT_MS` | Override the Timeout of the NVIngest requests.                                                            | `"1234"`                   |
| `envVars.MINIO_INTERNAL_ADDRESS`       | Override this to the cluster local DNS name of minio                                                      | `"nv-ingest-minio:9000"`   |
| `envVars.MINIO_PUBLIC_ADDRESS`         | Override this to publicly routable minio address, default assumes port-forwarding                         | `"http://localhost:9000"`  |
| `envVars.MINIO_BUCKET`                 | Override this for specific minio bucket to upload extracted images to                                     | `"nv-ingest"`              |

### Open Telemetry

Define environment variables as key/value dictionary pairs for configuring OTEL Deployments
A sane set of parameters is set for the deployed version of OpenTelemetry with this Helm Chart.
Override any values to the Open Telemetry helm chart by overriding the `open-telemetry` value.

| Name                                      | Description                                                                                                                                                    | Value                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `otelEnabled`                             | Whether to enable OTEL collection                                                                                                                              | `true`                                  |
| `otelDeployed`                            | Whether to deploy OTEL from this helm chart                                                                                                                    | `true`                                  |
| `opentelemetry-collector`                 | Configures the opentelemetry helm chart - see https://github.com/open-telemetry/opentelemetry-helm-charts/blob/main/charts/opentelemetry-collector/values.yaml |                                         |
| `otelEnvVars`                             | Adds arbitrary environment variables for configuring OTEL using key-value pairs, for example NAME: value                                                       | `sane {}`                               |
| `otelEnvVars.OTEL_EXPORTER_OTLP_ENDPOINT` | Default deployment target for GRPC otel - Default "http://{{ .Release.Name }}-opentelemetry-collector:4317"                                                    |                                         |
| `otelEnvVars.OTEL_SERVICE_NAME`           |                                                                                                                                                                | `"nemo-retrieval-service"`              |
| `otelEnvVars.OTEL_TRACES_EXPORTER`        |                                                                                                                                                                | `"otlp"`                                |
| `otelEnvVars.OTEL_METRICS_EXPORTER`       |                                                                                                                                                                | `"otlp"`                                |
| `otelEnvVars.OTEL_LOGS_EXPORTER`          |                                                                                                                                                                | `"none"`                                |
| `otelEnvVars.OTEL_PROPAGATORS`            |                                                                                                                                                                | `"tracecontext baggage"`                |
| `otelEnvVars.OTEL_RESOURCE_ATTRIBUTES`    |                                                                                                                                                                | `"deployment.environment=$(NAMESPACE)"` |
| `otelEnvVars.OTEL_PYTHON_EXCLUDED_URLS`   |                                                                                                                                                                | `"health"`                              |
| `zipkinDeployed`                          | Whether to deploy Zipkin with OpenTelemetry from this helm chart                                                                                               | `true`                                  |

### Ingress parameters

| Name                                 | Description                                           | Value                    |
| ------------------------------------ | ----------------------------------------------------- | ------------------------ |
| `ingress.enabled`                    | Enables ingress.                                      | `false`                  |
| `ingress.className`                  | Specify class name for Ingress.                       | `""`                     |
| `ingress.annotations`                | Specify additional annotations for ingress.           | `{}`                     |
| `ingress.hosts`                      | Specify list of hosts each containing lists of paths. |                          |
| `ingress.hosts[0].host`              | Specify name of host.                                 | `chart-example.local`    |
| `ingress.hosts[0].paths[0].path`     | Specify ingress path.                                 | `/`                      |
| `ingress.hosts[0].paths[0].pathType` | Specify path type.                                    | `ImplementationSpecific` |
| `ingress.tls`                        | Specify list of pairs of TLS `secretName` and hosts.  | `[]`                     |

### Probe parameters

| Name                                | Description                               | Value     |
| ----------------------------------- | ----------------------------------------- | --------- |
| `livenessProbe.enabled`             | Enables `livenessProbe``                  | `false`   |
| `livenessProbe.httpGet.path`        | `LivenessProbe`` endpoint path            | `/health` |
| `livenessProbe.httpGet.port`        | `LivenessProbe`` endpoint port            | `http`    |
| `livenessProbe.initialDelaySeconds` | Initial delay seconds for `livenessProbe` | `120`     |
| `livenessProbe.timeoutSeconds`      | Timeout seconds for `livenessProbe`       | `20`      |
| `livenessProbe.periodSeconds`       | Period seconds for `livenessProbe`        | `10`      |
| `livenessProbe.successThreshold`    | Success threshold for `livenessProbe`     | `1`       |
| `livenessProbe.failureThreshold`    | Failure threshold for `livenessProbe`     | `20`      |

### Probe parameters

| Name                               | Description                              | Value     |
| ---------------------------------- | ---------------------------------------- | --------- |
| `startupProbe.enabled`             | Enables `startupProbe``                  | `false`   |
| `startupProbe.httpGet.path`        | `StartupProbe`` endpoint path            | `/health` |
| `startupProbe.httpGet.port`        | `StartupProbe`` endpoint port            | `http`    |
| `startupProbe.initialDelaySeconds` | Initial delay seconds for `startupProbe` | `120`     |
| `startupProbe.timeoutSeconds`      | Timeout seconds for `startupProbe`       | `10`      |
| `startupProbe.periodSeconds`       | Period seconds for `startupProbe`        | `30`      |
| `startupProbe.successThreshold`    | Success threshold for `startupProbe`     | `1`       |
| `startupProbe.failureThreshold`    | Failure threshold for `startupProbe`     | `220`     |

### Probe parameters

| Name                                 | Description                                | Value     |
| ------------------------------------ | ------------------------------------------ | --------- |
| `readinessProbe.enabled`             | Enables `readinessProbe``                  | `false`   |
| `readinessProbe.httpGet.path`        | `ReadinessProbe`` endpoint path            | `/health` |
| `readinessProbe.httpGet.port`        | `ReadinessProbe`` endpoint port            | `http`    |
| `readinessProbe.initialDelaySeconds` | Initial delay seconds for `readinessProbe` | `120`     |
| `readinessProbe.timeoutSeconds`      | Timeout seconds for `readinessProbe`       | `10`      |
| `readinessProbe.periodSeconds`       | Period seconds for `readinessProbe`        | `30`      |
| `readinessProbe.successThreshold`    | Success threshold for `readinessProbe`     | `1`       |
| `readinessProbe.failureThreshold`    | Failure threshold for `readinessProbe`     | `220`     |

### Service parameters

| Name                         | Description                                                                                                                               | Value       |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `service.type`               | Specifies the service type for the deployment.                                                                                            | `ClusterIP` |
| `service.name`               | Overrides the default service name                                                                                                        | `""`        |
| `service.port`               | Specifies the HTTP Port for the service.                                                                                                  | `8000`      |
| `service.nodePort`           | Specifies an optional HTTP Node Port for the service.                                                                                     | `nil`       |
| `service.annotations`        | Specify additional annotations to be added to service.                                                                                    | `{}`        |
| `service.labels`             | Specifies additional labels to be added to service.                                                                                       | `{}`        |
| `serviceAccount`             | Options to specify service account for the deployment.                                                                                    |             |
| `serviceAccount.create`      | Specifies whether a service account should be created.                                                                                    | `true`      |
| `serviceAccount.annotations` | Sets annotations to be added to the service account.                                                                                      | `{}`        |
| `serviceAccount.name`        | Specifies the name of the service account to use. If it is not set and create is "true", a name is generated using a "fullname" template. | `""`        |

### Secret Creation

Manage the creation of secrets used by the helm chart

| Name                       | Description                                            | Value   |
| -------------------------- | ------------------------------------------------------ | ------- |
| `ngcSecret.create`         | Specifies whether to create the ngc api secret         | `false` |
| `ngcSecret.password`       | The password to use for the NGC Secret                 | `""`    |
| `imagePullSecret.create`   | Specifies whether to create the NVCR Image Pull secret | `false` |
| `imagePullSecret.password` | The password to use for the NVCR Image Pull Secret     | `""`    |


