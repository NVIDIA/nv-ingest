# NVIDIA NVIngest

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
    --set image.repository="#placeholder" \
    --set image.tag="24.08-rc2" \
    #placeholder

```

Optionally you can create your own versions of the `Secrets` if you do not want to use the creation via the helm chart.


```bash

NAMESPACE=nvidia-nims
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

You can also use an External Secret Store like Vault, the name of the secret name expected for the NGC API is `ngc-api` and the secret name expected for NVCR is `nvcrimagepullsecret`.

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
| `envVars.MESSAGE_CLIENT_PORT`          | Override this value if disabling Redis deployment in this chart.                                          | `"6379"`                   |
| `envVars.NV_INGEST_DEFAULT_TIMEOUT_MS` | Override the Timeout of the NVIngest requests.                                                            | `"1234"`                   |

### Open Telemetry

Define environment variables as key/value dictionary pairs for configuring OTEL Deployments
A sane set of parameters is set for the deployed version of OpenTelemetry with this Helm Chart.
Override any values to the Open Telemetry helm chart by overriding the `open-telemetry` value.

| Name                                      | Description                                                                                                                                                    | Value                                                            |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `otelEnabled`                             | Whether to enable OTEL collection                                                                                                                              | `false`                                                          |
| `otelDeployed`                            | Whether to deploy OTEL from this helm chart                                                                                                                    | `false`                                                          |
| `otelEnvVars`                             | Adds arbitrary environment variables for configuring OTEL using key-value pairs, for example NAME: value                                                       | `sane {}`                                                        |
| `otelEnvVars.OTEL_EXPORTER_OTLP_ENDPOINT` |                                                                                                                                                                | `"http://$(HOST_IP):4317" # sends to gRPC receiver on port 4317` |
| `otelEnvVars.OTEL_SERVICE_NAME`           |                                                                                                                                                                | `"nemo-retrieval-service"`                                       |
| `otelEnvVars.OTEL_TRACES_EXPORTER`        |                                                                                                                                                                | `"otlp"`                                                         |
| `otelEnvVars.OTEL_METRICS_EXPORTER`       |                                                                                                                                                                | `"otlp"`                                                         |
| `otelEnvVars.OTEL_LOGS_EXPORTER`          |                                                                                                                                                                | `"none"`                                                         |
| `otelEnvVars.OTEL_PROPAGATORS`            |                                                                                                                                                                | `"tracecontext baggage"`                                         |
| `otelEnvVars.OTEL_RESOURCE_ATTRIBUTES`    |                                                                                                                                                                | `"deployment.environment=$(NAMESPACE)"`                          |
| `otelEnvVars.OTEL_PYTHON_EXCLUDED_URLS`   |                                                                                                                                                                | `"health"`                                                       |
| `opentelemetry-collector`                 | Configures the opentelemetry helm chart - see https://github.com/open-telemetry/opentelemetry-helm-charts/blob/main/charts/opentelemetry-collector/values.yaml |                                                                  |
| `zipkinDeployed`                          | Whether to deploy Zipkin with OpenTelemetry from this helm chart                                                                                               | `false`                                                          |

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


