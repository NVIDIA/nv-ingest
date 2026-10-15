# OpenShift deployment

Use this guide when you install the [NeMo Retriever Helm chart](./README.md) on **OpenShift 4.x** with the default **restricted-v2** Security Context Constraint (SCC) and **Pod Security Admission (PSA) `restricted`** profile.

For general Kubernetes and Helm deployment choices, refer to [Deployment options](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/). For chart values and NIM wiring, refer to the [Helm chart README](./README.md).

## Overview { #openshift-deployment }

The chart defaults target generic Kubernetes clusters that allow fixed numeric UIDs (`runAsUser` / `runAsGroup` / `fsGroup` **1000**). OpenShift namespaces under **restricted-v2** assign a per-namespace UID/GID range instead. A stock `helm install` without overrides therefore fails SCC validation, emits PSA warnings, or crashes on log paths the random UID cannot write.

We do **not** change chart defaults for OpenShift-only behavior (that would affect other platforms). Use the overrides below, or save the YAML block into a local values file and pass `-f <file>` on every `helm install` / `helm upgrade`.

### Cluster posture (typical hardened namespaces)

| Control | Typical default on a new OpenShift project |
| --- | --- |
| SCC | **restricted-v2** (first match in priority order) |
| PSA | `pod-security.kubernetes.io/warn=restricted` (and often `audit=restricted`; `enforce` may be unset on dev clusters) |
| UID assignment | SCC injects `runAsUser` / `fsGroup` from the namespace range (for example `1000750000–1000759999`) |

On clusters with **PSA `enforce=restricted`**, missing container `securityContext` fields become hard rejections, not warnings.

### Override reference (maps to chart limitations)

| Symptom on stock install | Cause | Helm override |
| --- | --- | --- |
| `FailedCreate`: UID/GID **1000** not in namespace range | Hardcoded `service.podSecurityContext` UID/GID/fsGroup | Omit `runAsUser`, `runAsGroup`, and `fsGroup`; keep only `runAsNonRoot: true` |
| PSA warning: `allowPrivilegeEscalation`, capabilities, `seccompProfile` | Empty `service.securityContext` | Set restricted baseline on `service.securityContext` (refer to the sample below) |
| `PermissionError` on `/var/lib/nemo-retriever/retriever-service.log` when `persistence.enabled=false` | Default log path is image-owned; random UID cannot write without a PVC | Point `serviceConfig.logging.file` at `/tmp/...` (chart mounts `emptyDir` at `/tmp`) |
| `CreateContainerConfigError`: non-numeric image `USER nemo` on **vectordb** | Vectordb container has no `securityContext` block for SCC to annotate | Set `serviceConfig.vectordb.enabled=false` for a minimal service-only install, or patch the vectordb Deployment after install (below) |
| PSA warnings on **otel-collector** | Otel Deployment has no `securityContext` in the chart | `topology.otel.enabled=false` unless you patch that Deployment |
| Audio/video fails or pod never gets `ffmpeg` | `service.installFfmpeg=true` runs sudo at startup; **restricted-v2** blocks privilege escalation (`no-new-privileges`) | Prebuild a service image with `ffmpeg`/`ffprobe` baked in (refer to [Audio and video on restricted OpenShift](#audio-and-video-ffmpeg-on-restricted-openshift)); leave `service.installFfmpeg=false` |
| `ImagePullBackOff` for a service image in the **internal OpenShift registry** | Chart-rendered `imagePullSecrets` may omit the namespace SA `dockercfg` secret required for internal-registry pulls | List every required pull secret under `imagePullSecrets` (refer to [Internal registry pull secrets](#internal-registry-pull-secrets)) |
| Optional NIM `CrashLoopBackOff` with missing `.so` in logs | GPU/CUDA libraries not on `LD_LIBRARY_PATH` for some NIM Operator stacks on OCP | Append paths through `nimOperator.<key>.env` (refer to [Optional NIM runtime environment](#optional-nim-runtime-environment)) |

### Recommended value overrides

Save the block below as a local values file (for example `openshift-restricted.yaml`) and pass `-f openshift-restricted.yaml` on every `helm install` / `helm upgrade`. Use this profile for **restricted-v2** / PSA **restricted** namespaces with the NIM Operator, a prebuilt `ffmpeg` service image when you need [audio and video](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/audio-video.md), and optional NIM env overrides from later sections.

```yaml
# OpenShift overrides for nemo-retriever Helm chart (restricted-v2 / PSA restricted).
# helm install retriever ./nemo_retriever/helm -f openshift-restricted.yaml ...

service:
  podSecurityContext:
    runAsNonRoot: true
    # Do NOT set runAsUser, runAsGroup, or fsGroup — OpenShift SCC assigns them.
  securityContext:
    runAsNonRoot: true
    allowPrivilegeEscalation: false
    capabilities:
      drop: ["ALL"]
    seccompProfile:
      type: RuntimeDefault
  # Leave installFfmpeg false on restricted-v2; use a custom image for audio/video.
  installFfmpeg: false

serviceConfig:
  logging:
    # Writable without persistence PVC (chart always mounts emptyDir at /tmp).
    file: /tmp/retriever-service.log
  vectordb:
    # Set false for a minimal service-only install; refer to the vectordb patch below if you enable it.
    enabled: false

topology:
  otel:
    enabled: false
```

In `topology.mode: split`, `service.securityContext` applies to the gateway,
realtime, and batch service containers. It also applies to the
`wait-for-gateway` init containers in the realtime and batch Deployments.

When **`persistence.enabled=true`**, you can keep the default log path under `persistence.mountPath` (`/var/lib/nemo-retriever`) because the PVC is mounted and SCC-assigned `fsGroup` applies. When persistence is off, always relocate logs to `/tmp` (or another path backed by `service.extraVolumes`).

### Audio and video (ffmpeg) on restricted OpenShift { #audio-and-video-ffmpeg-on-restricted-openshift }

The Helm chart supports `service.installFfmpeg=true`, which installs `ffmpeg`/`ffprobe` at container startup through passwordless `sudo`. On OpenShift **restricted-v2** SCC, that path is blocked: PSA restricted sets `allowPrivilegeEscalation: false` and the SCC enforces **no-new-privileges**, so the entrypoint cannot elevate to install packages.

For audio and video extraction on OpenShift, **do not** set `service.installFfmpeg=true`. Instead, extend the service image on a connected build host and point the chart at that tag (same pattern as [air-gapped custom service images](./README.md#1-service-image)):

```dockerfile
FROM nvcr.io/nvidia/nemo-microservices/nrl-service:<BASE_TAG>
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER nemo
```

Push the result to NGC, your private registry, or the [OpenShift internal registry](#internal-registry-pull-secrets), then set `service.image.repository` / `service.image.tag` accordingly.

### Internal registry pull secrets { #internal-registry-pull-secrets }

When you rebuild the service image into the OpenShift internal registry (`image-registry.openshift-image-registry.svc:5000/...`), pods normally pull through the namespace ServiceAccount's automatic `kubernetes.io/dockercfg` secret. The chart renders an explicit `imagePullSecrets` list on every Pod, so you must list **every** secret each Pod needs—including the SA `dockercfg` secret when `service.image` points at the internal registry.

If the rendered list contains only `ngc-secret` (for NGC NIM images) while `service.image` uses the internal registry, the service Pod can fail with `ImagePullBackOff`.

**Internal-registry service image only** — clear the chart-managed NGC pull secret name so the helper does not inject `ngc-secret` alone:

```yaml
ngcImagePullSecret:
  create: false
  name: ""   # Explicitly empty — clears the default "ngc-secret"

imagePullSecrets:
  - name: default-dockercfg-xxxxx   # replace with your SA secret (section below)
```

When `ngcImagePullSecret.name` is empty and NIM Operator CRs are enabled, set a non-empty `nimOperator.<key>.image.pullSecrets` for each enabled NIM (or keep the chart-wide name and use the "Both NGC NIMs..." pattern below). An empty global name with empty per-NIM `pullSecrets` fails chart render.

**Both NGC NIMs and an internal-registry service image** — list every secret the pods need:

```yaml
ngcImagePullSecret:
  create: false
  name: ngc-secret

imagePullSecrets:
  - name: ngc-secret
  - name: default-dockercfg-xxxxx
```

Look up the ServiceAccount pull secret name:

```bash
oc get sa default -n nemo-retriever -o jsonpath='{.imagePullSecrets[*].name}{"\n"}'
```

Pre-create `ngc-secret` / `ngc-api` with `oc create secret` when `ngcImagePullSecret.create=false` and `ngcApiSecret.create=false`, as in the install examples below.

### Optional NIM runtime environment { #optional-nim-runtime-environment }

On some OpenShift clusters with the NIM Operator, **optional** NIM pods (for example Parakeet ASR or Omni caption) need an explicit `LD_LIBRARY_PATH` so CUDA/driver libraries from the GPU Operator stack are visible at runtime. Symptoms are `CrashLoopBackOff` and log lines referencing a missing `.so` (for example `libcudart` or `libcudnn`).

Helm **replaces** the whole `env` list when you override `nimOperator.<key>.env` in a values file. Copy the chart defaults from `values.yaml` for that NIM, then append `LD_LIBRARY_PATH`:

```yaml
nimOperator:
  audio:
    env:
      # Retain chart defaults, then append LD_LIBRARY_PATH.
      - name: NIM_TAGS_SELECTOR
        value: "name=parakeet-1-1b-ctc-en-us,mode=ofl,vad=default,diarizer=disabled"
      - name: NIM_TRITON_LOG_VERBOSE
        value: "1"
      - name: LD_LIBRARY_PATH
        value: "/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
  nemotron_3_nano_omni_30b_a3b_reasoning:
    env:
      # Retain chart defaults, then append LD_LIBRARY_PATH.
      - name: NIM_HTTP_API_PORT
        value: "8000"
      - name: NIM_TRITON_LOG_VERBOSE
        value: "1"
      - name: LD_LIBRARY_PATH
        value: "/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
```

Paths vary with GPU Operator version and node image. Inspect a healthy GPU workload on the same cluster (`oc exec` into a CUDA sample pod) or the failing NIM pod's filesystem before pinning production values.

### Call the Omni NIM directly { #omni-caption-manual-smoke-testing }

The retriever service caption profile already sends `chat_template_kwargs.enable_thinking=false` to the Omni NIM during ingest. To confirm the in-cluster Omni NIM returns caption text in `message.content`, call `/v1/chat/completions` directly (for example with `curl` against the `NIMService`) and include the same flag:

```json
{
  "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

For pipeline scope (PDF chart regions are not captioned), refer to [Image captioning (support matrix)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#image-captioning) and [Image captioning (pipeline scope)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/multimodal-extraction.md#image-captioning) in the extraction docs.

### Example install (service only, no in-cluster NIMs)

Use this flow when you want only the retriever service pod: disable the in-cluster NIM Operator stack (`nims.enabled=false`), skip the persistence PVC, and skip the results PVC. Pre-create NGC pull/API secrets, then install with the restricted OpenShift values file:

```bash
oc new-project nemo-retriever

oc create secret docker-registry ngc-secret -n nemo-retriever \
  --docker-server=nvcr.io --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"

oc create secret generic ngc-api -n nemo-retriever \
  --from-literal=NGC_API_KEY="$NGC_API_KEY" \
  --from-literal=NGC_CLI_API_KEY="$NGC_API_KEY"

helm install retriever ./nemo_retriever/helm -n nemo-retriever \
  -f openshift-restricted.yaml \
  --set ngcImagePullSecret.create=false \
  --set ngcApiSecret.create=false \
  --set nims.enabled=false \
  --set persistence.enabled=false \
  --set retrieverResults.enabled=false \
  --set service.image.repository=nvcr.io/nvidia/nemo-microservices/nrl-service \
  --set service.image.tag=26.8.2
```

Verify pods:

```bash
oc get pods -n nemo-retriever
oc describe pod -l app.kubernetes.io/name=nemo-retriever -n nemo-retriever
```

You should see SCC-assigned numeric `runAsUser` on the service and
`wait-for-gateway` init containers that declare a `securityContext` block, and
no PSA warnings after overrides are applied.

### Example install with NIM Operator (in-cluster NIMs)

After you install the NIM Operator and GPU Operator, reuse `openshift-restricted.yaml` and the NGC secrets from the service-only example. Point `service.image` at a **ffmpeg-enabled** build when you use [audio and video](#audio-and-video-ffmpeg-on-restricted-openshift). Add [optional NIM `LD_LIBRARY_PATH`](#optional-nim-runtime-environment) overrides if ASR or Omni pods crash on missing libraries.

```bash
helm install retriever ./nemo_retriever/helm -n nemo-retriever \
  -f openshift-restricted.yaml \
  --set ngcImagePullSecret.create=false \
  --set ngcApiSecret.create=false \
  --set service.image.repository=nvcr.io/nvidia/nemo-microservices/nrl-service \
  --set service.image.tag=26.8.2
```

After install, confirm workloads reach Ready before you run ingest:

```bash
oc get pods -n nemo-retriever
oc get nimservice -n nemo-retriever
```

When you enable optional NIMs, wait for Parakeet ASR and Omni caption pods to become Ready before you run [audio and video](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/audio-video.md) or caption extraction. If ASR or Omni pods crash with missing `.so` errors, apply the [LD_LIBRARY_PATH overrides](#optional-nim-runtime-environment) above and upgrade the release.

### Enabling the vectordb Deployment on OpenShift

`serviceConfig.vectordb.enabled=true` renders a **vectordb** container from the same image (`USER nemo`, non-numeric). The chart does not yet expose a `securityContext` value for that container. After `helm install`, patch the Deployment so OpenShift can inject a numeric UID into the container spec:

```bash
RELEASE=retriever
NS=nemo-retriever
VDB_DEPLOY="${RELEASE}-nemo-retriever-vectordb"

oc patch deployment "$VDB_DEPLOY" -n "$NS" --type=json -p='[
  {"op": "add", "path": "/spec/template/spec/containers/0/securityContext", "value": {
    "allowPrivilegeEscalation": false,
    "capabilities": {"drop": ["ALL"]},
    "runAsNonRoot": true,
    "seccompProfile": {"type": "RuntimeDefault"}
  }}
]'
```

Re-apply the patch after `helm upgrade` if the Deployment is recreated. A future chart release may add first-class `topology.vectordb.securityContext` values.

### Enabling the OpenTelemetry collector on OpenShift

The chart's otel-collector Deployment likewise lacks `securityContext` fields. Prefer `topology.otel.enabled=false` (as in the sample values) unless you operate your own collector or patch `*-otel` the same way as vectordb.

### What we intentionally do not require on OpenShift

Do **not** bind the namespace to **anyuid** SCC or set PSA `enforce=privileged` unless your security team explicitly approves it. The overrides above are intended to keep **restricted-v2** / PSA **restricted** posture.

## Related topics

- [Pre-Requisites & Support Matrix](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md)
- [Deployment options](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/)
- [Helm chart README](./README.md)
- [Audio and video](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/audio-video.md)
