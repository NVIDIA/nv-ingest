# nemo-retriever Helm chart

A Kubernetes Helm chart for running the **service** mode of
[`nemo-retriever`](../README.md): a FastAPI document ingestion server that
streams uploads through a set of NVIDIA NIM microservices
(object detection, OCR, and text embedding by default) and exposes
result + status APIs over HTTP / SSE.

Use **Helm** (this chart and/or the **additional Library charts** documented in the
[NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/))
for supported NIM and service deployment.

The chart ships two deployable layers behind feature flags:

- **the service** — always on; one Deployment (standalone) or three
  Deployments (split topology: gateway / realtime / batch), built from
  `Dockerfile --target service`.
- **the NIMs** — optional, GPU-backed `NIMCache` + `NIMService` custom
  resources (`apiVersion: apps.nvidia.com/v1alpha1`) reconciled by the
  **NVIDIA NIM Operator**. The chart auto-wires the operator-managed
  Service URLs into the retriever-service config when the operator CRDs
  are present in the cluster.

> **NIM Operator prerequisite.** The NIM templates are gated on the
> `apps.nvidia.com/v1alpha1` API group. Install the NIM Operator before
> running `helm install`:
> https://docs.nvidia.com/nim-operator/
>
> Without the operator the chart still installs cleanly — every NIMCache /
> NIMService template short-circuits and the service falls back to
> external NIM URLs supplied via `serviceConfig.nimEndpoints.*`.

> **Gateway scheduler state is explicitly ephemeral.** Split mode enforces one
> gateway replica and uses a `Recreate` rollout so independent in-memory
> schedulers never overlap. A gateway restart, rollout, eviction, or node failure
> loses every accepted job, queued item, active lease, status record, and SSE
> catch-up event owned by that process. Drain accepted work before any rollout.

> For behavioral consistency between local HuggingFace deployments and Helm service deployments: 
> `results = ingestor.ingest(...return_results=True)
> return_results defaults to True. This incurs a significant performance and system memory usage cost. 
> Unless you know explicitly you need to fetch extraction results to the client, you should use:
> return_results=False
> If you must return results, you may need to increase pod memory specs to support the increased pod memory usage.

---

## Layout

```
nemo_retriever/helm/
├── Chart.yaml
├── values.yaml
├── README.md            <-- this file
├── openshift.md         <-- OpenShift restricted-v2 install guide
├── .helmignore
├── examples/
│   └── values-b200-4gpu-bo767.yaml       # measured four-B200 BO767 profile
└── templates/
    ├── _helpers.tpl
    ├── NOTES.txt
    ├── configmap.yaml                         # renders retriever-service.yaml
    ├── deployment.yaml                        # the service Deployment(s)
    ├── service.yaml                           # ClusterIP/NodePort for the service
    ├── ingress.yaml                           # optional Ingress
    ├── hpa.yaml                               # optional HorizontalPodAutoscaler
    ├── servicemonitor.yaml                    # optional Prometheus ServiceMonitor
    ├── serviceaccount.yaml
    ├── pvc.yaml                               # general persistence PVC
    ├── secrets.yaml                           # ngc-secret + ngc-api
    └── nims/
        ├── nemotron-page-elements-v3.yaml    # NIMCache + NIMService
        ├── nemotron-table-structure-v1.yaml   # NIMCache + NIMService
        ├── nemotron-ocr-v2.yaml               # NIMCache + NIMService
        ├── nemotron-3-embed-1b.yaml             # NIMCache + NIMService (text embed)
        ├── llama-nemotron-rerank-vl-1b-v2.yaml  # NIMCache + NIMService (optional; auto-wired when enabled)
        ├── nemotron-parse.yaml                # NIMCache + NIMService (optional; auto-wired when enabled)
        ├── nemotron-3-nano-omni-30b-a3b-reasoning.yaml  # NIMCache + NIMService (optional; auto-wired when enabled)
        └── audio.yaml                         # NIMCache + NIMService (optional; not auto-wired)
```

---

## Quick start

The examples in this README use the Helm release name `retriever`.
`nemo-retriever` is the chart name. It is not the release name. When a
command omits `--namespace`, Helm installs into the current kubectl
namespace. The [Recommended minimal install](#recommended-minimal-install-2682)
and [Full teardown](#full-teardown) set `REL` and `NS` explicitly so
cleanup targets the same release.

### Persistent storage prerequisite { #persistent-storage-prerequisite }

The default chart creates **seven** PersistentVolumeClaims: three
chart-managed service claims and four NIM Operator NIMCache claims for
the core NIMs. `helm install` reports `STATUS: deployed` when the
release is rendered, even if every claim stays `Pending`. Confirm a
working persistent-volume binding strategy before you install.

Run the following preflight commands:

```bash
kubectl get storageclass
kubectl get pv
```

Use one of the following strategies:

- A default StorageClass (`(default)` in `kubectl get storageclass`)
  backed by a working provisioner.
- Explicit `storageClass` values for every default claim, each backed
  by a working provisioner or matching persistent volumes.
- Compatible static persistent volumes, or pre-created claims where
  the chart supports `existingClaim`.

When a `storageClass` value is empty, the chart omits
`storageClassName`. Kubernetes then assigns the default StorageClass
if one exists. If none exists, the claim binds only to a compatible
classless persistent volume that matches the requested size and
`ReadWriteOnce` access mode.

The following table lists the default claims for a release named
`retriever`:

| Example claim name | Default size | Helm value path |
| --- | --- | --- |
| `retriever-nemo-retriever-data` | `50Gi` | `persistence.storageClass` |
| `retriever-nemo-retriever-retriever-results` | `50Gi` | `retrieverResults.storageClass` |
| `retriever-nemo-retriever-vectordb-data` | `50Gi` | `topology.vectordb.persistence.storageClass` |
| `nemotron-page-elements-v3-pvc` | `25Gi` | `nimOperator.page_elements.storage.pvc.storageClass` |
| `nemotron-table-structure-v1-pvc` | `25Gi` | `nimOperator.table_structure.storage.pvc.storageClass` |
| `nemotron-ocr-v2-pvc` | `25Gi` | `nimOperator.ocr.storage.pvc.storageClass` |
| `nemotron-3-embed-1b-pvc` | `50Gi` | `nimOperator.vlm_embed.storage.pvc.storageClass` |

When `nims.enabled=false`, the four NIMCache claims are not created.
The three chart-managed claims still are, unless you disable those
persistence blocks. Enabling an optional NIM adds another NIMCache
claim for that key.

To pin a named StorageClass on every default claim, pass the following
`--set` flags with your cluster class name:

```bash
helm install retriever ./nemo_retriever/helm \
  --set persistence.storageClass=<STORAGE_CLASS> \
  --set retrieverResults.storageClass=<STORAGE_CLASS> \
  --set topology.vectordb.persistence.storageClass=<STORAGE_CLASS> \
  --set nimOperator.page_elements.storage.pvc.storageClass=<STORAGE_CLASS> \
  --set nimOperator.table_structure.storage.pvc.storageClass=<STORAGE_CLASS> \
  --set nimOperator.ocr.storage.pvc.storageClass=<STORAGE_CLASS> \
  --set nimOperator.vlm_embed.storage.pvc.storageClass=<STORAGE_CLASS> \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password=$NGC_API_KEY
```

Set each per-NIM `nimOperator.<key>.storage.pvc.storageClass` path.
The chart-level `nimOperator.nimCache.pvc.storageClass` value is not
applied to the four core NIMCache resources.

`persistence.existingClaim` and `retrieverResults.existingClaim` skip
chart PVC creation and mount the named claim. The VectorDB claim does
not have an `existingClaim` path.

If `helm install` already succeeded and claims stay `Pending`, refer
to [Helm install succeeds but PersistentVolumeClaims stay Pending](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/troubleshoot.md#helm-pending-pvcs).

### GPU scheduling prerequisite { #gpu-scheduling-prerequisite }

The [model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements)
table lists **Total GPUs: 1** for Core Features because the four
default NIMs can co-reside on one A10G or better GPU with supported
profiles. That figure is VRAM capacity.
It is not the number of exclusive Kubernetes GPU requests this chart
makes.

A default install creates four `NIMService` resources
(`page_elements`, `table_structure`, `ocr`, `vlm_embed`). Each
renders `spec.resources.limits.nvidia.com/gpu: 1` from
`nimOperator.nimServiceGpuLimit` (default `1`). On a conventional
cluster without MIG, time-slicing, or another sharing mechanism, the
scheduler consumes **four allocatable GPU slots across eligible
nodes**. Four one-GPU nodes satisfy that topology. A cluster that
has only one allocatable GPU schedules only one NIM pod. The other
three stay `Pending`.

Choose one of the following before you install:

- Provide four allocatable `nvidia.com/gpu` slots across eligible
  nodes for the default core topology. Each optional NIM adds its
  per-NIM GPU request. Optional keys, including the default
  `answer_llm` Nemotron 3.5 Lightning NIM, use `nimServiceGpuLimit` (one GPU).
- Configure GPU sharing so the cluster advertises at least four
  `nvidia.com/gpu` slots. Time-slicing is the documented sharing
  path. It works on GPUs that do not support Multi-Instance GPU
  (MIG), including A10G.

This chart does not pack the four NIMs onto a single `nvidia.com/gpu`
request. Sharing is cluster configuration through the
[NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html).
Time-slicing does not isolate GPU memory. Combined VRAM of the
scheduled NIMs must still fit on the physical GPU.

Confirm allocatable GPU slots across the cluster:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu
```

Sum `GPU` across eligible nodes. A default core install needs four
slots across the cluster. Four nodes that each report `1` are
enough. A single node needs a value of `4` or greater only when you
pack all four core NIMs onto one physical GPU with sharing and
placement constraints.

The following GPU Operator time-slicing ConfigMap advertises four
replicas per physical GPU. On a node with one physical GPU, that
creates four logical slots, which is enough for the four default
NIMServices. A node with two physical GPUs advertises eight slots.
The NVIDIA GPU Operator must already be installed. Apply the
ConfigMap in the GPU Operator namespace (commonly `gpu-operator`)
before `helm install`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config-all
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4
```

```bash
kubectl create -n gpu-operator -f time-slicing-config-all.yaml
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    -n gpu-operator --type merge \
    -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config-all", "default": "any"}}}}'
```

That ClusterPolicy patch is a cluster-administrator change.
`default: "any"` applies the four-replica configuration to all
eligible GPU Operator nodes. It can oversubscribe unrelated GPU
workloads that share those nodes.

Time-slicing oversubscribes allocatable slots. It does not place
the four NIM pods on one physical GPU. The scheduler can spread
them across GPUs or nodes. The one-physical-GPU recipe requires
both of the following: the target node has one physical GPU and
advertises at least four replicas, and all four NIMServices are
pinned to that node. Confirm that node's `Allocatable`
`nvidia.com/gpu` is `4` or greater, and pin the four core
NIMServices as shown below.

The default `answer_llm` Nemotron 3.5 Lightning NIMService is outside the
four-core-NIM sharing recipe and adds one GPU request. If you override
the slot with a multi-GPU profile, provide the required physical GPUs;
time-slice replicas do not satisfy tensor-parallel requirements. Refer
to [Answer generation](#answer-generation-llm).

On a multi-GPU or multi-node cluster, pin the four core
NIMServices to a single-GPU node. Set
`nimOperator.<key>.nodeSelector` on `page_elements`,
`table_structure`, `ocr`, and `vlm_embed`. `nodeSelector`
constrains node placement. On a multi-GPU node, it does not
ensure all four pods receive logical replicas from the same
physical GPU. This chart does not render pod affinity or
topology spread constraints:

```yaml
nimOperator:
  page_elements:
    nodeSelector:
      kubernetes.io/hostname: <gpu-node>
  table_structure:
    nodeSelector:
      kubernetes.io/hostname: <gpu-node>
  ocr:
    nodeSelector:
      kubernetes.io/hostname: <gpu-node>
  vlm_embed:
    nodeSelector:
      kubernetes.io/hostname: <gpu-node>
```

On a mixed cluster, omit `devicePlugin.config.default` and label
only the target node so other GPU nodes keep exclusive access:

```bash
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    -n gpu-operator --type merge \
    -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config-all"}}}}'
kubectl label node <gpu-node> nvidia.com/device-plugin.config=any
```

Increase `replicas` if you also enable optional NIMs that request
one GPU each on the same shared GPU and combined VRAM still fits.
Count each of those Helm GPU requests. Do not increase time-slice
replicas to satisfy tensor-parallel GPU counts. For the full
procedure, node labels, and limitations, refer to
[Time-Slicing GPUs in Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html).
MIG is an advanced GPU Operator configuration outside this chart.
The chart does not set a MIG strategy, MIG profile, or MIG
resource requests. Default NIMServices request `nvidia.com/gpu`.
A working MIG deployment is GPU-specific and profile-specific and
can require per-NIM `resources` overrides. For GPU Operator MIG,
refer to
[GPU Operator with MIG](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html).
The NIM Operator also documents DRA-based sharing. This chart does
not render `draResources`; refer to
[NIM Operator DRA](https://docs.nvidia.com/nim-operator/latest/dra.html)
if you manage ResourceClaims outside the chart.

If `helm install` already succeeded and NIM pods stay `Pending` on
`nvidia.com/gpu`, refer to
[Core NIM pods stay Pending for GPU](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/troubleshoot.md#helm-pending-gpus).

### Measured four-B200 BO767 profile { #measured-four-b200-bo767-profile }

Use [`examples/values-b200-4gpu-bo767.yaml`](./examples/values-b200-4gpu-bo767.yaml)
to reproduce the best four-physical-GPU BO767 configuration measured with the
26.08.1 chart. The profile pins the tested service and NIM images. It deploys
three embed replicas, three OCR replicas, one page-elements replica, and one
table-structure replica. OCR and object-detection NIMs use a maximum pipeline
batch size of `8`. The embed NIM uses FP16 precision.

On four NVIDIA B200 GPUs, this configuration processed 54,730 BO767 pages at
92.790 pages per second. Recall@5 was 0.860747, Recall@10 was 0.907164, and
nDCG@10 was 0.755657. These measurements describe the tested workload and
hardware. They are not general performance guarantees.

The target node must advertise two time-sliced `nvidia.com/gpu` resources per
physical GPU and use the NVIDIA device plugin's `packed` allocation policy.
Set `TARGET_NODE` to that node's `kubernetes.io/hostname` label. The commands
below pass the target as a node selector for all four NIMs, which prevents
Kubernetes from scheduling any replica on another eligible GPU node.
The final placement must contain the following physical GPU pairs:

| Physical GPU | NIM workloads |
| --- | --- |
| 1 | Embed and OCR |
| 2 | Embed and OCR |
| 3 | Embed and OCR |
| 4 | Page elements and table structure |

The final values file cannot select a physical GPU UUID. A single Helm install
can therefore produce a slower `embed+embed`, `embed+OCR`, and `OCR+OCR`
placement. Stage the replicas to fill one packed GPU at a time.

Start with one embed and one OCR replica while page elements and table
structure are disabled:

```bash
REL=retriever
NS=nemo-retriever
PROFILE=./nemo_retriever/helm/examples/values-b200-4gpu-bo767.yaml
TARGET_NODE=b200-node-name
NODE_SELECTOR_ARGS=(
  "--set-string=nimOperator.page_elements.nodeSelector.kubernetes\\.io/hostname=${TARGET_NODE}"
  "--set-string=nimOperator.table_structure.nodeSelector.kubernetes\\.io/hostname=${TARGET_NODE}"
  "--set-string=nimOperator.ocr.nodeSelector.kubernetes\\.io/hostname=${TARGET_NODE}"
  "--set-string=nimOperator.vlm_embed.nodeSelector.kubernetes\\.io/hostname=${TARGET_NODE}"
)

kubectl get node "${TARGET_NODE}" \
  -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu

helm upgrade --install "${REL}" ./nemo_retriever/helm \
  -n "${NS}" --create-namespace \
  -f "${PROFILE}" \
  "${NODE_SELECTOR_ARGS[@]}" \
  --set nimOperator.page_elements.enabled=false \
  --set nimOperator.table_structure.enabled=false \
  --set nimOperator.ocr.replicas=1 \
  --set nimOperator.vlm_embed.replicas=1

kubectl wait -n "${NS}" --for=jsonpath='{.status.state}'=Ready \
  nimservice/llama-nemotron-embed-vl-1b-v2 nimservice/nemotron-ocr-v2 \
  --timeout=30m
```

Add each subsequent embed replica before its matching OCR replica. Wait for
each Deployment so the packed allocator fills the next physical GPU pair:

```bash
set -euo pipefail

wait_for_deployment_replicas() {
  local deployment="$1"
  local replicas="$2"
  local generation

  kubectl wait deployment/"${deployment}" -n "${NS}" \
    --for=jsonpath='{.spec.replicas}'="${replicas}" --timeout=10m || return 1
  generation="$(kubectl get deployment/"${deployment}" -n "${NS}" \
    -o jsonpath='{.metadata.generation}')" || return 1
  kubectl wait deployment/"${deployment}" -n "${NS}" \
    --for=jsonpath='{.status.observedGeneration}'="${generation}" --timeout=10m || return 1
  kubectl wait deployment/"${deployment}" -n "${NS}" \
    --for=jsonpath='{.status.updatedReplicas}'="${replicas}" --timeout=10m || return 1
  kubectl wait deployment/"${deployment}" -n "${NS}" \
    --for=jsonpath='{.status.readyReplicas}'="${replicas}" --timeout=10m
}

for REPLICAS in 2 3; do
  kubectl patch nimservice llama-nemotron-embed-vl-1b-v2 -n "${NS}" \
    --type merge -p "{\"spec\":{\"replicas\":${REPLICAS}}}"
  wait_for_deployment_replicas llama-nemotron-embed-vl-1b-v2 "${REPLICAS}" \
    || exit 1

  kubectl patch nimservice nemotron-ocr-v2 -n "${NS}" \
    --type merge -p "{\"spec\":{\"replicas\":${REPLICAS}}}"
  wait_for_deployment_replicas nemotron-ocr-v2 "${REPLICAS}" \
    || exit 1
done
```

Apply the final profile to enable page elements and table structure. Wait for
the four NIMServices and the retriever service before sending traffic:

```bash
helm upgrade "${REL}" ./nemo_retriever/helm -n "${NS}" -f "${PROFILE}" \
  "${NODE_SELECTOR_ARGS[@]}"

kubectl wait -n "${NS}" --for=jsonpath='{.status.state}'=Ready \
  nimservice/llama-nemotron-embed-vl-1b-v2 \
  nimservice/nemotron-ocr-v2 \
  nimservice/nemotron-page-elements-v3 \
  nimservice/nemotron-table-structure-v1 \
  --timeout=30m
kubectl rollout status deployment/"${REL}"-nemo-retriever \
  -n "${NS}" --timeout=10m
```

Verify placement by grouping the visible UUID reported by each runtime NIM
pod. Continue only when each physical UUID has the expected pair:

```bash
for POD in $(kubectl get pods -n "${NS}" -o name | grep -E \
  'llama-nemotron-embed-vl|nemotron-ocr-v2-|nemotron-page-elements-v3-|nemotron-table-structure-v1-' \
  | grep -v -- '-job-'); do
  printf '%s ' "${POD}"
  kubectl exec -n "${NS}" "${POD}" -- \
    nvidia-smi --query-gpu=uuid --format=csv,noheader
done | sort -k2,2 -k1,1
```

#### Reproduce the measured BO767 run

The measured run used
[`harness/runfiles/bo767_vl_text_hybrid_beir_service.json`](../harness/runfiles/bo767_vl_text_hybrid_beir_service.json).
The runfile selects text embedding at element granularity and automatic
retrieval. The resolved benchmark disables captioning, deduplication,
chunking, page deduplication, and reranking. It queries the top 10 results and
sets `overwrite=true`, which replaces the target table before ingestion.

Create a dataset-paths file that points to the SSD copy of the corpus. The
measured machine used the following file:

```yaml
schema_version: 1
datasets:
  bo767:
    path: /raid/jperez/data/bo767
    query_file: /raid/jperez/nemo_retriever/data/bo767_query_gt.csv
```

In one terminal, forward the deployed service port:

```bash
kubectl port-forward -n "${NS}" \
  service/"${REL}"-nemo-retriever 17671:7670
```

From the repository root in another terminal, run the same harness command
used for the measurement:

```bash
uv run --project nemo_retriever retriever-harness run-files \
  --output-dir /raid/jperez/nemo_retriever_runs/bo767_scale_260801_embed3_ocr3_4gpu_balanced_batch8/results \
  --session-name bo767_scale_260801_embed3_ocr3_4gpu_balanced_batch8 \
  --mode service \
  --service-endpoint http://localhost:17671 \
  --dataset-paths /raid/jperez/nemo_retriever_runs/bo767_scale_260801_embed3_ocr3_4gpu_balanced_batch8/dataset_paths.yaml \
  nemo_retriever/harness/runfiles/bo767_vl_text_hybrid_beir_service.json
```

The timed run started after the service and all four NIMServices were ready.
The NIM containers and model caches were warm, but the corpus did not receive
an untimed ingestion warmup pass. Use a new output directory and update the
machine-local paths when you reproduce the run elsewhere.

This profile targets BO767 PDF ingestion. `service.installFfmpeg` remains
`false`. Enable FFmpeg separately for audio or video workflows.

### 1. Service image { #1-service-image }

The chart defaults to the image published to NGC:

```
nvcr.io/nvidia/nemo-microservices/nrl-service:26.8.2
```

Release-published tags of that image are multi-architecture (`linux/amd64` and `linux/arm64`). Kubernetes pulls the variant that matches the node.

Pulling from `nvcr.io` requires an NGC pull secret — either set
`ngcImagePullSecret.create=true` (see below) or pre-create one in the
namespace named `ngc-secret`.

To run a locally built image instead, build and push it from the repo root,
then override `service.image.repository` / `service.image.tag`:

```bash
# from the repo root:
docker build \
    --target service \
    --build-arg DOWNLOAD_DEFAULT_TOKENIZER=True \
    --build-arg RETRIEVER_VERSION=<TAG> \
    --build-arg RETRIEVER_RELEASE_TYPE=release \
    -t <YOUR_REGISTRY>/nemo-retriever-service:<TAG> .
docker push <YOUR_REGISTRY>/nemo-retriever-service:<TAG>
```

Audio and video extraction require the `ffmpeg` and `ffprobe` system
binaries inside the service container. The bundled service image can install
them at container startup when you set `service.installFfmpeg=true`, which
sets `INSTALL_FFMPEG=true` for the image entrypoint:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set service.image.repository=<YOUR_REGISTRY>/nemo-retriever-service \
  --set service.image.tag=<TAG> \
  --set service.installFfmpeg=true
```

Do not also set `INSTALL_FFMPEG` in `service.env`; the chart fails rendering
when both are configured so the rendered Pod does not contain duplicate
environment variables.

When `service.installFfmpeg=false` (the default), the service still starts
normally and processes PDF, image, text and HTML uploads. Audio / video
uploads are rejected up-front with **HTTP 501**:

```text
Audio and video ingestion require FFmpeg in the retriever service
container, but the following dependencies are missing: ffmpeg, ffprobe.
Re-deploy the Helm chart with `--set service.installFfmpeg=true` …
```

The retriever-service container also logs a `WARNING` at startup when
FFmpeg is missing so cluster operators can fix the deployment before
the first media upload arrives, instead of debugging a Ray worker
traceback (`RuntimeError: MediaChunkActor requires media dependencies;
missing: ffmpeg, ffprobe`) after the fact. The same WARNING is emitted
on every pod (gateway, realtime, batch) because all roles classify
uploads — flipping `service.installFfmpeg=true` updates them all.

Runtime installation uses passwordless `sudo` scoped to installing the
`ffmpeg` package in the service image. The pod must have network egress to the
Ubuntu package repositories, a writable root filesystem, and a security policy
that allows sudo/setuid behavior. Do not set
`service.securityContext.allowPrivilegeEscalation: false` or
`service.securityContext.readOnlyRootFilesystem: true` for this path.

For air-gapped or locked-down clusters, see
[Deployment options — Air-gapped and disconnected deployment](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/#air-gapped-deployment).
On a connected staging host you can extend the service image, for example:

```dockerfile
FROM <YOUR_REGISTRY>/nemo-retriever-service:<BASE_TAG>
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER nemo
```

### 2. Install with external NIM endpoints (operator not required)

Complete the [persistent storage prerequisite](#persistent-storage-prerequisite)
before you install. With `nims.enabled=false`, the chart still creates
the three service claims unless you disable those persistence blocks.

If you already have NIM endpoints reachable from the cluster (e.g. another
namespace, or NVIDIA Build), turn the master switch off and supply the
URLs directly:

```bash
helm install retriever ./nemo_retriever/helm \
  --set nims.enabled=false \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set serviceConfig.nimEndpoints.pageElementsInvokeUrl=http://page-elements.svc:8000/v1/page-elements \
  --set serviceConfig.nimEndpoints.tableStructureInvokeUrl=http://table-structure.svc:8000/v1/table-structure \
  --set serviceConfig.nimEndpoints.ocrInvokeUrl=http://ocr.svc:8000/v1/ocr \
  --set serviceConfig.nimEndpoints.embedInvokeUrl=http://embed.svc:8000/v1/embeddings

```
`ngcApiSecret` materialises an `ngc-api` Secret containing both
`NGC_API_KEY` and `NGC_CLI_API_KEY` keys; the service container reads it
via `optional: true` `secretKeyRef`, so the install still succeeds when
the secret is absent (useful for fully local NIM endpoints).

### 3. Install with the NIM Operator (in-cluster NIMs)

Complete the [persistent storage prerequisite](#persistent-storage-prerequisite)
and the [GPU scheduling prerequisite](#gpu-scheduling-prerequisite)
before you install. The default path creates all seven claims and four
exclusive `nvidia.com/gpu: 1` NIMService requests.

Install the [NIM Operator](https://docs.nvidia.com/nim-operator/) first so
the `NIMCache` / `NIMService` CRDs (`apps.nvidia.com/v1alpha1`) are
registered. A plain `helm install` reconciles the four core NIMs
(`page_elements`, `table_structure`, `ocr`, `vlm_embed`) — every other
NIM (the VL reranker `rerankqa`, Nemotron Parse, Omni 30B, and the
Parakeet `audio` ASR NIM) is **disabled by default** to honor the
"optional and disabled by default" contract in
[deployment-options.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/deployment-options.md);
refer to [Recommended minimal install](#recommended-minimal-install-2682)
for the opt-in `--set` flags that turn any of them on.

```bash
helm install retriever ./nemo_retriever/helm \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password=$NGC_API_KEY
```

### Recommended minimal install (26.8.2) { #recommended-minimal-install-2682 }

Complete the [persistent storage prerequisite](#persistent-storage-prerequisite)
and the [GPU scheduling prerequisite](#gpu-scheduling-prerequisite)
before you install.

Deploy only the four core NIMs that the retriever service auto-wires
(`page_elements`, `table_structure`, `ocr`, `vlm_embed`). Set `REL` and
`NS` to the values you will reuse for upgrade and teardown. Replace
`NS` if you install into a namespace other than `default`.

```bash
REL=retriever
NS=default

helm install "${REL}" ./nemo_retriever/helm -n "${NS}" --create-namespace \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set service.image.tag=26.8.2
```

> The VL reranker (`rerankqa`), Nemotron Parse, the Nemotron 3 Nano Omni 30B caption NIM, the generic answer-generation LLM (`answer_llm`, Nemotron 3.5 Lightning defaults), and the Parakeet `audio` ASR NIM are **all off by default** — they only reconcile when you explicitly opt in. Opt-in flags:
>
> * VL reranker — `--set nimOperator.rerankqa.enabled=true` (auto-wires `nim_endpoints.rerank_invoke_url` / `rerank_model_name` — refer to [Query-time reranking](#query-time-reranking))
> * Nemotron Parse — `--set nimOperator.nemotron_parse.enabled=true` (enabling this NIM also sets the default PDF extract method to `nemotron_parse`)
> * Omni 30B captioner — `--set nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true`
> * Answer generation LLM — `--set nimOperator.answer_llm.enabled=true`
> * Parakeet ASR — `--set nimOperator.audio.enabled=true` (also set `serviceConfig.nimEndpoints.audioGrpcEndpoint=audio:50051` to wire ASR into the service, plus `service.installFfmpeg=true` if your image does not bundle ffmpeg)
>
> This matches the "optional and disabled by default" contract in [deployment-options.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/deployment-options.md) and avoids silently pulling ≈ 62 GiB of Omni weights, loading a large two-GPU LLM, or claiming extra dedicated GPUs on a "default" install. Refer to the [model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements) table for per-NIM GPU and disk costs.

The chart auto-wires the operator-managed in-cluster URLs of the three
"core" NIMs into the service's `nim_endpoints` block:

| key | operator-managed Service | invoke path |
| --- | ------------------------ | ----------- |
| `nimOperator.page_elements` | `nemotron-page-elements-v3` | `/v1/page-elements` |
| `nimOperator.table_structure` | `nemotron-table-structure-v1` | `/v1/table-structure` |
| `nimOperator.ocr` | `nemotron-ocr-v2` | `/v1/ocr` |
| `nimOperator.vlm_embed`       | `nemotron-3-embed-1b` | `/v1/embeddings` |

### Query reranking (optional)

The VL reranker NIM is optional and disabled by default. Set `nimOperator.rerankqa.enabled=true` to opt in, and keep `nims.enabled=true` (the default). When those flags are true and the NIM Operator CRDs are present, the chart auto-wires `nim_endpoints.rerank_invoke_url` and `rerank_model_name`. A `POST /v1/query` request with `rerank=true` then uses the in-cluster ranking Service. If `nims.enabled=false`, the chart does not auto-wire those fields. Set `serviceConfig.nimEndpoints.rerankInvokeUrl` to override, or to point at a hosted ranking endpoint. Refer to [Query-time reranking](#query-time-reranking) for the resolution order.

Track operator reconciliation with:

```bash
kubectl get nimcache,nimservice -n <namespace>
kubectl describe nimservice nemotron-object-detection -n <namespace>
```

First-time NIMCache reconciliation downloads model weights to a PVC. By
default (`nimOperator.nimCache.keepOnUninstall: true`) every **NIMCache**
carries `helm.sh/resource-policy: keep` so those downloads survive
`helm uninstall`. **NIMService** CRs do not use `keep` and are removed by
Helm on uninstall.

### Why NIM resources still exist after `helm uninstall`

| What you see | Typical cause |
|--------------|----------------|
| `NIMCache` + PVC remain | **Expected** when `keepOnUninstall` is true (default). Helm intentionally skips deleting caches so you do not re-pull multi‑GiB weights. |
| `NIMService` CR remains | **Not expected** on a normal uninstall. Usually an **orphan** from a failed install/upgrade (release never recorded the resource, or the chart renamed a NIM). |
| Deployments / GPU pods still running | Often the operator workload for a **kept** `NIMCache`, or a stale `NIMService` that Helm did not own. Check `kubectl get nimservice,nimcache -n <ns>`. |
| `nemotron-*-job-*` pods in `Error` | The NIM Operator's **model-download Job** for a `NIMCache` (not the retriever service). Failed cache pulls retry and leave Error pods until the Job or `NIMCache` is deleted. Common after a failed `helm install` when the release is rolled back but `keep` retains the cache CR. |
| `helm uninstall` appears to do nothing | Wrong release name or namespace. The chart name is `nemo-retriever`. The example release name is `retriever`. Confirm with `helm list --all-namespaces`. A missing or failed release (`helm list -n <ns> --all`) can also leave CRs without a release to clean them up. |

To change a NIM image on a later install or upgrade, delete the kept
`NIMCache` first. Refer to
[Changing a NIM image repository or tag](#changing-nim-image-repository-or-tag).

### Full teardown { #full-teardown }

On a development cluster, remove the Helm release, kept `NIMCache`
objects, and model PVCs. Set `REL` and `NS` to the same values you used
at install time. The
[Recommended minimal install](#recommended-minimal-install-2682) uses
`REL=retriever` and `NS=default`. If you omitted `--namespace` at
install time, Helm used the current kubectl namespace. Replace `NS` if
that namespace is not `default`.

1. Set the release identity and confirm the target:

   ```bash
   REL=retriever
   NS=default

   helm list -n "${NS}"
   kubectl get deployment,nimservice,nimcache -n "${NS}"
   ```

   If `helm list` shows no matching release, stop. Run
   `helm list --all-namespaces` and set `REL` and `NS` to the release
   you intend to remove.

2. Uninstall the release. Do not suppress Helm errors. If Helm reports
   that the release was not found, correct `REL` and `NS` before you
   delete NIM resources:

   ```bash
   helm uninstall "${REL}" -n "${NS}"
   helm list -n "${NS}" --all
   ```

3. Delete kept `NIMCache` objects and leftover `NIMService` CRs. Helm
   leaves **NIMCache** objects when
   `nimOperator.nimCache.keepOnUninstall` is `true` (the default). Each
   NIM uses a resource name from the chart, not the Helm release name.
   The following command deletes every default name this chart can
   create, including optional NIMs. `--ignore-not-found` skips names
   you did not install.

   If you overrode `nimOperator.ocr.nimServiceName`,
   `nimOperator.vlm_embed.nimServiceName`, or
   `nimOperator.answer_llm.nimServiceName`, replace those three default
   names with the values you set.

   ```bash
   kubectl delete nimservice,nimcache -n "${NS}" \
     nemotron-page-elements-v3 \
     nemotron-table-structure-v1 \
     nemotron-ocr-v2 \
     nemotron-3-embed-1b \
     llama-nemotron-rerank-vl-1b-v2 \
     nemotron-parse \
     nemotron-3-nano-omni-30b-a3b-reasoning \
     audio \
     answer-llm \
     --ignore-not-found
   ```

   List what remains. Delete leftover CRs by the names in that list
   before you continue. Remaining objects can be renamed caches or
   resources that another product owns.

   ```bash
   kubectl get nimservice,nimcache -n "${NS}"
   ```

   Do not run `kubectl delete nimservice,nimcache -n "${NS}" --all`
   unless this namespace contains only this install. That command
   deletes every `NIMService` and `NIMCache` in the namespace, including
   resources that another release or product owns.

4. Optional: drop model PVCs when you re-pull weights from NGC. Confirm
   claim names first. Default claim names use a `-pvc` suffix. Include
   optional NIM claims. If you overrode a `nimServiceName` value, use
   `<that-name>-pvc` instead of the default claim.

   ```bash
   kubectl get pvc -n "${NS}" -l 'app.kubernetes.io/managed-by=nvidia-nim-operator'
   kubectl delete pvc -n "${NS}" \
     nemotron-page-elements-v3-pvc \
     nemotron-table-structure-v1-pvc \
     nemotron-ocr-v2-pvc \
     nemotron-3-embed-1b-pvc \
     llama-nemotron-rerank-vl-1b-v2-pvc \
     nemotron-parse-pvc \
     nemotron-3-nano-omni-30b-a3b-reasoning-pvc \
     audio-pvc \
     answer-llm-pvc \
     --ignore-not-found
   ```

   List operator PVCs again and delete any leftover claim that belongs
   to this install:

   ```bash
   kubectl get pvc -n "${NS}" -l 'app.kubernetes.io/managed-by=nvidia-nim-operator'
   ```

**Dev installs** that should not retain caches on uninstall:

```bash
REL=retriever
NS=default

helm upgrade --install "${REL}" ./nemo_retriever/helm -n "${NS}" \
  --set nimOperator.nimCache.keepOnUninstall=false
```

---

## Values reference (highlights)

The full schema lives in [`values.yaml`](./values.yaml). Below is the
short list of knobs you'll touch first.

### Service

| Path                          | Default                            | Notes |
|-------------------------------|------------------------------------|-------|
| `service.image.repository`    | `nvcr.io/nvidia/nemo-microservices/nrl-service` | NGC image; override to pin a different build or use a local registry. |
| `service.image.tag`           | `26.8.2`                           | Also injected as `RETRIEVER_SERVICE_VERSION` so `/openapi.json` `info.version` matches the running image tag. |

| `service.replicas`            | `1`                                | Keep at 1 because standalone job and scheduler state are process-local. |
| `service.installFfmpeg`       | `false`                            | Install `ffmpeg`/`ffprobe` at container startup by setting `INSTALL_FFMPEG=true`. Requires network egress, writable root filesystem, and sudo/setuid allowed. Not for air-gapped clusters — use a custom image instead. |
| `service.resources.requests`  | `16 / 16Gi`                        | Tune in tandem with `serviceConfig.pipeline.*Workers`. |
| `service.resources.limits`    | `96 / 96Gi`                        |       |
| `service.gpu.enabled`         | `false`                            | The service does **not** need a GPU. |

For audio and video extraction, set `service.installFfmpeg=true` when your
cluster allows runtime package installation. **OpenShift restricted-v2** blocks
that path — use a prebuilt service image instead; refer to [Audio and video on restricted OpenShift](./openshift.md#audio-and-video-ffmpeg-on-restricted-openshift).
For air-gapped clusters, refer to [Deployment options — Air-gapped and disconnected deployment](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/#air-gapped-deployment).

### Audio and video (Parakeet ASR) { #audio-video-parakeet }

Parakeet ASR is disabled by default. The chart default is `nimOperator.audio.enabled=false`. The chart does not auto-wire the in-cluster ASR endpoint when you enable the audio NIM.

To run self-hosted Parakeet for [audio and video extraction](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/audio-video.md), set both of the following values:

```yaml
nimOperator:
  audio:
    enabled: true

serviceConfig:
  nimEndpoints:
    audioGrpcEndpoint: audio:50051
```

Equivalent Helm flags are `--set nimOperator.audio.enabled=true` and `--set serviceConfig.nimEndpoints.audioGrpcEndpoint=audio:50051`.

Enabling only `nimOperator.audio.enabled=true` renders the Parakeet `NIMCache` and `NIMService`. The ConfigMap still sets `audio_grpc_endpoint` to `null`. The retriever service cannot send ASR traffic until you also set `serviceConfig.nimEndpoints.audioGrpcEndpoint`. Disable other optional NIMs you do not need. Refer to [Recommended minimal install](#recommended-minimal-install-2682).

After you set those values, complete the following steps:

1. Pin the ASR `NIMService` to a **dedicated GPU** with `nimOperator.audio.resources`, `nodeSelector`, or `tolerations` (refer to [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html)).
2. Confirm the GPU SKU in [Model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements) (footnote ⁴ lists Blackwell limitations).
3. Set `service.installFfmpeg=true` when the retriever service will process audio or video on clusters that allow runtime package install (refer to `service.installFfmpeg` above). On **OpenShift restricted-v2**, use a [prebuilt service image](./openshift.md#audio-and-video-ffmpeg-on-restricted-openshift) instead.

The in-cluster gRPC Service name is `audio` on port `50051`. Graph ingest does not read this Helm value. Pass the same endpoint through `ASRParams.audio_endpoints` in Python. Refer to [NIM Operator sub-stack](#nim-operator-sub-stack).

### Health probes

The service exposes unauthenticated health endpoints for Kubernetes probes:

| Endpoint | Purpose | Default Helm use |
| --- | --- | --- |
| `GET /v1/live` | Shallow process liveness. The endpoint does not check worker backends or wait for service readiness. | Startup and liveness probes. In split mode, the realtime and batch `wait-for-gateway` init containers poll this endpoint on the internal gateway startup Service while waiting for the gateway process to start. |
| `GET /v1/health` | Deep readiness. In split gateway mode, the endpoint checks the realtime and batch workers. It returns HTTP `503` when either required worker is unreachable or returns a non-2xx health response. | Readiness probe on the externally exposed gateway Service. |

In split topology, the gateway readiness probe uses `/v1/health`, which returns
HTTP `503` until realtime and batch workers are healthy. Worker Pods cannot start
until their `wait-for-gateway` init container reaches the gateway's shallow
`/v1/live` endpoint. The externally exposed gateway Service does not publish
endpoints for an unready Pod, which would deadlock a clean install. The chart
therefore renders an additional cluster-internal Service named
`<release>-nemo-retriever-gateway-startup` with
`publishNotReadyAddresses: true`. Init containers reach
`http://<release>-nemo-retriever-gateway-startup:<networkService.port>/v1/live`
through that Service so workers can start while the gateway Pod is still
unready. Client traffic continues to use the readiness-gated gateway Service, so
`/v1/health` still removes an unhealthy gateway from Service endpoints after
startup.

The split-mode `wait-for-gateway` init container uses `busybox:1.37`.
Refer to [Air-gapped deployment](#air-gapped-deployment) for the mirror inventory
and the values override limitation.

When a gateway returns HTTP `503` from `/v1/health`, Kubernetes removes it from
the readiness-gated gateway Service endpoints until its required workers are
ready. The response includes backend health details to help diagnose the
unavailable dependency.

### Service networking

| Path | Default | Notes |
|------|---------|-------|
| `networkService.port` | `7670` | Kubernetes Service port. The chart uses this port for generated Service DNS URLs. |
| `serviceConfig.server.port` | `7670` | Retriever service container listener port. |

You can set these values independently. When they differ, Kubernetes Services
listen on `networkService.port` and route to the container listener on
`serviceConfig.server.port`.

In standalone mode, the chart renders one Service named
`<release>-nemo-retriever`.

In split topology (`topology.mode: split`), the chart renders four Services:

| Service | Example name (release `retriever`) | Type | Client entrypoint |
| --- | --- | --- | --- |
| Gateway | `retriever-nemo-retriever-gateway` | `networkService.type` (default `NodePort`) | Yes |
| Gateway startup | `retriever-nemo-retriever-gateway-startup` | ClusterIP (cluster-internal only) | No |
| Realtime | `retriever-nemo-retriever-realtime` | ClusterIP | No |
| Batch | `retriever-nemo-retriever-batch` | ClusterIP | No |

The gateway startup Service publishes the gateway Pod address while it is still
unready so realtime and batch init containers can reach `/v1/live`. It is not a
client entrypoint. Refer to [Health probes](#health-probes).

### Service configuration (rendered into `retriever-service.yaml`)

| Path                                              | Default | Notes |
|---------------------------------------------------|---------|-------|
| `serviceConfig.server.port`                       | `7670`  | Retriever service container listener port. Refer to [Service networking](#service-networking). |
| `serviceConfig.logging.level`                     | `INFO`  | Retriever service log verbosity, rendered as `logging.level` in the ConfigMap. Typical values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. Setting `INGEST_LOG_LEVEL` in `service.env` does not change this field. |
| `serviceConfig.pipeline.realtimeWorkers`          | `24`    | Per-pod realtime worker count. |
| `serviceConfig.pipeline.batchWorkers`             | `48`    | Per-pod batch worker count. Refer to [Timeouts and alleviating ingest failures](#timeouts-and-alleviating-ingest-failures) if embed or pool errors appear under load. |
| `serviceConfig.resources.maxUploadBytes`          | `500000000` | Maximum upload file size in bytes; requests exceeding the limit are rejected before buffering. |
| `serviceConfig.sidecarStore.maxPayloadBytes`      | `33554432` | Maximum sidecar metadata upload size in bytes. The service rejects a larger upload with HTTP `413` before buffering the complete payload. This value cannot exceed `serviceConfig.resources.maxUploadBytes`. |
| `serviceConfig.nimEndpoints.*InvokeUrl`           | `""`    | Override the auto-resolved NIM Operator URL. Available knobs: `pageElementsInvokeUrl`, `tableStructureInvokeUrl`, `ocrInvokeUrl`, `embedInvokeUrl`, `rerankInvokeUrl` (refer to [Query-time reranking](#query-time-reranking)), and `captionInvokeUrl` (refer to [Image captioning (Omni 30B)](#image-captioning-omni-30b)). |
| `serviceConfig.nimEndpoints.rerankModelName`      | `""`    | Model id sent to the remote reranker. Auto-set to `nvidia/llama-nemotron-rerank-vl-1b-v2` whenever a rerank URL is resolved. |
| `serviceConfig.nimEndpoints.captionModelName`     | `""`    | Model id sent to the remote VLM. Auto-set to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` whenever a caption URL is resolved. |
| `serviceConfig.nimEndpoints.rerankInvokeUrl`      | `""`    | Ranking API URL used by `POST /v1/query` when `rerank=true`. Auto-wired from the optional `rerankqa` NIM when enabled; override to point at a hosted or external ranking endpoint. |
| `serviceConfig.nimEndpoints.rerankModelName`      | `""`    | Model ID sent to the ranking API. Auto-set to `nvidia/llama-nemotron-rerank-vl-1b-v2` whenever a rerank URL is resolved; override for a different compatible reranker. |
| `serviceConfig.nimEndpoints.audioGrpcEndpoint`    | `""`    | gRPC endpoint for Parakeet ASR. Not auto-wired from `nimOperator.audio`. Set `audio:50051` when you enable the audio NIM. |
| `serviceConfig.llm.enabled`                         | `false` | Enables `POST /v1/answer`. Auto-flips to true when `nimOperator.answer_llm` is enabled and the operator URL resolves. |
| `serviceConfig.llm.apiBase`                         | `""`    | OpenAI-compatible LLM base URL. Explicit value wins; otherwise `answer_llm` opt-in resolves to `http://answer-llm:8000/v1` by default. |
| `serviceConfig.llm.apiKeySecret.name`                | `""`    | Optional Secret name for external LLM credentials. Explicit values win; otherwise operator-managed `answer_llm` mounts its `authSecret` as `NEMO_RETRIEVER_LLM_API_KEY` so LiteLLM/OpenAI has a credential value without writing it to the ConfigMap. |
| `serviceConfig.llm.apiKeySecret.key`                 | `api_key` | Secret key for external LLM credentials. Operator-managed `answer_llm` uses `NGC_API_KEY` from `nimOperator.answer_llm.authSecret` when no explicit LLM Secret is set. |
| `serviceConfig.llm.model`                           | `""` | Optional explicit LiteLLM model id. Leave empty to inherit `nimOperator.answer_llm.model` when using the operator-managed answer LLM; set it for external endpoints. |
| `serviceConfig.llm.ragSystemPromptPrefix`           | `""` | Optional explicit RAG prompt prefix. Leave empty unless an endpoint needs model-specific prompt directives. |
| `serviceConfig.llm.reasoningEnabled`               | `true` | Request-level reasoning toggle for `/v1/answer`. Defaults to true for external OpenAI-compatible providers; set false for Nemotron endpoints that should receive portable no-reasoning controls. |
| `serviceConfig.agentic.enabled`                    | `false` | Enables `POST /v1/query` with `agentic=true`. The `agentic_query` MCP tool also requires this flag, plus `serviceConfig.mcp.enabled=true`. Not auto-enabled by `nimOperator.answer_llm`. Refer to [Agentic retrieval (self-hosted Nemotron 3.5 Lightning)](#agentic-retrieval-llm). |
| `serviceConfig.agentic.llmModel`                   | `""` | Chat model used by the inner agentic retrieval loop. Required when `invokeUrl` is set. Use the NIM-advertised ID (for Nemotron 3.5 Lightning, `nvidia/nemotron-3.5-lightning-30b-a3b`), not the LiteLLM `openai/` prefix. |
| `serviceConfig.agentic.invokeUrl`                  | `""` | OpenAI-compatible chat completions endpoint used by agentic retrieval. Not auto-populated from `answer_llm`. For the in-cluster Nemotron 3.5 Lightning NIM, set `http://answer-llm:8000/v1/chat/completions`. |
| `serviceConfig.agentic.requestTimeoutS`            | `1800` | Gateway and MCP timeout for the multi-step agentic retrieval call. |
| `serviceConfig.mcp.enabled`                       | `false` | Mounts FastMCP at `serviceConfig.mcp.path`. The bundled non-Helm `retriever-service.yaml` defaults to `true`; the chart defaults to `false` and returns HTTP `404` at that path until you opt in. Refer to [MCP HTTP endpoint](#mcp-http-endpoint). |
| `serviceConfig.mcp.path`                          | `/mcp` | HTTP mount path for the FastMCP app. Remote agents must connect to this path. |
| `serviceConfig.mcp.queryMethods`                  | `classic` | Retrieval tools to register: `classic` (`query` only), `agentic` (`agentic_query` only), or `all` (both). Agentic tools also require `serviceConfig.agentic.enabled=true`. |
| `serviceConfig.mcp.enableWriteTools`              | `true` | When `true`, registers the `ingest_documents` MCP tool. |
| `serviceConfig.vectordb.enabled`                  | `true`  | Deploy the LanceDB vectordb Pod. When `true` the chart **requires** a resolvable embed endpoint (refer to [VectorDB and the embed endpoint](#vectordb-and-the-embed-endpoint)); `helm install` / `helm upgrade` fails fast otherwise. |
| `serviceConfig.vectordb.lancedbUri`               | `/data/vectordb` | LanceDB on the vectordb Pod's PVC. |
| `serviceConfig.vectordb.indexMode`                | `auto` | `auto`, `dense`, or `hybrid`. Fresh `auto` storage creates FTS and uses hybrid retrieval; persistent dense storage remains dense until `hybrid` is requested explicitly. |
| `serviceConfig.vectordb.embedModel`               | `nvidia/nemotron-3-embed-1b` | Passed to vectordb + worker `embed_model_name`. |
| `serviceConfig.vectordb.embedModelProviderPrefix` | `""` | Optional LiteLLM provider prefix prepended to the remote embed model name. |
| `serviceConfig.vectordb.writeTimeoutSeconds`      | `300` | Rendered as `vectordb.write_timeout_s`. How long a worker waits for the vectordb Pod to acknowledge a record write. A write that is not acknowledged fails the document, so raise this value on slow storage. Refer to [Timeouts and alleviating ingest failures](#timeouts-and-alleviating-ingest-failures). |

### Sidecar metadata in split topology

`ServiceIngestor.vdb_upload()` uploads sidecar metadata to
`POST /v1/ingest/sidecar` and uses the returned opaque ID in the subsequent
ingest request. The public API retains its time-to-live and
`consume_on_read` reuse behavior.

Sidecar metadata uses the gateway's in-memory store in standalone and split
topologies. Configure the maximum allowed payload size as needed:

```yaml
serviceConfig:
  sidecarStore:
    maxPayloadBytes: 33554432
```

In `topology.mode: split`, the chart requires one gateway replica. Upload and
reference each sidecar while that gateway remains running. An uploaded sidecar
that has not yet been admitted is lost if the gateway restarts or is replaced.

At ingest admission, the gateway authorizes and binds the sidecar to the
work attachment before it leases work to a worker. Workers receive the bound
attachment with their work and do not resolve sidecar IDs. This supports
independent realtime and batch pools, worker replicas, and routing decisions
without requiring sidecar replication between workers.

#### VectorDB and the embed endpoint { #vectordb-and-the-embed-endpoint }

The VectorDB storage default is `indexMode: auto`; most users should leave it
unchanged. A fresh table creates and waits for its FTS index after the first
write, while an existing dense table is left dense. Set
`serviceConfig.vectordb.indexMode=hybrid` only when you explicitly want to
upgrade an existing dense table. Incremental rows remain searchable through
LanceDB's unindexed-tail scan; the service performs incremental FTS maintenance
automatically and reports FTS and maintenance state from `/v1/health`.

The vectordb Pod's `/v1/query` handler embeds the incoming query text
before searching LanceDB.  It needs a NIM embedding endpoint to do that,
and rendering the Deployment with an empty `--embed-endpoint` produces a
Pod that passes its `/v1/health` probe but answers every `/v1/query`
request with `HTTP 501 No embedding endpoint configured.` — a healthy
deployment that silently breaks retrieval.

To prevent this, the chart now refuses to render
`deployment-vectordb.yaml` when no embed endpoint can be resolved.
`helm install` / `helm upgrade --install` fails with a message listing
the three supported escape valves:

```
serviceConfig.vectordb.enabled=true but the embed endpoint could not be
resolved.  Pick one of:

  1. --set serviceConfig.nimEndpoints.embedInvokeUrl=http://<host>:<port>/v1/embeddings
  2. --set nimOperator.vlm_embed.enabled=true   # requires apps.nvidia.com/v1alpha1 CRDs
  3. --set serviceConfig.vectordb.enabled=false
```

Resolution order matches the rest of the chart (refer to [Mix and match NIM
sources](#3-install-with-the-nim-operator-in-cluster-nims)):

1. Explicit `serviceConfig.nimEndpoints.embedInvokeUrl` always wins.
2. Otherwise the operator-managed URL of
   `nimOperator.vlm_embed.nimServiceName` is used, provided
   `nimOperator.vlm_embed.enabled=true` **and** the
   `apps.nvidia.com/v1alpha1` CRDs are installed in the cluster.
3. Otherwise the chart fails the install.

#### Answer generation (operator-managed LLM) { #answer-generation-llm }

Enable the generic `answer_llm` NIM slot to add service-mode answer
generation on top of the VectorDB query path. The slot defaults to the
Nemotron 3.5 Lightning NIM, but the image, model id, service name, resources,
profile filter, and environment can be overridden for another
OpenAI-compatible LLM NIM.

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.answer_llm.enabled=true
```

When the NIM Operator CRDs are present, the chart renders an `answer-llm`
NIMCache/NIMService by default and writes this block into
`retriever-service.yaml`:

```yaml
llm:
  enabled: true
  model: "openai/nvidia/nemotron-3.5-lightning-30b-a3b"
  api_base: "http://answer-llm:8000/v1"
  rag_system_prompt_prefix: null
  reasoning_enabled: true
```

The retriever service then exposes `POST /v1/answer`, which calls the
VectorDB pod's `/v1/query` endpoint for context and sends those chunks to
the configured LLM endpoint. This path does not require tool calling.
The `answer_llm` NIM is not wired into `serviceConfig.agentic` and is
not tool-call ready by default. For agentic retrieval against that NIM,
refer to
[Agentic retrieval (self-hosted Nemotron 3.5 Lightning)](#agentic-retrieval-llm).
The `answer_llm` NIM starts with
`NIM_PASSTHROUGH_ARGS=--reasoning-parser nemotron_v3`, which separates
Lightning reasoning from the final answer content. `/v1/answer` controls
reasoning per request. By default, `serviceConfig.llm.reasoningEnabled=true`, so requests
leave reasoning behavior to the LLM endpoint defaults and avoid sending
provider-specific `chat_template_kwargs` to external OpenAI-compatible
endpoints. Set `serviceConfig.llm.reasoningEnabled=false` for Nemotron
endpoints that should skip reasoning; the service then adds both `/no_think`
and `chat_template_kwargs.enable_thinking=false`. The default Nemotron 3.5 Lightning
NIMService requests one GPU through `nimOperator.nimServiceGpuLimit` and lets
the NIM Operator select a compatible profile. The chart NIMCache PVC is
`250Gi`. Override `resources`, `modelProfile`, or `env` for deployments that
use a different profile or hardware topology. Refer to
[Model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements)
in the Support Matrix.

When `answer_llm` is enabled and no explicit `serviceConfig.llm.apiKeySecret`
is set, the service also mounts `nimOperator.answer_llm.authSecret` as
`NEMO_RETRIEVER_LLM_API_KEY`; OpenAI-compatible clients require a
credential value even for in-cluster NIM endpoints, and the key is never
rendered into the ConfigMap.

##### Call POST /v1/answer { #call-v1-answer }

After the answer LLM NIM is Ready, send answer requests to the retriever
gateway. The endpoint retrieves VectorDB chunks and generates a grounded
answer. Index at least one document and wait until that ingest job
succeeds before you call `/v1/answer`.

Reach the gateway on `networkService.port` (default `7670`). For a first
request against the default standalone release named `retriever`,
port-forward the Service and set `SERVICE_URL`:

```bash
kubectl port-forward -n <namespace> svc/retriever-nemo-retriever 7670:7670
export SERVICE_URL=http://127.0.0.1:7670
```

In split topology, port-forward `svc/retriever-nemo-retriever-gateway`
instead. You can also use the assigned NodePort (chart default `30670`)
or the Ingress host. Send `/v1/answer` to the gateway. Realtime and
batch worker pods return HTTP `404` for this route.

Ingest through that same gateway URL:

```bash
retriever ingest service /path/to/document.pdf \
  --service-url "${SERVICE_URL}"
```

Use `--service-api-token` or `NEMO_RETRIEVER_API_TOKEN` when public
gateway authentication is enabled. Refer to
[Ingest through a Retriever service](../docs/cli/README.md#ingest-through-a-retriever-service).

Public gateway authentication is off by default
(`serviceConfig.auth.enabled=false`). Omit the `Authorization` header
in that case. When you enable public authentication, send
`Authorization: Bearer ${NEMO_RETRIEVER_API_TOKEN}` and `X-NRL-Scope`
with a workspace scope that the token is allowed to use. Refer to
[Secrets](#secrets).

The following table lists the `POST /v1/answer` JSON fields.

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `query` | string | yes | | Question to answer from retrieved chunks. |
| `top_k` | integer | no | `5` | Number of VectorDB chunks to retrieve. Valid range is 1 through 1000. |
| `include_chunks` | boolean | no | `false` | When `true`, the response includes retrieved chunk texts. |
| `include_metadata` | boolean | no | `false` | When `true`, the response includes per-chunk metadata aligned with `chunks`. |
| `reasoning_enabled` | boolean or null | no | `null` | Per-request reasoning override. When omitted, the service uses `serviceConfig.llm.reasoningEnabled`. |
| `reference` | string or null | no | `null` | Optional ground-truth answer for scoring. |
| `judge` | boolean | no | `false` | When `true`, run LLM-as-judge scoring. `judge` requires `reference`. The gateway returns HTTP `422` when `judge` is `true` and `reference` is omitted. |

The only required field is `query`. The following example is a minimal
request:

```bash
curl -sS -X POST "${SERVICE_URL}/v1/answer" \
  -H 'Content-Type: application/json' \
  -d '{"query": "What does the indexed document say?"}'
```

The following example sets optional retrieval and reasoning fields.
Include the `Authorization` and `X-NRL-Scope` headers only when public
gateway authentication is enabled:

```bash
curl -sS -X POST "${SERVICE_URL}/v1/answer" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${NEMO_RETRIEVER_API_TOKEN}" \
  -H "X-NRL-Scope: ${NRL_SCOPE}" \
  -d '{
    "query": "What does the indexed document say?",
    "top_k": 5,
    "include_chunks": true,
    "include_metadata": true,
    "reasoning_enabled": true
  }'
```

A successful HTTP `200` response includes `query`, `answer`, `model`,
`latency_s`, and `chunk_count`. `chunks` and `metadata` are `null`
unless you set `include_chunks` or `include_metadata`. `error` is
`null` on HTTP `200`. Generation failures return HTTP `502`. The
gateway returns HTTP `404` when VectorDB or the answer LLM is not
enabled.

The following example is a representative HTTP `200` body when
`include_chunks` and `include_metadata` are `true`:

```json
{
  "query": "What does the indexed document say?",
  "answer": "The indexed document describes the retrieval workflow.",
  "model": "openai/nvidia/nemotron-3.5-lightning-30b-a3b",
  "latency_s": 1.24,
  "chunk_count": 1,
  "chunks": [
    "This document describes the retrieval workflow."
  ],
  "metadata": [
    {
      "source": "example.pdf",
      "page_number": 1
    }
  ],
  "error": null
}
```

When you send `reference`, the response also includes scoring fields
such as `token_f1`, `exact_match`, `answer_in_context`, and
`failure_mode`. When you send `judge` with `reference`, the response
also includes `judge_score`. The following request body enables judge
scoring:

```json
{
  "query": "What does the indexed document say?",
  "reference": "The indexed document describes the retrieval workflow.",
  "judge": true
}
```

The gateway publishes the live request schema at `/docs` and
`/openapi.json`.

For example, to try Nemotron 3 Nano as the answer LLM on an A100 80GB
node, override the operator-managed slot instead of adding a second
hard-coded LLM service:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.answer_llm.enabled=true \
  --set nimOperator.answer_llm.nimServiceName=nemotron-3-nano \
  --set nimOperator.answer_llm.image.repository=nvcr.io/nim/nvidia/nemotron-3-nano \
  --set nimOperator.answer_llm.image.tag=1.7.0-variant \
  --set nimOperator.answer_llm.model=openai/nvidia/nemotron-3-nano-30b-a3b \
  --set-json nimOperator.answer_llm.modelProfile='{"profiles":["5f89f01a0af587fd8bae50c611b1f358f92effdb9fb29362e1af0a986e5561c3"]}' \
  --set-json nimOperator.answer_llm.resources='{"limits":{"nvidia.com/gpu":1},"requests":{"nvidia.com/gpu":1}}' \
  --set nimOperator.answer_llm.env[0].name=NIM_HTTP_API_PORT \
  --set-string nimOperator.answer_llm.env[0].value=8000 \
  --set nimOperator.answer_llm.env[1].name=NIM_SERVED_MODEL_NAME \
  --set-string nimOperator.answer_llm.env[1].value=nvidia/nemotron-3-nano-30b-a3b \
  --set nimOperator.answer_llm.env[2].name=NIM_TENSOR_PARALLEL_SIZE \
  --set-string nimOperator.answer_llm.env[2].value=1
```

Use the repository and tag available in your NGC environment; staging
registries can use the same override shape with `nvstaging` image names
or tags. `nimOperator.answer_llm.model` is the LiteLLM model id used by
the retriever service; for an OpenAI-compatible in-cluster NIM, keep the
`openai/` prefix there and set `NIM_SERVED_MODEL_NAME` to the raw model
name advertised by the NIM. Leaving `modelProfile` empty preserves NIM
Operator auto-discovery, but it can cache every advertised profile on first
reconciliation; pin a known-compatible profile when you know the target GPU
topology.

`serviceConfig.llm.apiBase` and `serviceConfig.llm.model` can be set
explicitly to point `/v1/answer` at an external OpenAI-compatible LLM
instead of deploying an answer LLM in-cluster. For external credentials,
create a Kubernetes Secret and set `serviceConfig.llm.apiKeySecret.name`
plus `serviceConfig.llm.apiKeySecret.key`; Helm mounts the Secret as an
environment variable instead of writing the key into the ConfigMap.

`nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning` deploys Omni for
image captioning only. It does not enable `/v1/answer`. Omni
(`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`) is a supported
configurable VLM-capable answer-generation backend. Use one of the
following:

- Override the generic `answer_llm` slot with the Omni image
  `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:2.0.4-variant`,
  set `nimOperator.answer_llm.model` to
  `openai/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, set
  `NIM_SERVED_MODEL_NAME` to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`,
  and size `resources` plus `NIM_TENSOR_PARALLEL_SIZE` from the Omni rows in the
  [Support Matrix](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements)
  (one GPU on 80 GB or better, two GPUs on L40S). Leave `modelProfile`
  empty for NIM Operator auto-discovery, or pin a profile for your GPU.
- If the Omni caption NIM is already in the cluster, reuse it for
  `/v1/answer` without deploying a separate answer LLM:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true \
  --set serviceConfig.llm.enabled=true \
  --set serviceConfig.llm.apiBase=http://nemotron-3-nano-omni-30b-a3b-reasoning:8000/v1 \
  --set serviceConfig.llm.model=openai/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
```

Enabling caption Omni and the default Nemotron 3.5 Lightning `answer_llm` as separate
NIMServices adds their GPU and disk requirements. Reusing the caption
Omni endpoint for `/v1/answer` does not add a second Omni GPU or cache.

#### Agentic retrieval (self-hosted Nemotron 3.5 Lightning) { #agentic-retrieval-llm }

`nimOperator.answer_llm.enabled=true` deploys Nemotron 3.5 Lightning and auto-wires
it only to `serviceConfig.llm` for `POST /v1/answer`. That answer path
sends a plain text-generation request and does not require tool
calling. `serviceConfig.agentic` is a separate block. The chart does
not populate it from `answer_llm`.

The default Nemotron 3.5 Lightning NIM starts with
`NIM_PASSTHROUGH_ARGS=--reasoning-parser nemotron_v3`. Agentic retrieval
sends OpenAI-style tool-call messages with `tool_choice=auto`. A
self-hosted vLLM-backed Nemotron 3.5 Lightning NIM rejects those requests with
HTTP 400 unless you also pass `--enable-auto-tool-choice` and
`--tool-call-parser qwen3_coder`.

You can reuse the same Nemotron 3.5 Lightning NIM for agentic retrieval after you
add those arguments. `POST /v1/answer` continues to work. This gap
does not apply to NVIDIA-hosted Build endpoints.

If you set `nimOperator.answer_llm.env` in a values file, include
the full list. Change only the passthrough value:

```yaml
nimOperator:
  answer_llm:
    enabled: true
    env:
      - name: NIM_HTTP_API_PORT
        value: "8000"
      - name: NIM_MODEL_NAME
        value: "nvidia/nemotron-3.5-lightning-30b-a3b"
      - name: NIM_SERVED_MODEL_NAME
        value: "nvidia/nemotron-3.5-lightning-30b-a3b"
      - name: NIM_TENSOR_PARALLEL_SIZE
        value: "1"
      - name: NIM_PASSTHROUGH_ARGS
        value: "--reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"

serviceConfig:
  agentic:
    enabled: true
    llmModel: nvidia/nemotron-3.5-lightning-30b-a3b
    invokeUrl: http://answer-llm:8000/v1/chat/completions
```

Equivalent `--set` form when you do not use a values file.
Helm `--set` replaces the `env` list, so include every Nemotron 3.5 Lightning
environment entry and change only the `NIM_PASSTHROUGH_ARGS` value:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.answer_llm.enabled=true \
  --set nimOperator.answer_llm.env[0].name=NIM_HTTP_API_PORT \
  --set-string nimOperator.answer_llm.env[0].value=8000 \
  --set nimOperator.answer_llm.env[1].name=NIM_MODEL_NAME \
  --set-string nimOperator.answer_llm.env[1].value=nvidia/nemotron-3.5-lightning-30b-a3b \
  --set nimOperator.answer_llm.env[2].name=NIM_SERVED_MODEL_NAME \
  --set-string nimOperator.answer_llm.env[2].value=nvidia/nemotron-3.5-lightning-30b-a3b \
  --set nimOperator.answer_llm.env[3].name=NIM_TENSOR_PARALLEL_SIZE \
  --set-string nimOperator.answer_llm.env[3].value=1 \
  --set nimOperator.answer_llm.env[4].name=NIM_PASSTHROUGH_ARGS \
  --set-string nimOperator.answer_llm.env[4].value="--reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder" \
  --set serviceConfig.agentic.enabled=true \
  --set serviceConfig.agentic.llmModel=nvidia/nemotron-3.5-lightning-30b-a3b \
  --set serviceConfig.agentic.invokeUrl=http://answer-llm:8000/v1/chat/completions
```

`serviceConfig.agentic.llmModel` is the model ID advertised by the
NIM, not the LiteLLM `openai/` prefix used by
`serviceConfig.llm.model`. Change `invokeUrl` if you override
`nimOperator.answer_llm.nimServiceName`.

After the NIM is Ready, confirm the passthrough arguments:

```bash
kubectl exec -n <namespace> deploy/answer-llm -- printenv NIM_PASSTHROUGH_ARGS
```

The value must include `--enable-auto-tool-choice` and
`--tool-call-parser qwen3_coder`. For one-shot CLI use, port-forward
`service/answer-llm` and point `--agentic-invoke-url` at
`http://localhost:9000/v1/chat/completions`. For the CLI command,
service request, and MCP notes, refer to
[Self-hosted Helm Nemotron 3.5 Lightning](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/workflow-agentic-retrieval.md#self-hosted-helm-lightning).

For other self-hosted OpenAI-compatible NIMs, enable automatic tool
choice and the parser that model requires. Nemotron 3.5 Lightning uses `qwen3_coder` for tool calls and
`nemotron_v3` for reasoning.

The chart leaves the MCP HTTP mount disabled. Refer to
[MCP HTTP endpoint](#mcp-http-endpoint).

#### MCP HTTP endpoint { #mcp-http-endpoint }

The Helm chart sets `serviceConfig.mcp.enabled` to `false`. A
chart-rendered service does not mount the MCP endpoint, so remote
MCP agents receive HTTP `404` until you opt in. The default path is
`/mcp`. If you set `serviceConfig.mcp.path`, agents must use that
path. The bundled non-Helm `retriever-service.yaml` still sets
`mcp.enabled` to `true`.

Enable the mount:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set serviceConfig.mcp.enabled=true
```

`serviceConfig.mcp.queryMethods` is `classic` by default (`query`
only). Set `agentic` or `all` to register `agentic_query`. That tool
is omitted unless you also set
`--set serviceConfig.agentic.enabled=true` and configure
`serviceConfig.agentic.llmModel` plus
`serviceConfig.agentic.invokeUrl`.

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set serviceConfig.mcp.enabled=true \
  --set serviceConfig.mcp.queryMethods=all \
  --set serviceConfig.agentic.enabled=true \
  --set serviceConfig.agentic.llmModel=nvidia/nemotron-3.5-lightning-30b-a3b \
  --set serviceConfig.agentic.invokeUrl=http://answer-llm:8000/v1/chat/completions
```

Confirm the rendered ConfigMap before you rely on the endpoint:

```bash
helm template retriever ./nemo_retriever/helm \
  --set serviceConfig.vectordb.enabled=false \
  --set serviceConfig.mcp.enabled=true \
  | sed -n '/^    mcp:/,/^    llm:/p'
```

The block must show `enabled: true`. The stdio command
`retriever service mcp-stdio` talks to the REST API and does not
require this Helm flag. For remote-agent URL, query-method flags, and
stdio usage, refer to
[Query with MCP](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/workflow-agentic-retrieval.md#query-with-mcp).

### NIM Operator sub-stack

Each enabled NIM block under `nimOperator.<key>` renders operator resources
gated on three conditions ALL holding:

1. The `apps.nvidia.com/v1alpha1` CRDs are installed in the cluster.
2. The master switch `nims.enabled` is `true`.
3. The per-NIM `nimOperator.<key>.enabled` is `true`.

| Path                                   | Default | Notes |
|----------------------------------------|---------|-------|
| `nims.enabled`                         | `true`  | Master switch. Set false to render no NIM resources. |
| `nimOperator.page_elements.enabled` | `true` | Page Elements 2.0 service; auto-wired to `/v1/page-elements`. |
| `nimOperator.table_structure.enabled` | `true` | Table Structure 2.0 service; auto-wired to `/v1/table-structure`. |
| `nimOperator.<page_elements|table_structure>.image` | `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1` | Both services use the combined image but select distinct models. |
| `nimOperator.ocr.enabled`              | `true`  | OCR NIM. |
| `nimOperator.ocr.image`              | `nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0.1` | Default OCR NIM image. |
| `nimOperator.vlm_embed.enabled`        | `true`  | Text embedding NIM (also used by the vectordb Pod). |
| `nimOperator.vlm_embed.nimServiceName` | `nemotron-3-embed-1b` | NIMService / in-cluster DNS name. |
| `nimOperator.vlm_embed.image`          | `nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.2` | Default text embed NIM image. |
| `nimOperator.vlm_embed.env` | `NIM_HTTP_API_PORT=8000`, `NIM_TRITON_LOG_VERBOSE=1`, `OMP_NUM_THREADS=1`, `NIM_ENGINE_COUNT=1` | Environment for the default text embed NIM. Overrides replace the complete list. `NIM_PERFORMANCE_MODE=1` is optional. |
| `nimOperator.rerankqa.enabled`         | `false` | VL reranker NIM (optional). Set `true` to opt in — refer to [Query-time reranking](#query-time-reranking). Default `false` so chart installs honor the "optional and disabled by default" contract in [deployment-options.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/deployment-options.md) and do not silently provision an extra ≈ 3.1 GiB GPU NIM. The image points at the **VL** SKU (`llama-nemotron-rerank-vl-1b-v2`) per [prerequisites-support-matrix.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#default-helm-nims) — the text-only `llama-nemotron-rerank-1b-v2` silently degrades multimodal reranking and is not the documented POR. |
| `nimOperator.rerankqa.image`           | `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:2.3.0` | Default optional VL reranker NIM image. |
| `nimOperator.nemotron_parse.enabled`   | `false` | Structured-parse NIM (optional). Set `true` to deploy Parse. Enabling this NIM also sets the default PDF extract method to `nemotron_parse`. Default `false` so chart installs honor the "optional and disabled by default" contract in [deployment-options.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/deployment-options.md). Image tags follow the [image tag conventions](#image-tag-conventions). |
| `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled` | `false` | Omni 30B caption NIM (optional). Set `true` to enable image captioning — refer to [Image captioning (Omni 30B)](#image-captioning-omni-30b). This VLM is also a supported configurable `/v1/answer` backend. Enabling this key does not enable `/v1/answer`. Refer to [Answer generation (operator-managed LLM)](#answer-generation-llm). Default `false` so chart installs do not silently pull ≈ 62 GiB of BF16 weights or claim a second dedicated GPU. Image tag follows the [image tag conventions](#image-tag-conventions). |
| `nimOperator.answer_llm.enabled`       | `false` | Generic answer-generation LLM NIM (optional; Nemotron 3.5 Lightning defaults). Set `true` to enable `/v1/answer` — refer to [Answer generation (operator-managed LLM)](#answer-generation-llm). This opt-in does not enable agentic retrieval. Refer to [Agentic retrieval (self-hosted Nemotron 3.5 Lightning)](#agentic-retrieval-llm) for agentic endpoint configuration. Default `false` so installs do not silently claim an answer-generation GPU. |
| `nimOperator.answer_llm.model`         | `openai/nvidia/nemotron-3.5-lightning-30b-a3b` | LiteLLM/OpenAI model id inherited by `serviceConfig.llm.model` when the operator-managed answer LLM is enabled and no explicit service model is set. |
| `nimOperator.answer_llm.ragSystemPromptPrefix` | `""` | Optional prompt prefix inherited by `serviceConfig.llm.ragSystemPromptPrefix` only when explicitly set. Leave empty to keep the operator-managed LLM model-neutral and use `serviceConfig.llm.reasoningEnabled` for request-level reasoning control. |
| `nimOperator.audio.enabled`            | `false` | Parakeet ASR NIM (optional). Set `true` for audio/video transcription; pair with `serviceConfig.nimEndpoints.audioGrpcEndpoint=audio:50051` so the retriever-service can reach it. |
| `nimOperator.<key>.image.repository`   | `nvcr.io/nim/nvidia/...` | Per-NIM image. |
| `nimOperator.<key>.image.pullSecrets`  | `[]` | Per-NIM pull Secret name list. Empty inherits `ngcImagePullSecret.name` (default `ngc-secret`) on every NIMCache / NIMService. Non-empty replaces the chart-wide name for that NIM only. |
| `nimOperator.<key>.authSecret`         | `""` | Per-NIM auth Secret name. Empty inherits `ngcApiSecret.name` (default `ngc-api`). Non-empty replaces the chart-wide name for that NIM only. |
| `nimOperator.<key>.storage.pvc.size`   | `25Gi` (50Gi for vlm_embed/rerankqa, 100Gi parse, 300Gi VL) | NIMCache PVC size. |
| `nimOperator.<key>.storage.pvc.storageClass` | `""` | Per-NIM NIMCache StorageClass. An empty value renders an empty class on the NIMCache CR, so the operator-created claim uses the cluster default when one exists. Set this path for each enabled NIM. `nimOperator.nimCache.pvc.storageClass` is not applied to per-NIM caches. |
| `nimOperator.<key>.replicas`           | `1`     | Per-NIMService replica count. |
| `nimOperator.nimServiceGpuLimit`       | `1`     | Default `nvidia.com/gpu` limit on every NIMService when per-NIM `resources` is `{}`. Four core NIMs therefore request four GPU slots unless the cluster shares GPUs. Set to `null` for operator-only reconciliation (not reliable on all NIM Operator versions). Refer to [GPU limits and `helm upgrade`](#gpu-limits-and-helm-upgrade) and [GPU scheduling prerequisite](#gpu-scheduling-prerequisite). |
| `nimOperator.<key>.resources`          | `{}`    | Per-NIM override of the whole `resources` block. Empty uses `nimServiceGpuLimit`; non-empty replaces the chart default (may require `--force-conflicts` on later `helm upgrade`). |
| `nimOperator.modelProfile`             | `{}`    | Chart-wide NIMCache GPU/profile filter. Applied to every NIMCache that does not have its own override. Refer to [Filtering cached GPU profiles](#filtering-cached-gpu-profiles). |
| `nimOperator.<key>.modelProfile`       | `{}`    | Per-NIM NIMCache GPU/profile filter. Non-empty values REPLACE the chart-wide default (no merge). Refer to [Filtering cached GPU profiles](#filtering-cached-gpu-profiles). |
| `nimOperator.<key>.expose.service.port` | `8000` (9000 for audio) | HTTP port. |
| `nimOperator.<key>.expose.service.grpcPort` | `8001` (50051 for audio) | gRPC port. |

The embed NIM leaves performance mode disabled by default. To enable it on a
supported deployment, copy the complete `nimOperator.vlm_embed.env` list from
`values.yaml` into your values file. Then uncomment the optional setting:

```yaml
nimOperator:
  vlm_embed:
    env:
      - name: NIM_HTTP_API_PORT
        value: "8000"
      - name: NIM_TRITON_LOG_VERBOSE
        value: "1"
      - name: OMP_NUM_THREADS
        value: "1"
      - name: NIM_ENGINE_COUNT
        value: "1"
      # Optional: enable if your NIM build supports throughput mode for this SKU.
      # - name: NIM_PERFORMANCE_MODE
      #   value: "1"
```

> The four "core" NIMs (page_elements, table_structure, ocr, vlm_embed)
> are enabled and auto-wired by default. Optional NIMs stay off until
> `nimOperator.<key>.enabled` is `true`. When you opt in, the chart
> auto-wires Omni captioning and VL reranking into `nim_endpoints`
> (refer to [Image captioning (Omni 30B)](#image-captioning-omni-30b) and
> [Query-time reranking](#query-time-reranking)); other optional NIMs
> still need an explicit serviceConfig hook (for example
> `audioGrpcEndpoint` for Parakeet ASR). For minimal installs, prefer the
> [minimal install](#recommended-minimal-install-2682) overrides.

#### Filtering cached GPU profiles { #filtering-cached-gpu-profiles }

Every NIMCache the chart renders supports the NIM Operator's
`spec.source.ngc.model` block, which restricts which model profiles the
cache job downloads. The chart exposes this through two values:

| Path | Scope | Behaviour |
| ---- | ----- | --------- |
| `nimOperator.modelProfile` | Chart-wide | Applied to every NIMCache that doesn't carry its own override. |
| `nimOperator.<key>.modelProfile` | Per-NIM | When non-empty, **REPLACES** the chart-wide default (no merge). |

The chart-wide value and all per-NIM values, including `answer_llm`,
default to `{}`. With both values empty, the chart emits no `model:`
block and the NIM Operator falls back to its "cache every profile
applicable to the detected GPUs" default — fine on a single-GPU
laptop, but on heterogeneous clusters (or any cluster with ≥ 3 NIMs)
this wastes tens of GiB of PVC storage, NGC bandwidth, and cache-job
runtime.

For `answer_llm`, a single explicit profile in the effective cache
configuration also sets `NIMService.spec.storage.nimCache.profile`.
The service then requests the same explicit profile as its cache. A non-empty per-NIM configuration replaces the
chart-wide configuration for both cache and runtime profile selection.
With no explicit profile, GPU filters alone, or multiple profiles, the
chart leaves runtime profile selection automatic.

An explicit `NIM_MODEL_PROFILE` entry in `nimOperator.answer_llm.env`,
including a `valueFrom` entry, takes precedence: the chart omits the
generated storage profile. Keep that override consistent with the
cache configuration to avoid downloading another profile at startup.
To remove a configured pin, set `nimOperator.answer_llm.modelProfile.profiles=[]`.
If you also clear the per-NIM configuration entirely, the chart-wide
configuration applies.

The mapping is rendered verbatim under `spec.source.ngc.model`, so the
shape lines up 1:1 with the [NIMCache CRD](https://docs.nvidia.com/nim-operator/latest/reference-nimcache.html).
Two filter dimensions are supported (use whichever fits your cluster;
`gpus` is the common case):

```yaml
nimOperator:
  modelProfile:
    gpus:
      # NIMCache only downloads profiles compatible with at least one
      # of these GPU selectors. Each selector is {ids: [...], product: ...}.
      - ids: ["26B5"]                       # PCI device ID(s)
        product: "NVIDIA-H100-80GB-HBM3"    # NVIDIA marketing name
    # profiles:
    #   # Alternative: list of exact profile UUIDs from `ngc registry
    #   # model list-profiles <repo>/<image>:<tag>`.
    #   - "11111111-2222-3333-4444-555555555555"
```

Equivalent overrides via `--set`:

```bash
# Homogeneous H100 80 GB cluster — every NIMCache only pulls the H100 profile:
helm upgrade --install retriever ./nemo_retriever/helm \
  --set 'nimOperator.modelProfile.gpus[0].ids[0]=26B5' \
  --set 'nimOperator.modelProfile.gpus[0].product=NVIDIA-H100-80GB-HBM3'

# Restrict only the page_elements NIMCache to a specific profile UUID, leave the rest alone:
helm upgrade --install retriever ./nemo_retriever/helm \
  --set 'nimOperator.page_elements.modelProfile.profiles[0]=11111111-2222-3333-4444-555555555555'

# Chart-wide H100 default plus a per-NIM override (the override REPLACES the global; it does NOT merge):
helm upgrade --install retriever ./nemo_retriever/helm \
  --set 'nimOperator.modelProfile.gpus[0].product=NVIDIA-H100-80GB-HBM3' \
  --set 'nimOperator.vlm_embed.modelProfile.profiles[0]=22222222-3333-4444-5555-666666666666'
```

Tips:

- Run `ngc registry model list-profiles nvcr.io/nim/nvidia/<image>:<tag>` to enumerate the available profiles for any chart-pinned NIM image and pick the smallest profile that matches your GPU.
- Filter mismatches surface as `NIMCache` events such as `NoCompatibleProfile`; check with `kubectl describe nimcache <name>`.
- Empty effective cache configurations preserve operator profile discovery, including the default `answer_llm` configuration.

#### Image tag conventions { #image-tag-conventions }

Every NIM in this chart pins an exact NGC image tag in `values.yaml`
— there is no `:latest` floating reference. Two tag families show up:

| Family | Example | Meaning |
| ------ | ------- | ------- |
| Plain semver | `nemotron-object-detection:2.0.1` | A standard NIM release, identical bytes on every pull. Used by the four core NIMs and the reranker / ASR NIMs. |
| `<semver>-variant` | `nemotron-parse-v1.2:1.7.0-variant`, `nemotron-3-nano-omni-30b-a3b-reasoning:2.0.4-variant` | The Nemotron Parse and Nemotron 3 Nano Omni 30B builds that ship per-GPU TensorRT engine variants the NIM Operator selects from at reconciliation time (refer to the Omni and Parse rows in the [model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements) table). The `-variant` suffix is the NGC tag that ships alongside this chart and matches footnote ³ of the support matrix. |

For air-gapped mirror pipelines: mirror the *exact* tag — both the
plain semver and the `-variant` form — and do not substitute `:latest`.
Substituting `:latest` would pin to a moving target that may not match
the engine plans the NIM Operator profile expects for a given GPU.

If you want a different NIM build on a **new** install, override the tag
explicitly:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true \
  --set nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.image.tag=<your-tag>
```

and validate against the same release of the retriever service before
production rollout.

If a `NIMCache` for that NIM already exists, do not change
`image.repository` or `image.tag` with `helm upgrade` alone.
The NIM Operator rejects in-place `modelPuller` updates.
Delete the cache first, then upgrade. Refer to
[Changing a NIM image repository or tag](#changing-nim-image-repository-or-tag).

**Charts and captioning.** Charts and infographics use **Page Elements, Table Structure**
and **ocr**. For image
captioning, set `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true` — refer to
[Image captioning (Omni 30B)](#image-captioning-omni-30b) for the
chart-side wiring and
[Image captioning](https://docs.nvidia.com/nemo/retriever/latest/extraction/prerequisites-support-matrix/#image-captioning)
for the product matrix.

#### Changing a NIM image repository or tag { #changing-nim-image-repository-or-tag }

Each chart NIM renders a `NIMCache` whose `metadata.name` stays the same
across image changes. `spec.source.ngc.modelPuller` is the concatenation
of `nimOperator.<key>.image.repository` and `nimOperator.<key>.image.tag`.
The NIM Operator `NIMCache` CRD (including 3.1.2) marks `modelPuller`
immutable. Kubernetes rejects an update with a message similar to:

```text
modelPuller is an immutable field. Please create a new NIMCache resource instead when you want to change this container.
```

A `helm upgrade` that only changes the repository or tag fails
on the existing object. Other release resources can already have been
applied before that rejection, which leaves the release partially upgraded.

Do not rename `nimOperator.<key>.nimServiceName` as a workaround.
A new name creates a second cache and Service DNS label. The previous
`NIMCache` remains, especially when `nimOperator.nimCache.keepOnUninstall`
is `true` (the default).

The following table lists default `NIMCache` names for the core NIMs.
Optional NIMs follow the same immutable-`modelPuller` rule.

| Helm key | Default `NIMCache` name |
| --- | --- |
| `nimOperator.page_elements` | `nemotron-page-elements-v3` |
| `nimOperator.table_structure` | `nemotron-table-structure-v1` |
| `nimOperator.ocr` | `nimOperator.ocr.nimServiceName` (`nemotron-ocr-v2`) |
| `nimOperator.vlm_embed` | `nimOperator.vlm_embed.nimServiceName` (`nemotron-3-embed-1b`) |

**Before you change a repository or tag** on an existing release,
complete the following steps for every NIM whose image changes.
The affected NIM is unavailable while the operator re-caches weights.

1. Drain ingest traffic that depends on that NIM.
2. Confirm the live `modelPuller` value differs from the new
   `repository:tag`. Set `NS` to the namespace of your Helm release:

   ```bash
   NS=default
   CACHE=nemotron-page-elements-v3

   kubectl get nimcache "${CACHE}" -n "${NS}" \
     -o jsonpath='{.metadata.name}{" "}{.spec.source.ngc.modelPuller}{"\n"}'
   ```

   Compare that `modelPuller` value with
   `<new-repository>:<new-tag>`.
3. Delete the `NIMCache`. Helm `keep` annotations do not block
   `kubectl delete`:

   ```bash
   kubectl delete nimcache "${CACHE}" -n "${NS}"
   ```

4. If the operator-created PVC remains, delete it so the new image
   re-pulls weights. List operator PVCs, then delete the claim for
   that cache. Default claim names use a `-pvc` suffix, for example
   `nemotron-page-elements-v3-pvc`. Confirm the name from the list
   before you delete it. Refer to
   [Persistent storage prerequisite](#persistent-storage-prerequisite).

   ```bash
   kubectl get pvc -n "${NS}" -l 'app.kubernetes.io/managed-by=nvidia-nim-operator'
   kubectl delete pvc "${CACHE}-pvc" -n "${NS}"
   ```

5. Run `helm upgrade` with the new `image.repository` or `image.tag`.
   Helm creates a new `NIMCache` with the updated `modelPuller`.
6. Wait until the new cache is ready before you send traffic:

   ```bash
   kubectl get nimcache "${CACHE}" -n "${NS}"
   kubectl get nimservice "${CACHE}" -n "${NS}"
   kubectl describe nimcache "${CACHE}" -n "${NS}"
   ```

Repeat those steps for each NIM whose repository or tag changes,
including chart default tag bumps between releases.

`helm uninstall` does not remove kept `NIMCache` objects when
`keepOnUninstall` is `true`. A later install or upgrade with a
different image still requires this delete and re-cache sequence.

**If `helm upgrade` already failed** with the immutable-`modelPuller`
message, delete the existing `NIMCache` and its PVC, then re-run the
same upgrade so Helm creates the cache instead of patching it.

Changing `service.image.repository` or `service.image.tag` does not
use `NIMCache` and is not subject to this rule.

#### Query-time reranking { #query-time-reranking }

The VL reranker NIM (`llama-nemotron-rerank-vl-1b-v2`) backs
`POST /v1/query` with `rerank=true`. When you enable it,

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.rerankqa.enabled=true
```

the chart auto-wires two fields into the rendered
`retriever-service.yaml` ConfigMap (including every split-topology
role ConfigMap):

```yaml
nim_endpoints:
  rerank_invoke_url: "http://llama-nemotron-rerank-vl-1b-v2:8000/v1/ranking"
  rerank_model_name: "nvidia/llama-nemotron-rerank-vl-1b-v2"
```

Without those fields the gateway returns
`HTTP 400 Reranking is not configured` even when the Rerank NIM Pod is
Ready.

Resolution order mirrors every other NIM endpoint (see the
[NIM Operator sub-stack](#nim-operator-sub-stack) section):

1. Explicit `serviceConfig.nimEndpoints.rerankInvokeUrl` always wins
   (use this to point at a hosted or external ranking endpoint).
2. Otherwise the operator-managed URL of
   `llama-nemotron-rerank-vl-1b-v2` is used, provided
   `nimOperator.rerankqa.enabled=true`, `nims.enabled=true`,
   **and** the `apps.nvidia.com/v1alpha1` CRDs are installed.
3. Otherwise `rerank_invoke_url` stays `null` and query-time reranking
   stays disabled.

`serviceConfig.nimEndpoints.rerankModelName` follows the same order —
it defaults to the canonical VL reranker model id
(`nvidia/llama-nemotron-rerank-vl-1b-v2`) whenever the chart resolves any
rerank URL. Override only when pointing at a different ranking SKU.


#### Image captioning (Omni 30B) { #image-captioning-omni-30b }

The Nemotron 3 Nano Omni VLM is the canonical image-caption NIM for
this chart. When you enable it,

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true
```

the chart now auto-wires two fields into the rendered
`retriever-service.yaml` ConfigMap:

```yaml
nim_endpoints:
  caption_invoke_url: "http://nemotron-3-nano-omni-30b-a3b-reasoning:8000/v1/chat/completions"
  caption_model_name: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
```

The service derives `caption_enabled=true` from a non-null
`caption_invoke_url`, so the ingestion pipeline routes caption work to
the in-cluster Omni Pod with no manual ConfigMap edits.

Resolution order mirrors every other NIM endpoint (see the
[NIM Operator sub-stack](#nim-operator-sub-stack) section):

1. Explicit `serviceConfig.nimEndpoints.captionInvokeUrl` always wins
   (use this to point at a hosted endpoint, e.g.
   `https://integrate.api.nvidia.com/v1/chat/completions`).
2. Otherwise the operator-managed URL of
   `nemotron-3-nano-omni-30b-a3b-reasoning` is used, provided
   `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true`
   **and** the `apps.nvidia.com/v1alpha1` CRDs are installed.
3. Otherwise `caption_invoke_url` stays `null` and the caption stage
   is disabled.

`serviceConfig.nimEndpoints.captionModelName` follows the same order —
it defaults to the canonical Omni remote model id
(`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, matching
`nemo_retriever.common.modality.caption.model_profiles.OMNI_REMOTE_MODEL_ID`) whenever
the chart resolves any caption URL. Override only when pointing at a
different VLM SKU.

#### GPU limits and `helm upgrade` { #gpu-limits-and-helm-upgrade }

The chart defaults to **`nimOperator.nimServiceGpuLimit: 1`**, which
renders `spec.resources.limits.nvidia.com/gpu: 1` on every NIMService
unless a per-NIM `resources` map overrides it. A default core install
therefore requests **four GPU slots** (one per enabled NIMService).
Setting that per-NIM limit is required on NIM Operator **v3.1.2**
(and other versions tested on A100/H100): when the chart omits the
`resources` block entirely, the operator often **does not** populate
GPU limits from the model profile, and NIM pods start without GPU
access (`The NVIDIA Driver was not detected`).

This per-NIM `1` request is not the support-matrix VRAM figure of
one A10G for all four core models combined. On a conventional
cluster, plan for four allocatable GPUs across eligible nodes or
configure sharing.
Refer to [GPU scheduling prerequisite](#gpu-scheduling-prerequisite).

**Trade-off:** Helm and the NIM Operator may both server-side-apply
`spec.resources.limits.nvidia.com/gpu`. A later `helm upgrade --install`
can then fail with:

```
Error: UPGRADE FAILED: conflict occurred while applying object
  <ns>/<nim> apps.nvidia.com/v1alpha1, Kind=NIMService:
  Apply failed with 1 conflict:
  conflict with "manager" using apps.nvidia.com/v1alpha1:
    .spec.resources.limits.nvidia.com/gpu
```

**Operator-only mode** (omit GPU limits from Helm — only if your NIM
Operator version reliably reconciles them):

```yaml
nimOperator:
  nimServiceGpuLimit: null
```

**If upgrades hit SSA conflicts** after the operator has reconciled GPU
limits, use one of:

1. `helm upgrade --install … --force-conflicts --server-side`
2. `kubectl -n <ns> edit nimservice <name>` to set GPU limits outside Helm

To pin a non-default GPU count chart-wide, set `nimServiceGpuLimit: 2`
(or set per-NIM `resources.limits.nvidia.com/gpu`).

A failed upgrade that reports an immutable `modelPuller` field is a
different class of error. Refer to
[Changing a NIM image repository or tag](#changing-nim-image-repository-or-tag).

### OCR NIM configuration { #ocr-nim-configuration }

The core OCR NIM is configured under [`nimOperator.ocr`](./values.yaml) (the `ocr:`
block). Confirm `image.repository` and `image.tag` before you upgrade.
If either value changes on an existing release, delete the OCR `NIMCache`
before you upgrade. Refer to
[Changing a NIM image repository or tag](#changing-nim-image-repository-or-tag).

| Path | Role |
|------|------|
| `nimOperator.nimCache.keepOnUninstall` | When `true`, NIMCache CRs survive `helm uninstall` (`helm.sh/resource-policy: keep`). NIMService CRs are always removed. Set `false` for dev clusters that should fully tear down on uninstall. |
| `nimOperator.ocr.enabled` | Reconcile the OCR `NIMService` |
| `nimOperator.ocr.image.repository` | NIM image (default `nvcr.io/nim/nvidia/nemotron-ocr-v2`) |
| `nimOperator.ocr.image.tag` | Pin the image tag for reproducible upgrades |

Override the auto-wired in-cluster URL with `serviceConfig.nimEndpoints.ocrInvokeUrl`
when the OCR service runs outside the operator sub-stack.

### Persistence

Complete the [persistent storage prerequisite](#persistent-storage-prerequisite)
before a default install. Empty `storageClass` values omit
`storageClassName` and rely on a default StorageClass or a compatible
classless persistent volume.

| Path                       | Default                       | Notes |
|----------------------------|-------------------------------|-------|
| `persistence.enabled`      | `true`                        | Mount the pre-existing general PVC for logs and other non-scheduler uses. |
| `persistence.size`         | `50Gi`                        |       |
| `persistence.accessModes`  | `[ReadWriteOnce]`             | Access mode for the general PVC. |
| `persistence.storageClass` | `""`                          | Use cluster default unless set. Use `"-"` to disable a `storageClassName`. |
| `persistence.existingClaim` | `""`                         | When set, skip PVC creation and mount this claim. |
| `persistence.mountPath`    | `/var/lib/nemo-retriever`     | General persistent files only; scheduler state and payloads are never stored here. |
| `retrieverResults.enabled` | `true`                        | Create the results PVC unless `existingClaim` is set. |
| `retrieverResults.storageClass` | `""`                     | Use cluster default unless set. Use `"-"` to disable a `storageClassName`. |
| `retrieverResults.existingClaim` | `""`                     | When set, skip PVC creation and mount this claim. |
| `topology.vectordb.persistence.enabled` | `true`           | Create the VectorDB PVC when `serviceConfig.vectordb.enabled` is `true`. |
| `topology.vectordb.persistence.storageClass` | `""`        | Use cluster default unless set. Use `"-"` to disable a `storageClassName`. No `existingClaim` path. |

The gateway enforces active-lease budgets independently of worker replicas.
`serviceConfig.workQueue.maxActiveLeases.realtime` defaults to `8` and
`.batch` defaults to `48`; treat these as explicit downstream-capacity budgets,
not values inferred from pod or NIM counts.

#### Scheduler loss boundary and upgrade from durable releases

The work spool always uses `serviceConfig.workQueue.spoolDirectory` under the
gateway `/tmp` `emptyDir`; `persistence.enabled` does not change scheduler
behavior. During one gateway lifetime, FIFO order, lease caps, delivery attempts,
generations, stale-lease rejection, worker recovery after lease expiry, and
queued-plus-active demand metrics remain available.

Replacing the gateway loses accepted jobs, queued payloads, active leases, job
status history, and SSE catch-up state. After replacement, status and event
requests for old jobs return not found, and old callbacks and heartbeats return
`409`. Clients must create a new job and submit the documents again. Worker loss
remains recoverable through lease expiry only while the same gateway process is
alive. Public ingest and worker HTTP wire formats are unchanged.

Before upgrading from a release with durable scheduler checkpoints, drain the
gateway. The ephemeral implementation neither reads nor automatically deletes
`gateway-state.sqlite3` or payload files left under an older PVC. After rollback
to that release is no longer required, operators may manually remove its old
`work-queue` directory from the general PVC. The removed
`work_queue.persistence_enabled` key was internal and unreleased; delete it from
custom service configuration files.

### Secrets

| Path                              | Default        | Notes |
|-----------------------------------|----------------|-------|
| `ngcImagePullSecret.create`       | `false`        | Chart-managed dockerconfigjson Secret. |
| `ngcImagePullSecret.name`         | `ngc-secret`   | Name referenced by every Pod and, when per-NIM `image.pullSecrets` is empty, by every NIMCache / NIMService. |
| `ngcImagePullSecret.password`     | `""`           | NGC API key. |
| `ngcApiSecret.create`             | `false`        | Chart-managed Opaque Secret. |
| `ngcApiSecret.name`               | `ngc-api`      | Name referenced by NIMCache/NIMService `authSecret` when per-NIM `authSecret` is empty. |
| `ngcApiSecret.password`           | `""`           | NGC API key (populates `NGC_API_KEY` + `NGC_CLI_API_KEY`). |
| `imagePullSecrets`                | `[]`           | Extra pre-existing pull secrets appended to every Pod. |
| `serviceConfig.vectordb.internalAuth.enabled` | `false` | Enable the Secret-backed credential for VectorDB traffic and restricted gateway-to-worker handoffs. |
| `serviceConfig.vectordb.internalAuth.existingSecret.name` | `""` | Existing Secret shared by Retriever and VectorDB pods. |
| `serviceConfig.auth.scopeTokenSecret.name` | `""` | Existing Secret containing the public scope-token JSON file. |
| `serviceConfig.auth.enabled` | `false` | Require bearer authentication for the public gateway. |
| `serviceConfig.auth.allowInsecureInlineApiToken` | `false` | Explicit development-only gate for ConfigMap-backed `apiToken`. |

### Optional features

| Feature           | Toggle                          | Default |
|-------------------|---------------------------------|---------|
| Ingress           | `ingress.enabled`               | `true`  |
| Autoscaling (HPA) | `autoscaling.enabled`           | `false` (max=1 anyway) |
| ServiceMonitor    | `serviceMonitor.enabled`        | `false` |

---

## Configuration recipes

### Mount a custom retriever-service.yaml verbatim

The chart renders `retriever-service.yaml` from structured values so you
shouldn't normally need to ship a verbatim file. If you really want to,
mount one via `service.extraVolumes` + `service.extraVolumeMounts` at
`/etc/nemo-retriever/retriever-service.yaml` (which silently overrides the
chart-managed ConfigMap because `subPath` mounts win).

### Use externally managed Secrets

```yaml
ngcImagePullSecret:
  create: false        # don't render; reference an existing Secret
  name: my-org-ngc-pull
ngcApiSecret:
  create: false
  name: my-org-ngc-api
```

The chart will skip Secret creation. Make sure `my-org-ngc-pull` exists
as `kubernetes.io/dockerconfigjson` and `my-org-ngc-api` as `Opaque` with
an `NGC_API_KEY` key, in the release namespace. Retriever Pods and every
rendered NIMCache / NIMService inherit those names unless you set a
non-empty per-NIM `image.pullSecrets` or `authSecret` override.

Protect the public gateway and internal service calls with separate
pre-existing Secrets:

```yaml
serviceConfig:
  auth:
    scopeTokenSecret:
      name: nrl-public-auth
      key: scope-tokens.json
    enabled: true
  vectordb:
    internalAuth:
      enabled: true
      existingSecret:
        name: nrl-internal-vdb-auth
        key: token
```

`nrl-public-auth` must contain a JSON document such as
`{"tokens":[{"token":"<secret>","scopes":["workspace-123"]}]}` under the
configured key. `nrl-internal-vdb-auth` must contain a distinct, high-entropy
credential. In split topology, the public token file mounts only on the
gateway. The gateway authenticates public requests, then uses the internal
credential for restricted worker handoffs and pull-worker claims of
`/v1/internal/work/claim`. Workers do not receive or validate the public
bearer token, so `serviceConfig.auth.scopeTokenSecret.name` also requires
`serviceConfig.vectordb.internalAuth.enabled=true`. Internal authentication
is opt-in for local compatibility; enable it for production deployments. When
enabled, a missing Secret or key prevents the pods from starting instead of
falling back to unauthenticated VectorDB access. Inline
`serviceConfig.auth.apiToken` is rejected unless
`allowInsecureInlineApiToken=true`, and must never be used for production.

### Disable one NIM and supply an external URL for it

```yaml
nimOperator:
  vlm_embed:
    enabled: false   # don't deploy the embed NIM in-cluster

serviceConfig:
  nimEndpoints:
    embedInvokeUrl: https://integrate.api.nvidia.com/v1/embeddings
```

The chart's resolution order is **explicit URL → operator-managed URL →
empty**, so per-endpoint overrides Just Work.

### Roll the service after editing values

The `Deployment` carries a `checksum/config` annotation derived from the
ConfigMap, so `helm upgrade` automatically rolls the pod when any
`serviceConfig.*` value changes.

---

## Timeouts and alleviating ingest failures

Batch ingest fans out extract and embed work to remote NIM HTTP endpoints.
Under heavy parallelism a single slow or overloaded NIM can cause timeouts,
and a worker process crash can surface as many simultaneous `failed`
document callbacks even though only one root cause occurred.

### What the chart configures

| Layer | Default | Where it is set |
|-------|---------|-----------------|
| Remote embed HTTP calls | **600 s** (10 min) | Service image (`EmbedParams.request_timeout_s`); not a Helm value today. |
| Gateway → realtime/batch proxy | **300 s** | Rendered `gateway.timeout_s` in `retriever-service.yaml` (split topology). |
| Worker → vectordb record write | **300 s** | `serviceConfig.vectordb.writeTimeoutSeconds`, rendered as `vectordb.write_timeout_s` in `retriever-service.yaml`. |
| Embed model name | `serviceConfig.vectordb.embedModel` | Also copied into worker `nim_endpoints.embed_model_name` in the ConfigMap. |

Symptoms to look for in pod logs:

- `Embedding error occurred: timed out` or `httpx.ReadTimeout` on the **batch** pod.
- `Batch process pool broken (worker crash)` followed by many
  `BrokenProcessPool` failures on other in-flight documents.
- Embed NIM pod messages such as `failed to allocate pinned system memory`
  (GPU pressure from too many concurrent `/v1/embeddings` requests).
- `Failed to POST <n> records to vectordb for <file>` on the **batch** or
  **realtime** pod, followed by a `failed` document. The vectordb Pod did not
  acknowledge the record write in time.

The **gateway** pod usually only logs `status=failed` callbacks; diagnose on
**batch** (and **realtime** for page-sized uploads), plus the embed NIM pod.

### Recommended mitigations

**1. Lower batch worker concurrency (first step).**

The default `serviceConfig.pipeline.batchWorkers` is `48`, which can saturate
a single in-cluster embed NIM. If you see embed timeouts or pool crashes,
reduce batch parallelism to **16** and redeploy:

```bash
helm upgrade retriever ./nemo_retriever/helm \
  --reuse-values \
  --set serviceConfig.pipeline.batchWorkers=16
```

You can tune further (for example `8` on small GPU nodes), but **16** is a
reasonable starting point when moving off the default. Realtime workers
(`realtimeWorkers`, default `24`) are less likely to overload embed NIMs
because they handle smaller units of work; adjust them only if realtime
ingest shows the same timeout pattern.

**2. Confirm embed wiring.**

Ensure `nim_endpoints.embed_model_name` in the mounted config matches the
embed NIM SKU (`serviceConfig.vectordb.embedModel`, default
`nvidia/nemotron-3-embed-1b`). A model mismatch produces
HTTP 404 on `/v1/embeddings`, not a timeout, but is worth ruling out when
debugging failed ingests.

**3. Retry failed documents.**

Failures caused by a one-time pool restart are often transient. After lowering
`batchWorkers` and rolling the batch Deployment, resubmit documents that
failed with `rows=0`.

**4. Scale or isolate the embed NIM.**

If timeouts persist at `batchWorkers: 16`, add embed NIM replicas (when your
cluster has GPU capacity), point `serviceConfig.nimEndpoints.embedInvokeUrl`
at an external embed endpoint, or temporarily disable optional NIMs on
dev clusters to free GPU memory for `vlm_embed`.

**5. Client and ingress timeouts.**

Long batch jobs may exceed the gateway proxy timeout (300 s) or an Ingress
`proxy-read-timeout`. Increase ingress annotations if clients disconnect
while workers are still processing; see the commented example on
`ingress.annotations` in `values.yaml`.

**6. Raise the vectordb write timeout on slow storage.**

A worker fails the document when the vectordb Pod does not acknowledge its
record write:

```text
VectorDB write failed for report.pdf: <error>. The extracted rows are not
confirmed durable, so the document is not queryable.
```

That acknowledgement covers the row commit plus the index maintenance that
follows it, so a large batch on slow storage can exceed the default 300 s.
Raise the timeout and redeploy:

```bash
helm upgrade retriever ./nemo_retriever/helm \
  --reuse-values \
  --set serviceConfig.vectordb.writeTimeoutSeconds=900
```

Concurrent writes to one vectordb Pod do not block each other, so a write
timeout points at storage throughput or table size rather than write
serialization. Refer to
[LanceDB index creation fails during concurrent Helm ingestion](https://docs.nvidia.com/nemo/retriever/latest/extraction/troubleshoot/#lancedb-concurrent-index-creation).

---

## Queue-depth autoscaling (split mode)

In `topology.mode: split` deployments the realtime and batch worker
pods scale horizontally based on the gateway's **central outstanding demand** and
**95th-percentile processing latency**. Demand is queued records plus active leases from the gateway while
latency comes from workers; both publishers are always on (see
`nemo_retriever_work_queue_demand` in
[`prometheus.py`](../src/nemo_retriever/service/services/prometheus.py)).
The only choice you have to make is **how the metrics get from
Prometheus into the Kubernetes HPA**.

### Why queue depth (and not CPU)

CPU-based HPA reacts to *the pod that has already saturated its work*.
For an ingest pipeline that fans out to remote NIM endpoints, the work
spends most of its time blocked on HTTP — CPU stays low even when the
queue is full. Queue depth measures *demand to be served*, which is
what we actually want to scale on. A 95th-percentile-latency signal
rides alongside to catch the inverse case (a single hot pod whose
queue is shallow but whose per-item processing has stalled).

### Backend choices

The chart's `autoscaling.queueDepth.backend` controls which path is
wired up. All three options leave the metrics publisher untouched:

| backend                | When to pick it                                                  | Cluster prerequisite              |
|------------------------|------------------------------------------------------------------|-----------------------------------|
| `prometheus-adapter` *(default)* | Production. One adapter feeds HPA + Grafana + future autoscalers. | Prometheus Operator + `prometheus-community/prometheus-adapter`. |
| `cpu`                  | Bootstrap / dev cluster without Prometheus.                      | None — built-in.                   |
| `keda`                 | Already standardised on KEDA org-wide.                           | KEDA operator (you install + apply your own `ScaledObject`). |

The chart-recommended path is `prometheus-adapter`. The reasoning is
documented in `values.yaml`; in short, it keeps a single Prometheus as
the source of truth, supports HPA's multi-metric arithmetic-mean
evaluation out of the box, and doesn't force the chart to bundle new
CRDs.

### Wiring up prometheus-adapter (recommended)

The chart renders a ConfigMap named
`<release>-nemo-retriever-prom-adapter-rules` containing PromQL rules
for the External Metrics API. You point your existing
prometheus-adapter at it:

```bash
helm upgrade prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  --reuse-values \
  --set rules.existing=<release>-nemo-retriever-prom-adapter-rules
```

Then verify both metrics show up in the External Metrics API:

```bash
kubectl get --raw \
  "/apis/external.metrics.k8s.io/v1beta1/namespaces/$NS/nemo_retriever_gateway_work_queue_backlog?labelSelector=pool%3Drealtime" \
  | jq .
```

Once that returns a non-empty `items` array, the HPAs rendered by this
chart will start consuming them. The HPA annotation
`nemo-retriever.nvidia.com/hpa-signals` documents the active set per
HPA, e.g. `queueRatio=true latencyP95=true cpu=false`.

### CPU fallback (no Prometheus required)

Set `autoscaling.queueDepth.backend: cpu` and enable the CPU metric
under each role:

```yaml
autoscaling:
  queueDepth:
    backend: cpu
topology:
  realtime:
    hpa:
      metrics:
        queueBacklog: { enabled: false }
        processingLatencyP95: { enabled: false }
        cpu: { enabled: true, targetUtilizationPercentage: 60 }
  batch:
    hpa:
      metrics:
        queueBacklog: { enabled: false }
        processingLatencyP95: { enabled: false }
        cpu: { enabled: true, targetUtilizationPercentage: 80 }
```

The legacy `topology.<role>.hpa.targetCPUUtilizationPercentage` field
still works and behaves as an alias for the `metrics.cpu` block.

### KEDA path

Set `autoscaling.queueDepth.backend: keda` and disable the chart-managed
HPAs:

```yaml
autoscaling:
  queueDepth: { backend: keda }
topology:
  realtime: { hpa: { enabled: false } }
  batch:    { hpa: { enabled: false } }
```

Then apply your own `ScaledObject` for the realtime pool.
`spec.scaleTargetRef.name` must match the rendered realtime Deployment
in the same namespace. For the documented quickstart release `retriever`,
that Deployment is `retriever-nemo-retriever-realtime`. The chart names
that Deployment from the Helm fullname plus the `-realtime` suffix, so a
different release produces a different name. Discover the rendered name
with the `app.kubernetes.io/instance` and `app.kubernetes.io/component`
labels:

```bash
kubectl get deploy \
  -l app.kubernetes.io/instance=retriever,app.kubernetes.io/component=realtime \
  -o jsonpath='{.items[0].metadata.name}{"\n"}'
```

Replace `retriever` in the label selector with your Helm release name.
Repeat the same pattern for the batch pool with
`app.kubernetes.io/component=batch`. The following example uses the
documented quickstart realtime Deployment:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: retriever-nemo-retriever-realtime
spec:
  scaleTargetRef:
    name: retriever-nemo-retriever-realtime
  minReplicaCount: 2
  maxReplicaCount: 8
  cooldownPeriod: 300
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        metricName: nemo_retriever_pool_queue_depth_ratio
        threshold: "0.5"
        query: |
          avg by (pool) (
            nemo_retriever_pool_queue_depth{pool="realtime"}
            /
            on(pool, instance) group_left()
            nemo_retriever_pool_max_queue_size{pool="realtime"}
          )
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        metricName: nemo_retriever_pool_processing_duration_p95
        threshold: "30"
        query: |
          histogram_quantile(
            0.95,
            sum by (le, pool) (
              rate(nemo_retriever_pool_processing_duration_seconds_bucket{pool="realtime"}[2m])
            )
          )
```

KEDA's biggest win is **scale-from-zero**, which we don't use today —
both `minReplicas` defaults are ≥ 1 because the realtime pod is on the
hot path for SSE consumers. If you do want scale-from-zero (e.g. a
nightly batch-only job tenant), KEDA is the right tool and this is the
escape hatch.

### Tuning the thresholds

Per-role tuning lives under `topology.<role>.hpa.metrics`:

```yaml
topology:
  realtime:
    hpa:
      metrics:
        queueBacklog: { enabled: true, target: "24" }
        processingLatencyP95: { enabled: true, targetSeconds: "30" }
  batch:
    hpa:
      metrics:
        queueBacklog: { enabled: true, target: "48" }
        processingLatencyP95: { enabled: true, targetSeconds: "120" }
```

The demand `target` is outstanding documents per replica because the HPA uses
`type: AverageValue`. When migrating an existing values file, rename
`metrics.queueDepthRatio` to `metrics.queueBacklog`, choose a document-count
target near the role's execution-slot count, and rename
`prometheusAdapter.queueDepthRatioMetric` to `queueBacklogMetric`. Both legacy
keys remain aliases for this release: legacy enable/disable values win, legacy
metric names name the new demand metric, and fractional ratio targets are
replaced by the role's backlog-count default with an HPA annotation and NOTES
warning. The aliases are scheduled for removal in the following release.

### Verifying it scales

```bash
# Cause realtime pressure (anything that submits to /v1/ingest/job/.../page).
# Then watch the HPA decide:
kubectl get hpa -w

# And watch the active signals on each HPA:
kubectl get hpa <release>-realtime -o jsonpath='{.metadata.annotations.nemo-retriever\.nvidia\.com/hpa-signals}'
```

The dashboard's *Worker Pool Capacity* card on the **Overview** page
mirrors the same signal Prometheus is seeing, so it's a quick eyeball
sanity check before opening Grafana.

---

## Tracing and Zipkin

Helm installs the chart-owned OpenTelemetry Collector and Zipkin backend on by
default. This is intentional: the legacy 26.1.2 Helm chart shipped with a
managed Zipkin deployment enabled, so the new chart keeps a default trace
backend available for functional parity. OTLP trace and metric export is also
enabled by default for retriever service pods and chart-managed NIMs:

```yaml
topology:
  otel:
    enabled: true
  zipkin:
    enabled: true

service:
  otel:
    enabled: true

nimOperator:
  otel:
    enabled: true
```

Override `topology.otel.image` and `topology.zipkin.image` when you mirror
`otel/opentelemetry-collector-contrib:0.127.0` and `openzipkin/zipkin:3.5.0`
into a private registry. Refer to [Air-gapped deployment](#air-gapped-deployment).

Because Zipkin is chart-owned by default, an upgrade with default values can
create a Zipkin Deployment and Service. Set `topology.zipkin.enabled=false`
before upgrading if your deployment uses an external backend or should not run
chart-owned Zipkin.

With default values, retriever service pods and chart-managed NIMs emit OTLP to
the chart's OpenTelemetry Collector. The Collector exports traces to the
chart-owned Zipkin service and exposes received metrics in Prometheus format.
The chart configures a 5-second metric export interval. Set
`service.otel.enabled=false` or `nimOperator.otel.enabled=false` to opt out by
surface.

In this section, replace `<release>` with the name you passed to
`helm install`. The default Service names are `<release>-nemo-retriever`,
`<release>-nemo-retriever-zipkin`, and `<release>-nemo-retriever-otel`.
If your release name already contains `nemo-retriever`, or if you set
`nameOverride` or `fullnameOverride`, copy the Service names from `kubectl get svc`
instead. Open a job and read the Zipkin lookup key from either
the JSON body or the `x-trace-id` response header:

```bash
kubectl port-forward svc/<release>-nemo-retriever 7670:7670

curl -s -D headers.txt -o job.json \
  -X POST http://localhost:7670/v1/ingest/job \
  -H 'content-type: application/json' \
  -d '{"expected_documents":1}'

TRACE_ID=$(jq -r .trace_id job.json)
grep -i x-trace-id headers.txt
```

Port-forward Zipkin and query the trace directly:

```bash
kubectl port-forward svc/<release>-nemo-retriever-zipkin 9411:9411
curl "http://localhost:9411/api/v2/trace/${TRACE_ID}"
```

### Prometheus metrics from the OpenTelemetry Collector

When `topology.otel.enabled=true`, the chart-owned OpenTelemetry Collector
exposes metrics received through OTLP in Prometheus format. The endpoint uses
`topology.otel.ports.prometheus`, which defaults to port `8889`. The
chart-owned OpenTelemetry Collector Service exposes the same port. With default
values, the retriever service and chart-managed NIMs export OTLP metrics to this
endpoint.

The retriever service also retains its native Prometheus `/metrics` endpoint.
Enable `serviceMonitor.enabled=true` when a Prometheus Operator should scrape
that endpoint directly. Direct scraping remains useful for service metrics that
do not use OTLP, including the worker-pool metrics used by split-mode
autoscaling.

After a successful ingestion, allow up to 30 seconds for metric export and
verify the endpoint by port-forwarding the Collector Service:

```bash
kubectl port-forward svc/<release>-nemo-retriever-otel 8889:8889
metric_found=false
for attempt in {1..30}; do
  if curl -fsS http://127.0.0.1:8889/metrics | grep -q '^nemo_retriever_'; then
    metric_found=true
    break
  fi
  sleep 1
done
test "${metric_found}" = true
```

Set `topology.otel.ports.prometheus` to use a different port. The chart updates
the Collector listener and Service port together. If the command does not find
a metric after 30 seconds, confirm that `topology.otel.enabled` and the
workload's OpenTelemetry settings are enabled, then inspect the Collector logs.

Common opt-out and override knobs:

```yaml
topology:
  zipkin:
    enabled: false                 # do not deploy chart-owned Zipkin
    exporter:
      enabled: false               # keep Zipkin deployed, but do not export traces to it
      endpoint: http://external-zipkin:9411/api/v2/spans

service:
  otel:
    enabled: true                  # required for the following service overrides to render
    env:
      OTEL_METRICS_EXPORTER: none  # retain tracing, but do not export service metrics
      OTEL_METRIC_EXPORT_INTERVAL: "10000" # service metric cadence in milliseconds
# To opt out of all service OTLP telemetry instead, set `service.otel.enabled: false`.

nimOperator:
  otel:
    enabled: true                  # required for the following inherited NIM override to render
    env:
      NIM_OTEL_METRICS_EXPORTER: "console" # do not send inherited NIM metrics to OTLP
# To opt out of all inherited NIM OTLP telemetry instead, set `nimOperator.otel.enabled: false`.
  page_elements:
    otel:
      enabled: false               # per-NIM opt-out
  ocr:
    otel:
      env:
        NIM_OTEL_METRICS_EXPORTER: "console" # per-NIM metric-export opt-out
        TRITON_OTEL_RATE: "10"     # per-NIM Triton OTel override
```

Set `topology.zipkin.exporter.endpoint` when you run your own Zipkin-compatible
collector. Set `topology.otel.enabled=false` to disable the chart-owned collector
and all chart-rendered collector wiring. Values in `service.env` override the
chart-managed service OpenTelemetry environment variables. Existing NIM
container environment variables take precedence over inherited
`nimOperator.otel.env` values, and per-NIM `nimOperator.<key>.otel.env` values
override the inherited NIM values.

---

## OpenShift deployment { #openshift-deployment }

OpenShift install procedures, **restricted-v2** / PSA **restricted** value overrides, prebuilt `ffmpeg` images, internal registry pull secrets, optional NIM `LD_LIBRARY_PATH` tuning, and install examples are in **[OpenShift deployment](./openshift.md)**. Pass `-f openshift-restricted.yaml` from that guide when you install on OpenShift.

---

## Air-gapped deployment { #air-gapped-deployment }

Refer to [Deployment options — Air-gapped and disconnected deployment](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/#air-gapped-deployment) for overview and workflow. Chart-specific reference for mirroring:

### Container images to mirror (chart defaults)

Verify tags on the Git branch or tag you ship (for example `main` or
your release tag). Defaults below match
[`values.yaml`](./values.yaml) on the current chart.

| Role | `nimOperator` key | Default image (`repository:tag`) |
|------|-------------------|----------------------------------|
| Retriever service | — | `service.image.repository`:`service.image.tag` (override for production) |
| Page Elements | `page_elements` | `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1` |
| Table Structure | `table_structure` | `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1` |
| OCR | `ocr` | `nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0.1` |
| Text embed | `vlm_embed` | `nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.2` |
| VL reranker (optional) | `rerankqa` | `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:2.3.0` |
| Nemotron Parse (optional) | `nemotron_parse` | `nvcr.io/nim/nvidia/nemotron-parse-v1.2:1.7.0-variant` |
| Omni caption or configurable answer VLM (optional) | `nemotron_3_nano_omni_30b_a3b_reasoning` | `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:2.0.4-variant` |
| Answer LLM (optional, Nemotron 3.5 Lightning default) | `answer_llm` | `nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:2.0.9-variant` |
| Parakeet ASR (optional) | `audio` | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.5.0` |

GPU SKU support for `audio` is in [Model hardware requirements](https://github.com/NVIDIA/NeMo-Retriever/blob/main/docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements).

The following table lists default chart-owned auxiliary images. These images are not NIM Operator custom resources.

| Role | Values path | Default image (`repository:tag`) | When rendered |
|------|-------------|----------------------------------|---------------|
| OpenTelemetry Collector | `topology.otel.image` | `otel/opentelemetry-collector-contrib:0.127.0` | `topology.otel.enabled=true` (default) |
| Zipkin | `topology.zipkin.image` | `openzipkin/zipkin:3.5.0` | `topology.otel.enabled=true` and `topology.zipkin.enabled=true` (default) |
| BusyBox wait-for-gateway init | not overridable | `busybox:1.37` | `topology.mode=split` on realtime and batch workers |

Also mirror images for the vectordb sidecar, Redis, or other subcharts if
your values enable them.

### Helm values for a private registry

Example overrides (replace placeholders):

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  -f my-airgap-values.yaml
```

`my-airgap-values.yaml` should include at least:

```yaml
service:
  image:
    repository: <PRIVATE_REGISTRY>/nemo-retriever-service
    tag: <PINNED_TAG>
    pullPolicy: IfNotPresent

imagePullSecrets:
  - name: my-private-registry

ngcImagePullSecret:
  create: false
  name: ""   # Explicitly empty — clears the default "ngc-secret"

nimOperator:
  page_elements:
    image:
      repository: <PRIVATE_REGISTRY>/nemotron-object-detection
      tag: "2.0.1"
      pullPolicy: IfNotPresent
  # Repeat for table_structure, ocr, vlm_embed, and any optional keys you enable.

topology:
  otel:
    image:
      repository: <PRIVATE_REGISTRY>/opentelemetry-collector-contrib
      tag: "0.127.0"
      pullPolicy: IfNotPresent
  zipkin:
    image:
      repository: <PRIVATE_REGISTRY>/zipkin
      tag: "3.5.0"
      pullPolicy: IfNotPresent
```

The split-mode `wait-for-gateway` init container image is `busybox:1.37`.
That reference is hard-coded in `templates/deployment.yaml` and is not in
`values.yaml`. Mirror the exact tag, or run `topology.mode=standalone` so those
init containers are not rendered.

- Set `nimOperator.<key>.image.pullSecrets` to your mirror pull secret
  (for example `my-private-registry`) when it differs from
  `ngcImagePullSecret.name`. Empty per-NIM `pullSecrets` inherit the
  chart-wide name.
- Leave `serviceConfig.nimEndpoints.*` empty when operator-managed NIMs
  are in-cluster; set explicit URLs only for external or mirrored services
  outside the chart.
- For **offline captioning**, enable
  `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning` and point the pipeline
  caption endpoint at the in-cluster NIM URL (refer to
  [Image captioning](https://docs.nvidia.com/nemo/retriever/latest/extraction/prerequisites-support-matrix/#image-captioning)).

### Mirroring pattern

```bash
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"
docker pull nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1
docker tag nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1 \
  <PRIVATE_REGISTRY>/nemotron-object-detection:2.0.1
docker push <PRIVATE_REGISTRY>/nemotron-object-detection:2.0.1
```

For bulk sync, prefer [skopeo](https://github.com/containers/skopeo) or
[crane](https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md).
Record `repository@sha256:...` digests for regulated environments.

---

## Roadmap

1. **External scheduler backend** — introduce shared job, queue, lease, and
   SSE state before allowing more than one gateway replica.
2. **NetworkPolicies** restricting the service Pod to the NIM Pods
   only.
3. **Gateway autoscaling** on inflight-uploads (currently fixed
   `topology.gateway.replicas`) — shared scheduler and SSE ownership must land first.

---

## Validation

The chart is exercised in CI with `helm lint` and `helm template`. Run
locally:

```bash
helm lint nemo_retriever/helm

# Operator CRDs present: vectordb resolves vlm_embed via the operator URL.
helm template r nemo_retriever/helm \
  --api-versions apps.nvidia.com/v1alpha1 > /tmp/r-op.yaml

# Operator CRDs absent: vectordb has no operator URL to fall back to, so
# either disable vectordb or supply an explicit embed endpoint.
helm template r nemo_retriever/helm \
  --set serviceConfig.vectordb.enabled=false > /tmp/r.yaml
#   or:
# helm template r nemo_retriever/helm \
#   --set serviceConfig.nimEndpoints.embedInvokeUrl=http://embed.svc:8000/v1/embeddings \
#   > /tmp/r.yaml
```

Both renders should succeed cleanly and parse as valid Kubernetes manifests
(`kubectl apply --dry-run=client -f /tmp/r.yaml`). Refer to [VectorDB and the
embed endpoint](#vectordb-and-the-embed-endpoint) for why
`helm template r nemo_retriever/helm` without flags is rejected as a
misconfiguration.
