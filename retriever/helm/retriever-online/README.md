# retriever-online Helm Chart

Deploys the **Retriever online ingest** service (Ray Serve REST API) on Kubernetes. The service is exposed on **port 7670** by default and accepts document ingestion requests at `POST /ingest`.

## Prerequisites

- Kubernetes cluster
- Helm 3
- Docker image built from `retriever/Dockerfile` (see [retriever/DOCKER.md](../DOCKER.md))

## Install

Build and push the image (from repo root):

```bash
docker build -f retriever/Dockerfile -t <your-registry>/retriever-online:latest .
docker push <your-registry>/retriever-online:latest
```

Install the chart:

```bash
# From the nv-ingest repo root
helm install retriever-online retriever/helm/retriever-online \
  --namespace retriever \
  --create-namespace \
  --set image.repository=<your-registry>/retriever-online \
  --set image.tag=latest
```

With persistence for LanceDB and optional model volume:

```bash
helm install retriever-online retriever/helm/retriever-online \
  --namespace retriever \
  --create-namespace \
  --set image.repository=<your-registry>/retriever-online \
  --set image.tag=latest \
  --set persistence.enabled=true \
  --set persistence.size=10Gi \
  --set modelVolume.enabled=true \
  --set modelVolume.existingClaim=nemotron-ocr-model-pvc
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `service.port` | Service port (cluster-internal and NodePort/LoadBalancer if used) | `7670` |
| `image.repository` | Image repository | `retriever-online` |
| `image.tag` | Image tag | `latest` |
| `replicaCount` | Number of replicas | `1` |
| `persistence.enabled` | Use a PVC for LanceDB data | `false` |
| `persistence.size` | Size of PVC when persistence enabled | `10Gi` |
| `modelVolume.enabled` | Mount a PVC for Nemotron OCR model | `false` |
| `modelVolume.existingClaim` | Name of existing PVC for model (required if enabled) | - |
| `env.ONLINE_LANCEDB_URI` | LanceDB directory in container | `/data/lancedb` |
| `env.NEMOTRON_OCR_MODEL_DIR` | Nemotron OCR model path | `/workspace/models/nemotron-ocr-v1` |
| `env.ONLINE_EMBED_ENDPOINT` | Optional remote embedding NIM URL | unset |

The application listens on **port 7670** inside the container; the Service exposes the same port.

## Access

- **ClusterIP (default)**: Other pods can reach `http://<release-name>-retriever-online.<namespace>.svc.cluster.local:7670`.
- **Port-forward**: `kubectl port-forward svc/<release-name>-retriever-online 7670:7670 -n <namespace>` then use `http://localhost:7670`.
- **LoadBalancer**: Set `service.type=LoadBalancer` to get an external IP and access on port 7670.

## Upgrade / Uninstall

```bash
helm upgrade retriever-online retriever/helm/retriever-online -n retriever --set image.tag=v0.2.0
helm uninstall retriever-online -n retriever
```
