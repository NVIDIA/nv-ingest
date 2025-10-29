## NIM Operator Manifests and Bootstrap

This directory contains versioned NIM manifests (caches and services) and helper scripts to install the NVIDIA NIM Operator and deploy selected services into your Kubernetes cluster.

### Contents
- **nim-cache-<version>.yaml**: NIMCache resources for model artifacts (PVC-backed)
- **nim-services-all-<version>.yaml**: NIMService resources for deployable services
- **bootstrap-nim.sh**: Interactive installer with multi-select toggles and confirmation
- **install-nim-operator.sh**: Installs the NIM Operator via Helm
- **setup-nim-cache.sh**: Applies all caches for a given version
- **setup-nim-services.sh**: Applies all services for a given version
- **delete-all-nim-cache.sh**: Deletes all NIMCache resources
- **delete-all-nim-services.sh**: Deletes all NIMService resources

### Prerequisites
- **kubectl** and **helm** installed and configured for your cluster
- **NVIDIA GPU nodes** with appropriate drivers; many services target MIG profiles
  - Manifests may include `nodeSelector: nvidia.com/mig.config: all-1g.23gb` and tolerations
- **StorageClass** available and sized appropriately for caches
  - Example in caches: `storageClass: pdx02-cdot03-ainfs01-nfsv3` (adjust for your cluster)
- **NGC credentials** and required Kubernetes secrets in the target namespace:
  - `ngc-secret`: image pull secret
  - `ngc-api-secret`: API token secret
  - You can use the helper in this repo: `scripts/private_local/kubernetes/helm/nim-operator/create-secrets.sh`

### Quick start (interactive)
Use the interactive bootstrap to choose which caches/services to install and confirm before apply.

```bash
bash /home/jdyer/Development/nv-ingest/helm/nim-operator/bootstrap-nim.sh
```

What it does:
- Prompts for:
  - Target application namespace (default `nv-ingest`)
  - NIM Operator namespace (default `nim-operator`)
  - Version (matches `nim-cache-<ver>.yaml` and `nim-services-all-<ver>.yaml`, default `1.5.0`)
- Discovers all caches/services from the versioned YAMLs
- Lets you toggle selections:
  - `t <num>` toggle item, `a` select all, `n` select none, `c` continue
- Optionally installs/verifies the NIM Operator
- Filters manifests to only your selections and applies them in the chosen namespace
- Shows resulting resources (`nimcaches`, `nimservices`)

### Non-interactive scripts
Install operator (idempotent best-effort):
```bash
bash /home/jdyer/Development/nv-ingest/helm/nim-operator/install-nim-operator.sh <OPERATOR_NAMESPACE>
# defaults: OPERATOR_NAMESPACE=nim-operator, version=v2.0.2
```

Apply all caches/services for a version:
```bash
# Namespace defaults to nv-ingest; version defaults to 1.5.0
bash /home/jdyer/Development/nv-ingest/helm/nim-operator/setup-nim-cache.sh <NAMESPACE> <VERSION>
bash /home/jdyer/Development/nv-ingest/helm/nim-operator/setup-nim-services.sh <NAMESPACE> <VERSION>
```

### Cleanup
```bash
bash /home/jdyer/Development/nv-ingest/helm/nim-operator/delete-all-nim-services.sh
bash /home/jdyer/Development/nv-ingest/helm/nim-operator/delete-all-nim-cache.sh
helm uninstall nim-operator -n <OPERATOR_NAMESPACE>
```

### Troubleshooting
- **CRDs present?**
  ```bash
  kubectl get crds | grep -i nim
  ```
- **Operator healthy?**
  ```bash
  kubectl get pods -n <OPERATOR_NAMESPACE>
  kubectl logs -n <OPERATOR_NAMESPACE> deploy/nim-operator -f
  ```
- **Resources applied?**
  ```bash
  kubectl get nimcaches -n <NAMESPACE>
  kubectl get nimservices -n <NAMESPACE>
  kubectl describe nimservice/<NAME> -n <NAMESPACE>
  ```
- **Secrets exist in target namespace?** Ensure `ngc-secret` and `ngc-api-secret` are created in the same namespace where you apply caches/services.
- **Storage issues?** PVCs created? StorageClass correct and has capacity? Adjust `storageClass` and `size` in `nim-cache-<version>.yaml`.
- **GPU scheduling/MIG?** If your cluster does not use MIG or uses a different profile, update `nodeSelector` and `tolerations` in the service manifest accordingly.

### Adding or updating versions
- Add new manifest files following the naming pattern:
  - `nim-cache-<version>.yaml`
  - `nim-services-all-<version>.yaml`
- The interactive bootstrap will prompt for the version and automatically discover items from those files.
- Keep `pullSecrets`/`authSecret` names consistent with your secret setup.

### Notes
- Manifests reference `pullSecrets: ngc-secret` and `authSecret: ngc-api-secret`; both must exist in the target namespace.
- The interactive bootstrap applies only the selected items by filtering the versioned YAMLs into a temporary manifest and applying it via `kubectl`.
- Defaults: namespace `nv-ingest`, operator namespace `nim-operator`, version `1.5.0` (override at prompt).
