# Service Manager Implementation

This document describes the new service manager architecture for managing Docker Compose and Helm deployments.

## Architecture

The service management logic has been refactored using a Strategy Pattern with the following components:

### File Structure

```
tools/harness/src/nv_ingest_harness/service_manager/
├── __init__.py          # Factory function and exports
├── base.py              # Abstract ServiceManager base class
├── docker_compose.py    # Docker Compose implementation
└── helm.py              # Helm implementation
```

### Components

1. **ServiceManager (base.py)**: Abstract base class defining the interface for all service managers
   - `start(no_build: bool) -> int`: Start services
   - `stop() -> int`: Stop and cleanup services
   - `check_readiness(timeout_s: int) -> bool`: Poll for service readiness
   - `get_service_url(service: str) -> str`: Get service endpoint URLs

2. **DockerComposeManager (docker_compose.py)**: Manages services using Docker Compose
   - Uses profiles from config
   - Supports `--build` flag via `no_build` parameter
   - Polls health endpoint for readiness

3. **HelmManager (helm.py)**: Manages services using Helm
   - Installs/upgrades Helm releases
   - Supports both remote charts (from Helm repos) and local charts
   - Supports custom values files and inline values
   - Automatically sets up kubectl port-forward for local access
   - Waits for deployment readiness
   - Cleans up port-forward and releases on stop
   - Note: Docker Compose "profiles" are not used in Helm deployments

4. **create_service_manager()**: Factory function that creates the appropriate manager based on config

## Configuration

Add the following to `test_configs.yaml`:

```yaml
active:
  # ... existing config ...
  
  # Deployment configuration
  deployment_type: docker-compose  # Options: docker-compose, helm
  
  # Docker Compose specific (ignored when deployment_type: helm)
  profiles:
    - retrieval
    - table-structure
  
  # Helm specific (only used when deployment_type: helm)
  helm_bin: helm  # Helm binary command (e.g., "helm", "microk8s helm", "k3s helm")
  helm_chart: nim-nvstaging/nv-ingest  # Remote chart (set to null for local ./helm chart)
  helm_chart_version: 26.1.0-RC7  # Chart version (required for remote charts)
  helm_release: nv-ingest
  helm_namespace: nv-ingest
  helm_values_file: .helm-overrides.yaml  # Optional: path to values file
  service_port: 7670  # Port for accessing services
  # helm_values:  # Optional: inline Helm values
  #   api.enabled: true
  #   redis.enabled: true
```

## Usage

### Docker Compose (Default)

```bash
# Use Docker Compose (default behavior, unchanged)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed
```

### Helm with Remote Chart

1. Update `test_configs.yaml` to use Helm with a remote chart:
   ```yaml
   active:
     deployment_type: helm
     helm_bin: helm  # Use "microk8s helm" for MicroK8s, "k3s helm" for K3s
     helm_chart: nim-nvstaging/nv-ingest
     helm_chart_version: 26.1.0-RC7
     helm_release: nv-ingest
     helm_namespace: nv-ingest
     helm_values_file: .helm-overrides.yaml
   ```

2. Run tests:
   ```bash
   uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed
   ```

### Helm with Local Chart

1. Update `test_configs.yaml` to use Helm with the local chart:
   ```yaml
   active:
     deployment_type: helm
     helm_bin: helm  # Use "microk8s helm" for MicroK8s, "k3s helm" for K3s
     helm_chart: null  # Use local ./helm chart
     helm_release: nv-ingest
     helm_namespace: nv-ingest
     helm_values_file: .helm-overrides.yaml
   ```

2. Run tests:
   ```bash
   uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed
   ```

## Benefits

- **Clean separation**: Each orchestrator has its own implementation
- **Extensible**: Easy to add new deployment types (Kubernetes manifests, Nomad, etc.)
- **Testable**: Each manager can be tested in isolation
- **Configuration-driven**: Switch between Docker Compose and Helm via config
- **Backward compatible**: Existing Docker Compose workflows continue to work

## Implementation Details

### run.py Changes

The `run_datasets` function was refactored to:
1. Create a service manager using the factory function
2. Use the manager's `start()` method instead of direct Docker Compose commands
3. Use the manager's `check_readiness()` method instead of the standalone function
4. Use the manager's `stop()` method for cleanup

### Removed Code

- `stop_services()` function (replaced by `service_manager.stop()`)
- Direct Docker Compose command construction in `run_datasets()`

### Preserved Code

- `readiness_wait()` function (kept for backward compatibility, though not used internally)
- `COMPOSE_FILE` variable (kept for backward compatibility)

## Helm Binary Configuration

The `helm_bin` configuration option supports different Helm installations:

### Standard Helm
```yaml
helm_bin: helm
helm_sudo: false
```

### MicroK8s (with sudo)
If you need root privileges for MicroK8s:
```yaml
helm_bin: microk8s helm
helm_sudo: true  # Runs: sudo microk8s helm ...
```

**Note**: Using `helm_sudo: true` allows the main Python process to run as your user (with access to your venv), while only the Helm commands run with sudo. This avoids the need to run the entire harness as root.

**Alternative**: Add your user to the microk8s group to avoid needing sudo:
```bash
sudo usermod -a -G microk8s $USER
newgrp microk8s
```
Then set `helm_sudo: false`.

### K3s
```yaml
helm_bin: k3s helm
helm_sudo: true  # If needed
```

### Custom Path
```yaml
helm_bin: /usr/local/bin/helm
helm_sudo: false
```

### Environment Variables
You can also set these via environment variables:
```bash
export HELM_BIN="microk8s helm"
export HELM_SUDO="true"
uv run nv-ingest-harness-run --dataset=bo20 --case=e2e --managed
```

The command is split on spaces, so multi-word commands like `microk8s helm` work correctly.

## Docker Compose Profiles vs Helm Values

**Important**: Docker Compose "profiles" are specific to Docker Compose and are **not** used in Helm deployments.

### Docker Compose
Uses `profiles` to enable/disable service groups:
```yaml
deployment_type: docker-compose
profiles:
  - retrieval
  - table-structure
```

### Helm
Use `helm_values` or `helm_values_file` to configure which services to enable:
```yaml
deployment_type: helm
helm_values_file: .helm-overrides.yaml
# Or use inline values:
# helm_values:
#   api.enabled: true
#   redis.enabled: true
#   yolox.enabled: true
```

For Helm deployments, configure services through your values file or inline `helm_values` according to your Helm chart's schema.

## Port Forwarding for Helm Deployments

The Helm manager automatically sets up `kubectl port-forward` to make services accessible on localhost. This happens after the Helm release is installed and runs in the background.

### Configuration

```yaml
kubectl_bin: kubectl  # or "microk8s kubectl", "k3s kubectl"
kubectl_sudo: null  # Defaults to same as helm_sudo if not set
service_port: 7670  # Port to forward (local:remote)
```

### What Happens

1. **After Helm install**: Automatically runs port-forward in background
   ```bash
   sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670
   ```

2. **Retry logic**: If pods are not ready yet (Pending status), retries every 5 seconds for up to 120 seconds

3. **During tests**: Services are accessible at `http://localhost:7670`

4. **On cleanup**: Port-forward process is automatically terminated

### Retry Behavior

The port-forward will automatically retry if:
- Pods are in `Pending` state
- Error message contains "pod is not running"

Example output:
```
Waiting for pod to be ready (timeout: 120s)...
  Attempt 1: Pod not ready yet (elapsed: 7s)
  Attempt 2: Pod not ready yet (elapsed: 12s)
  Attempt 3: Pod not ready yet (elapsed: 17s)
Port forwarding started (PID: 12345)
```

### Troubleshooting

If port forwarding fails after timeout, you can manually set it up:
```bash
sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670
```

The harness will continue and attempt to connect to the port.

## Future Enhancements

Potential improvements:
1. Add support for Kubernetes manifest files
2. Implement kubectl-based readiness checks for Helm
3. Add support for port-forwarding in Helm manager
4. Add retry logic for transient failures
5. Support for multiple service endpoints
