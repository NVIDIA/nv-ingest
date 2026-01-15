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
  
  # Docker Compose configuration
  compose:
    profiles:
      - retrieval
      - table-structure
  
  # Helm configuration
  helm:
    bin: helm  # Helm binary command (e.g., "helm", "microk8s helm", "k3s helm")
    chart: nim-nvstaging/nv-ingest  # Remote chart (set to null for local ./helm chart)
    chart_version: 26.1.0-RC7  # Chart version (required for remote charts)
    release: nv-ingest
    namespace: nv-ingest
    values_file: .helm-overrides.yaml  # Optional: path to values file
    # values:  # Optional: inline Helm values
    #   api.enabled: true
    #   redis.enabled: true
```

## Usage

### Docker Compose (Default)

```bash
# Use Docker Compose (default behavior)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed

# Or explicitly specify:
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --deployment-type=compose
```

### Helm with Remote Chart

1. Update `test_configs.yaml` to configure Helm settings:
   ```yaml
   active:
     helm:
       bin: helm  # Use "microk8s helm" for MicroK8s, "k3s helm" for K3s
       chart: nim-nvstaging/nv-ingest
       chart_version: 26.1.0-RC7
       release: nv-ingest
       namespace: nv-ingest
       values_file: .helm-overrides.yaml
   ```

2. Run tests with `--deployment-type=helm`:
   ```bash
   uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --deployment-type=helm
   ```

### Helm with Local Chart

1. Update `test_configs.yaml` to configure Helm to use the local chart:
   ```yaml
   active:
     deployment_type: helm
     helm:
       bin: helm  # Use "microk8s helm" for MicroK8s, "k3s helm" for K3s
       chart: null  # Use local ./helm chart
       release: nv-ingest
       namespace: nv-ingest
       values_file: .helm-overrides.yaml
   ```

2. Run tests with `--deployment-type=helm`:
   ```bash
   uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --deployment-type=helm
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
helm:
  bin: microk8s helm
  sudo: true  # Runs: sudo microk8s helm ...
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
helm:
  bin: k3s helm
  sudo: true  # If needed
```

### Custom Path
```yaml
helm:
  bin: /usr/local/bin/helm
  sudo: false
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
compose:
  profiles:
    - retrieval
    - table-structure
```

### Helm
Use `helm.values_file` or `helm.values` to configure which services to enable:
```yaml
helm:
  values_file: .helm-overrides.yaml
  # Or use inline values:
  # values:
  #   api.enabled: true
  #   redis.enabled: true
  #   yolox.enabled: true
```

For Helm deployments, configure services through your values file or inline `helm_values` according to your Helm chart's schema.

## Port Forwarding for Helm Deployments

The Helm manager automatically sets up `kubectl port-forward` for multiple services to make them accessible on localhost. This happens after the Helm release is installed and runs in the background.

### Configuration

```yaml
helm:
  kubectl_bin: kubectl  # or "microk8s kubectl", "k3s kubectl"
  kubectl_sudo: null  # Defaults to same as helm_sudo if not set

  # Multiple port forwards (supports wildcards)
  port_forwards:
    - service: nv-ingest
      local_port: 7670
      remote_port: 7670
    - service: nv-ingest-milvus
      local_port: 19530
      remote_port: 19530
    - service: "*embed*"  # Wildcard pattern
      local_port: 8012
      remote_port: 8000
```

### Wildcard Service Matching

You can use wildcards (`*`) in service names to match dynamic service names:

- `"*embed*"` - Matches any service with "embed" in the name (e.g., `nv-ingest-embed-nim`, `embed-service`)
- `"nv-ingest-*"` - Matches services starting with "nv-ingest-"
- `"*-milvus"` - Matches services ending with "-milvus"

The manager queries `kubectl get services` to find matches and starts port-forward for each.

### What Happens

1. **After Helm install**: Automatically starts port-forward for all configured services
   ```bash
   sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670 (background)
   sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest-milvus 19530:19530 (background)
   sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest-embed-nim 8012:8000 (background)
   ```

2. **Retry logic**: If pods are not ready yet (Pending status), retries every 5 seconds for up to 120 seconds per service

3. **During tests**: Services are accessible at their configured local ports
   - Main API: `http://localhost:7670`
   - Milvus: `localhost:19530`
   - Embedding: `http://localhost:8012`

4. **On cleanup**: All port-forward processes are automatically terminated (always, regardless of `--keep-up`)

### Retry Behavior

Each port-forward will automatically retry if:
- Pods are in `Pending` state
- Error message contains "pod is not running"

Example output:
```
$ sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670 (background)
Waiting for nv-ingest pod to be ready (timeout: 120s)...
  Attempt 1: Pod not ready yet (elapsed: 7s)
  Attempt 2: Pod not ready yet (elapsed: 12s)
Port forwarding started for nv-ingest (7670:7670) (PID: 12345)

$ sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest-milvus 19530:19530 (background)
Waiting for nv-ingest-milvus pod to be ready (timeout: 120s)...
Port forwarding started for nv-ingest-milvus (19530:19530) (PID: 12346)

$ sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest-embed-nim 8012:8000 (background)
Waiting for nv-ingest-embed-nim pod to be ready (timeout: 120s)...
Port forwarding started for nv-ingest-embed-nim (8012:8000) (PID: 12347)
```

### Default Configuration

If `helm.port_forwards` is not specified, the manager defaults to forwarding only the main service on port 7670:
```yaml
helm:
  port_forwards:
    - service: nv-ingest  # Uses helm.release value
      local_port: 7670
      remote_port: 7670
```

### Troubleshooting

If port forwarding fails after timeout, you can manually set it up:
```bash
sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670 &
sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest-milvus 19530:19530 &
sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest-embed-nim 8012:8000 &
```

The harness will continue and attempt to connect to the ports.

## Port Forwards and `--keep-up`

**Important**: Port-forward processes are **always** cleaned up at the end, even when using `--keep-up`:

- `--keep-up` keeps the **Helm release** installed
- Port-forwards are **always terminated** to prevent orphaned processes
- Helpful commands are printed to manually recreate port-forwards if needed

### Example with `--keep-up`:

```bash
uv run nv-ingest-harness-run --case=e2e --dataset=bo20 --managed --keep-up
```

Output at the end:
```
Stopping 3 port forward(s)...
  Stopping nv-ingest (7670:7670) (PID: 12345)...
  Stopping nv-ingest-milvus (19530:19530) (PID: 12346)...
  Stopping nv-ingest-embed-nim (8012:8000) (PID: 12347)...

============================================================
Services are kept running (--keep-up enabled)
Port forwards have been cleaned up to prevent orphaned processes.

To manually recreate port forwards, run:
============================================================
  sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670 &
  sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest-milvus 19530:19530 &
  # For pattern: *embed*
  sudo microk8s kubectl port-forward -n nv-ingest service/nv-ingest-embed-nim 8012:8000 &
============================================================
```

### Why Clean Up Port Forwards?

1. **Prevents port conflicts** - Next run can bind to the same ports
2. **No orphaned processes** - Avoids accumulating background processes
3. **Clean state** - Each run starts fresh with its own port-forwards
4. **Easy to recreate** - Commands are provided if manual access is needed

## Future Enhancements

Potential improvements:
1. Add support for Kubernetes manifest files
2. Implement kubectl-based readiness checks for Helm
3. Add support for port-forwarding in Helm manager
4. Add retry logic for transient failures
5. Support for multiple service endpoints
