# Service Manager Implementation

This document describes the service manager architecture for managing Docker Compose and Helm deployments in the nv-ingest test harness.

## Status: ✅ Implemented

The service manager has been fully implemented and is actively used for managing both Docker Compose and Helm-based deployments.

## Architecture

The service management logic uses a Strategy Pattern with the following components:

### File Structure

```
tools/harness/src/nv_ingest_harness/service_manager/
├── __init__.py          # Factory function and exports
├── base.py              # Abstract ServiceManager base class
├── docker_compose.py    # Docker Compose implementation
└── helm.py              # Helm implementation (with port-forward support)
```

### Components

1. **ServiceManager (base.py)**: Abstract base class defining the interface for all service managers
   - `start(no_build: bool) -> int`: Start services
   - `stop(clean: bool) -> int`: Stop and cleanup services (clean=True removes volumes/namespace)
   - `restart(build: bool, clean: bool, timeout: int) -> int`: Restart services (stop, start, wait for readiness)
   - `check_readiness(timeout_s: int, check_milvus: bool) -> bool`: Poll for service readiness
   - `get_service_url(service: str) -> str`: Get service endpoint URLs
   - `dump_logs(artifacts_dir: Path) -> int`: Dump service logs to artifacts directory (abstract method)

2. **DockerComposeManager (docker_compose.py)**: Manages services using Docker Compose
   - Uses profiles from config to enable/disable service groups
   - Supports `--no-build` flag to skip building Docker images
   - Polls health endpoint for main service and optionally Milvus
   - Cleans up containers and volumes on stop
   - **SKU Override Support**: Accepts `sku` parameter to load GPU-specific override files
   - **Log Dumping**: Implements `dump_logs()` to capture container logs
     - Dumps individual container logs: `container_<name>.log`
     - Dumps combined logs: `docker_compose_combined.log`
     - Includes timestamps and handles errors gracefully

3. **HelmManager (helm.py)**: Manages services using Helm
   - Installs/upgrades Helm releases with version support
   - Supports both remote charts (from Helm repos) and local `./helm` chart
   - Supports custom values files and inline values
   - **SKU override**: Optional `helm/overrides/values-<sku>.yaml` via `-f` when `--sku` is set
   - **Port-forwarding**: Automatically sets up resilient `kubectl port-forward` for services
   - **Wildcard matching**: Supports dynamic service name patterns (e.g., `*embed*`)
   - **Auto-restart**: Port-forwards automatically restart on pod restarts/failures
   - **Cleanup**: Port-forwards always cleaned up (even with `--keep-up`)
   - Polls health endpoint for readiness
   - **Log Dumping**: Implements `dump_logs()` to capture pod logs
     - Dumps logs for each container in each pod: `pod_<name>_<container>.log`
     - Captures previous logs if container restarted: `pod_<name>_<container>_previous.log`
     - Dumps pod status: `pod_status.txt`
     - Dumps pod events: `pod_events.txt`
     - Includes timestamps and handles multi-container pods

4. **create_service_manager()**: Factory function that creates the appropriate manager based on `deployment_type` config
   - Accepts `sku` parameter; passes it to `DockerComposeManager` or `HelmManager` for GPU-specific overrides

## SKU Override Support

The service manager supports GPU-specific configuration overrides via the `--sku` CLI option for both Docker Compose and Helm deployments.

### Implementation Details

#### CLI Layer (`cli/run.py`, `cli/nightly.py`)
- Both CLIs accept `--sku` parameter (e.g., `--sku=a10g`)
- Parameter is passed through to `create_service_manager()`
- Subprocess calls (in nightly.py) include `--sku` argument

#### Service Manager Factory (`service_manager/__init__.py`)
- `create_service_manager()` accepts `sku` parameter
- Passes `sku` to `DockerComposeManager` (Compose) or `HelmManager` (Helm)

#### Docker Compose Manager (`service_manager/docker_compose.py`)
- Accepts `sku` parameter in `__init__()`
- Checks for override file: `docker-compose.<sku>.yaml`
- Prints warning if specified but not found
- Uses `_build_compose_cmd()` helper to construct commands
- All docker compose commands include override file via multiple `-f` flags

#### Helm Manager (`service_manager/helm.py`)
- Accepts `sku` parameter in `__init__()`
- In `start()`, if `sku` is set and `helm/overrides/values-<sku>.yaml` exists, adds `-f <path>` to the `helm upgrade --install` command
- Prints warning if override file is specified but not found

### SKU Parameter Flow

**Compose:**
```
CLI (run.py/nightly.py)
  └─> --sku=a10g
       └─> create_service_manager(config, repo_root, sku)
            └─> DockerComposeManager(config, repo_root, sku)
                 └─> Check: docker-compose.a10g.yaml exists?
                      └─> _build_compose_cmd() adds: -f docker-compose.yaml -f docker-compose.a10g.yaml
                           └─> Used by: start(), stop(), dump_logs()
```

**Helm:**
```
CLI (run.py/nightly.py)
  └─> --sku=a10g
       └─> create_service_manager(config, repo_root, sku)
            └─> HelmManager(config, repo_root, sku)
                 └─> start(): if helm/overrides/values-a10g.yaml exists, add -f <path> to helm upgrade --install
```

### Available SKU Override Files
- **Compose:** `docker-compose.a10g.yaml`, `docker-compose.l40s.yaml`, `docker-compose.a100-40gb.yaml` (repo root)
- **Helm:** `helm/overrides/values-a10g.yaml`, `helm/overrides/values-l40s.yaml`, `helm/overrides/values-a100-40gb.yaml`

### Usage Examples

```bash
# Run with A10G GPU settings
python -m nv_ingest_harness.cli.run \
  --dataset=bo767 \
  --case=e2e \
  --managed \
  --deployment-type=compose \
  --sku=a10g

# Nightly benchmarks with L40S settings
python -m nv_ingest_harness.cli.nightly \
  --deployment-type=compose \
  --managed \
  --sku=l40s

# Helm with A10G GPU override (loads helm/overrides/values-a10g.yaml)
python -m nv_ingest_harness.cli.run \
  --dataset=bo767 --case=e2e --managed \
  --deployment-type=helm \
  --sku=a10g
```

### Override File Structure

Override files typically contain GPU-specific tuning parameters:

```yaml
# docker-compose.a10g.yaml
services:
  page-elements:
    environment:
      - NIM_TRITON_MAX_BATCH_SIZE=1
  
  graphic-elements:
    environment:
      - NIM_TRITON_MAX_BATCH_SIZE=1
```

Docker Compose automatically merges the base file with the override file, with the override taking precedence.

## Service Log Dumping

The service manager automatically captures logs from managed services to the artifacts directory when tests complete.

### Implementation Details

#### Base Service Manager (`service_manager/base.py`)
- Added abstract method `dump_logs(artifacts_dir: Path) -> int`
- Required for all service manager implementations
- Returns exit code (0 = success)

#### Docker Compose Log Dumping (`service_manager/docker_compose.py`)
- Lists all containers using `docker compose ps -q`
- Dumps individual container logs with container names: `container_<name>.log`
- Dumps combined logs from all services: `docker_compose_combined.log`
- **Dumps environment variables for each container**: `container_<name>_env.txt`
- Includes timestamps and handles errors gracefully
- Uses `docker logs` for individual containers and `docker compose logs` for combined output
- Uses `docker inspect` to capture container environment variables
- Timeouts: 60s per container, 120s for combined logs, 30s for env vars
- Continues on errors to ensure cleanup completes

#### Helm Log Dumping (`service_manager/helm.py`)
- Lists all pods in namespace using `kubectl get pods`
- Dumps logs for each container in each pod: `pod_<name>_<container>.log`
- Captures previous logs if container restarted: `pod_<name>_<container>_previous.log`
- Dumps pod status: `pod_status.txt`
- Dumps pod events: `pod_events.txt`
- **Dumps environment variables for each container**: `pod_<name>_<container>_env.txt`
- Includes timestamps in all logs
- Handles multi-container pods correctly
- Uses `kubectl exec ... env` to capture runtime environment variables
- Falls back to pod spec env vars if exec fails
- Timeouts: 60s per container/pod, 120s for status/events, 30s for env vars
- Continues on errors to ensure cleanup completes

#### CLI Integration (`cli/run.py`, `cli/nightly.py`)

**run.py:**
- Determines log directory:
  - `<session_dir>/service_logs/` if using session directory
  - `<artifact_dir>/service_logs/` as fallback
  - Timestamped directory in artifacts root if no other location available
- Calls `service_manager.dump_logs(logs_dir)` before stopping services
- Log dumping occurs before port-forward cleanup and service shutdown

**nightly.py:**
- Dumps logs to `<session_dir>/service_logs/`
- Integrated into existing service lifecycle management
- Logs captured before cleanup in main orchestration loop

### Log Directory Structure

#### Docker Compose
```
service_logs/
├── container_nv-ingest-1.log
├── container_nv-ingest-1_env.txt
├── container_redis-1.log
├── container_redis-1_env.txt
├── container_milvus-standalone-1.log
├── container_milvus-standalone-1_env.txt
├── container_embedding-1.log
├── container_embedding-1_env.txt
└── docker_compose_combined.log
```

#### Helm (Kubernetes)
```
service_logs/
├── pod_nv-ingest-ms-runtime-abc123_nv-ingest.log
├── pod_nv-ingest-ms-runtime-abc123_nv-ingest_env.txt
├── pod_nv-ingest-ms-runtime-abc123_nv-ingest_previous.log
├── pod_redis-master-0_redis.log
├── pod_redis-master-0_redis_env.txt
├── pod_milvus-standalone-xyz789_milvus.log
├── pod_milvus-standalone-xyz789_milvus_env.txt
├── pod_status.txt
└── pod_events.txt
```

### Usage Examples

Log dumping happens automatically when running in managed mode:

```bash
# Single dataset with managed services
python -m nv_ingest_harness.cli.run --case=e2e --dataset=bo767 --managed --deployment-type=compose
# Logs saved to: artifacts/bo767_compose_<timestamp>/service_logs/

# Multiple datasets with session directory
python -m nv_ingest_harness.cli.run --case=e2e --dataset=bo767,earnings --managed --session-name=test_session
# Logs saved to: artifacts/test_session/service_logs/

# Nightly benchmarks with Helm
python -m nv_ingest_harness.cli.nightly --managed --deployment-type=helm
# Logs saved to: artifacts/nightly_<timestamp>/service_logs/
```

### Benefits

1. **Debugging**: Complete service logs and environment variables preserved for post-mortem analysis
2. **Automatic**: No manual log or env var collection needed
3. **Organized**: Logs and env vars stored with test artifacts for easy correlation
4. **Comprehensive**: Captures all container/pod logs, environment variables, statuses, and events
5. **Configuration Visibility**: Environment variables help diagnose configuration-related issues
6. **Failure Analysis**: Helps diagnose infrastructure and configuration issues in test failures
7. **Non-blocking**: Continues on errors to ensure cleanup completes

### Error Handling

- Timeouts handled with subprocess timeout parameters
- Missing or failed log captures print warnings but don't fail the test
- Directory creation errors handled gracefully
- Process errors (return codes) logged but don't stop cleanup

## Configuration

The service manager can be configured in two ways:

1. **CLI flag** (highest priority): `--deployment-type=helm` or `--deployment-type=compose`
2. **YAML config** (default): Set `deployment_type` in `test_configs.yaml`

**Precedence**: CLI args > Environment variables > YAML config

### Docker Compose Configuration

```yaml
active:
  # Optional: Set deployment type default (defaults to 'compose' if not specified)
  # Can be overridden with --deployment-type CLI flag
  deployment_type: compose
  
  # Docker Compose-specific settings
  profiles:
    - retrieval
    - reranker
    # Add other profiles as needed
  
  hostname: localhost  # Default hostname for service URLs
  readiness_timeout: 600  # Seconds to wait for services to become ready
```

### Helm Configuration

```yaml
active:
  # Optional: Set deployment type default to 'helm'
  # Can be overridden with --deployment-type CLI flag
  deployment_type: helm
  
  # Helm-specific settings
  helm_bin: helm  # Helm binary (e.g., "helm", "microk8s helm", "k3s helm")
  helm_sudo: false  # Use sudo for Helm commands (set to true if needed)
  helm_chart: nim-nvstaging/nv-ingest  # Remote chart (set to null for local ./helm)
  helm_chart_version: 26.1.0-RC7  # Chart version (required for remote charts)
  helm_release: nv-ingest  # Helm release name
  helm_namespace: nv-ingest  # Kubernetes namespace
  helm_values_file: .helm-env  # Optional: path to values file
  # helm_values:  # Optional: inline Helm values (dict)
  #   api:
  #     enabled: true
  
  # kubectl configuration (for port-forwarding)
  kubectl_bin: kubectl  # kubectl binary (e.g., "kubectl", "microk8s kubectl")
  kubectl_sudo: null  # Defaults to same as helm_sudo if not set
  
  # Port forwarding configuration
  helm_port_forwards:
    - service: nv-ingest  # Service name (supports wildcards)
      local_port: 7670
      remote_port: 7670
    - service: nv-ingest-milvus
      local_port: 19530
      remote_port: 19530
    - service: "*embed*"  # Wildcard: matches any service with "embed" in name
      local_port: 8012
      remote_port: 8000
  
  hostname: localhost  # Hostname for port-forwarded services
  readiness_timeout: 600  # Seconds to wait for services to become ready
```

## Usage

### CLI Options

The service manager is controlled via CLI flags:

- `--managed`: Enable managed mode (starts/stops services automatically)
- `--deployment-type=<type>`: Set deployment type (`compose` or `helm`)
  - Overrides `deployment_type` in YAML config
  - Defaults to `compose` if not specified in either place
- `--sku=<sku>`: GPU-specific override (Compose: `docker-compose.<sku>.yaml`; Helm: `helm/overrides/values-<sku>.yaml`). e.g. `a10g`, `l40s`, `a100-40gb`
- `--no-build`: Skip building Docker images (Docker Compose only)
- `--keep-up`: Keep services running after test completes (does not apply to port-forwards)
- `--doc-analysis`: Show per-document element breakdown in results

### Docker Compose (Default)

```bash
# Default behavior uses Docker Compose (no need to specify deployment type)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed

# Explicitly specify Docker Compose (same as default)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --deployment-type=compose

# With GPU-specific settings (SKU override)
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --sku=a10g

# Skip rebuilding images
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --no-build

# Keep services running after test
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --keep-up
```

### Helm with Remote Chart

**Option 1: CLI flag (no YAML changes needed)**
```bash
# Override deployment type via CLI
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --deployment-type=helm
```

**Option 2: Set default in YAML**

1. Configure Helm as default in `test_configs.yaml`:
   ```yaml
   active:
     deployment_type: helm  # Set Helm as default
     helm_bin: helm  # Use "microk8s helm" for MicroK8s, "k3s helm" for K3s
     helm_chart: nim-nvstaging/nv-ingest
     helm_chart_version: 26.1.0-RC7
     helm_release: nv-ingest
     helm_namespace: nv-ingest
     helm_values_file: .helm-env
   ```

2. Run tests (uses Helm from YAML config):
   ```bash
   uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed
   ```

### Helm with Local Chart

Configure for local chart in `test_configs.yaml`:
```yaml
active:
  deployment_type: helm  # Optional: set as default
  helm_chart: null  # Use local ./helm chart
  helm_release: nv-ingest
  helm_namespace: nv-ingest
  helm_values_file: .helm-env
```

Run tests:
```bash
# Use YAML config default
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed

# Or explicitly override to Helm
uv run nv-ingest-harness-run --case=e2e --dataset=bo767 --managed --deployment-type=helm
```

### Multiple Datasets

Run tests across multiple datasets sequentially (services start once):

```bash
uv run nv-ingest-harness-run --case=e2e --dataset=bo767,earnings --managed
```

## Benefits

- **Clean separation**: Each orchestrator has its own implementation
- **Extensible**: Easy to add new deployment types (Kubernetes manifests, Nomad, etc.)
- **Testable**: Each manager can be tested in isolation
- **Configuration-driven**: Switch between Docker Compose and Helm via config
- **Backward compatible**: Existing Docker Compose workflows continue to work

## Key Implementation Details

### Integration in CLI (run.py)

The `run_datasets()` function uses the service manager as follows:

1. **Initialization**: Creates service manager via factory based on `deployment_type`
   ```python
   service_manager = create_service_manager(config, REPO_ROOT)
   ```

2. **Startup**: Starts services and waits for readiness
   ```python
   service_manager.start(no_build=no_build)
   service_manager.check_readiness(timeout_s)
   ```

3. **Testing**: Runs test cases while services remain up

4. **Cleanup**: Handles port-forwards and service shutdown
   ```python
   # Always cleanup port forwards (prevents orphaned processes)
   service_manager._stop_port_forwards()
   
   # Only uninstall services if not keeping up
   if not keep_up:
       service_manager.stop()
   else:
       # Print commands to manually recreate port forwards
       service_manager.print_port_forward_commands()
   ```

### Docker Compose Implementation

The `DockerComposeManager`:

- Builds command with multiple `--profile` flags
- Supports `--build` or `--no-build` flag
- Uses `docker compose down -v --remove-orphans` for cleanup
- Checks both main service (`/v1/health/ready`) and optionally Milvus health endpoints

### Helm Implementation

The `HelmManager`:

- Supports multi-word commands (`microk8s helm`, `k3s helm`)
- Optional sudo for both Helm and kubectl commands
- Handles both remote and local charts
- Merges values from files and inline values
- **Port-forwarding**:
  - Wrapped in resilient bash loop for auto-restart
  - Creates new process group for clean termination
  - Retries on pod not ready errors
  - Wildcard service matching via `kubectl get services`
  - Always cleaned up (even with `--keep-up`)
  - Provides recreation commands for manual use

## Helm Binary and sudo Configuration

The `helm_bin` and `helm_sudo` configuration options support different Helm installations:

### Standard Helm (no sudo)
```yaml
helm_bin: helm
helm_sudo: false
```

### MicroK8s with sudo
If you need root privileges for MicroK8s:
```yaml
helm_bin: microk8s helm
helm_sudo: true  # Runs: sudo microk8s helm ...
```

**Note**: Using `helm_sudo: true` allows the main Python process to run as your user (with access to your venv), while only the Helm commands run with sudo. This avoids running the entire harness as root.

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

### kubectl Configuration

The `kubectl_bin` and `kubectl_sudo` settings work the same way:

```yaml
kubectl_bin: kubectl  # or "microk8s kubectl", "k3s kubectl"
kubectl_sudo: null  # Defaults to same as helm_sudo if not set
```

### Environment Variables

You can also set these via environment variables (useful for CI/CD):
```bash
export HELM_BIN="microk8s helm"
export HELM_SUDO="true"
export KUBECTL_BIN="microk8s kubectl"
uv run nv-ingest-harness-run --dataset=bo20 --case=e2e --managed --deployment-type=helm
```

The commands are split on spaces, so multi-word commands like `microk8s helm` work correctly.

## Docker Compose Profiles vs Helm Values

**Important**: Docker Compose "profiles" are specific to Docker Compose and are **not** used in Helm deployments.

### Docker Compose

Uses `profiles` to enable/disable service groups defined in `docker-compose.yaml`:
```yaml
profiles:
  - retrieval
  - reranker
```

These map to services marked with `profiles:` in the Docker Compose file.

### Helm

Uses `helm_values_file` or `helm_values` to configure which services to enable:
```yaml
helm_values_file: .helm-env
# Or use inline values:
helm_values:
  api:
    enabled: true
  redis:
    enabled: true
  yolox:
    enabled: true
```

For Helm deployments, configure services through your values file or inline `helm_values` according to your Helm chart's schema (typically under `<service>.enabled`).

## Port Forwarding for Helm Deployments

The Helm manager automatically sets up **resilient** `kubectl port-forward` for multiple services to make them accessible on localhost. Port-forwards are wrapped in auto-restart loops to handle pod restarts and failures.

### Configuration

```yaml
helm_port_forwards:
  # Simple service forward
  - service: nv-ingest
    local_port: 7670
    remote_port: 7670
  
  # Milvus vector database
  - service: nv-ingest-milvus
    local_port: 19530
    remote_port: 19530
  
  # Wildcard pattern for dynamic service names
  - service: "*embed*"  # Matches any service with "embed" in the name
    local_port: 8012
    remote_port: 8000
```

### Default Configuration

If `helm_port_forwards` is not specified, the manager defaults to forwarding only the main service:
```yaml
helm_port_forwards:
  - service: nv-ingest  # Uses helm_release value
    local_port: 7670
    remote_port: 7670
```

### Wildcard Service Matching

Port-forwards support wildcards (`*`) in service names to match dynamic service names:

- `"*embed*"` - Matches any service with "embed" in the name (e.g., `nv-ingest-embed-nim`, `embed-service`)
- `"nv-ingest-*"` - Matches services starting with "nv-ingest-"
- `"*-milvus"` - Matches services ending with "-milvus"

The manager queries `kubectl get services` to find matches and starts port-forward for each matching service.

### Auto-Restart Behavior

Each port-forward is wrapped in a resilient bash loop:

```bash
while true; do
    echo "[$(date)] Starting port-forward for <service>..." >&2
    kubectl port-forward -n <namespace> service/<service> <local>:<remote>
    EXIT_CODE=$?
    echo "[$(date)] Port-forward exited with code $EXIT_CODE, restarting in 5s..." >&2
    sleep 5
done
```

This ensures port-forwards automatically recover from:
- Pod restarts
- Node failures
- Network interruptions
- Any other transient failures

### Lifecycle

1. **After Helm install**: Automatically starts port-forward for all configured services
   ```bash
   $ kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670 (background, auto-restart)
   Waiting for nv-ingest pod to be ready (timeout: 120s)...
   Port forwarding started for nv-ingest (7670:7670) (PID: 12345, auto-restart enabled)
   ```

2. **Retry logic**: If pods are not ready yet (Pending status), retries every 5 seconds for up to 120 seconds

3. **During tests**: Services are accessible at their configured local ports
   - Main API: `http://localhost:7670`
   - Milvus: `localhost:19530`
   - Embedding: `http://localhost:8012`

4. **On cleanup**: All port-forward processes and their children are terminated (always, regardless of `--keep-up`)

### Cleanup Behavior

Port-forwards are **always** cleaned up at the end to prevent orphaned processes:

- Uses process group termination (`killpg`) to stop bash wrapper and all kubectl children
- Fallback cleanup via `pgrep` to catch any orphaned processes in the namespace
- Graceful SIGTERM followed by force SIGKILL if needed
- Works with or without sudo

### Orphaned Process Detection

If port-forwards are not properly tracked, the manager performs a fallback cleanup:

```bash
Checking for orphaned port-forward processes in namespace 'nv-ingest'...
  Found 2 orphaned port-forward process(es), cleaning up...
    Killing PID 12345...
    Killing PID 12346...
  Orphaned process cleanup complete
```

This uses `pgrep -f "port-forward.*-n <namespace>"` to find and terminate any orphaned processes.

## Port Forwards and `--keep-up`

**Important**: Port-forward processes are **always** cleaned up at the end, even when using `--keep-up`:

- `--keep-up` keeps the **Helm release** installed (services continue running in Kubernetes)
- Port-forwards are **always terminated** to prevent orphaned processes and port conflicts
- Helpful commands are printed to manually recreate port-forwards if needed

### Why Clean Up Port Forwards?

1. **Prevents port conflicts** - Next run can bind to the same ports without conflicts
2. **No orphaned processes** - Avoids accumulating background processes across multiple runs
3. **Clean state** - Each run starts fresh with its own port-forwards
4. **Easy to recreate** - Commands are provided if manual access is needed after test

### Example with `--keep-up`

```bash
uv run nv-ingest-harness-run --case=e2e --dataset=bo20 --managed --deployment-type=helm --keep-up
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

To manually recreate port forwards (with auto-restart), run:
============================================================
  # Auto-restarting port-forward for nv-ingest:
  while true; do kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670; sleep 5; done &

  # Auto-restarting port-forward for nv-ingest-milvus:
  while true; do kubectl port-forward -n nv-ingest service/nv-ingest-milvus 19530:19530; sleep 5; done &

  # Auto-restarting port-forward for nv-ingest-embed-nim:
  while true; do kubectl port-forward -n nv-ingest service/nv-ingest-embed-nim 8012:8000; sleep 5; done &

============================================================
```

The provided commands include the auto-restart loop for resilience against pod restarts.

## Benefits

- **Clean separation**: Each orchestrator has its own implementation
- **Extensible**: Easy to add new deployment types (Kubernetes manifests, Nomad, etc.)
- **Testable**: Each manager can be tested in isolation
- **Configuration-driven**: Switch between Docker Compose and Helm via config
- **Backward compatible**: Existing Docker Compose workflows continue to work
- **Resilient port-forwarding**: Auto-restart on failures for Helm deployments
- **Clean cleanup**: Prevents orphaned processes and port conflicts
- **Flexible**: Supports various Kubernetes distributions (standard, MicroK8s, K3s)

## Advanced Features

### Multiple Port Forwards per Service

You can forward multiple ports from the same service:

```yaml
helm_port_forwards:
  - service: nv-ingest
    local_port: 7670
    remote_port: 7670
  - service: nv-ingest  # Same service, different port
    local_port: 8080
    remote_port: 8080
```

The implementation consolidates these into a single port-forward command:
```bash
kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670 8080:8080
```

### Readiness Checks

Both managers support health checking:

- **Main service**: `http://localhost:7670/v1/health/ready`
- **Milvus** (optional): `http://localhost:9091/healthz`

Configure timeout and Milvus checking:
```yaml
readiness_timeout: 600  # Seconds to wait
# Milvus check is enabled by default in check_readiness()
```

### Session Management

The service manager integrates with the harness's session management:

- Artifacts stored per dataset in structured session directories
- Results consolidated into `results.json` with metadata
- Session summary generated with all test outcomes
- Compatible with nightly runner for automated testing

## Troubleshooting

### Port-forward fails to establish

If you see timeout errors during port-forward setup:

1. **Check pod status**:
   ```bash
   kubectl get pods -n nv-ingest
   ```

2. **Check service exists**:
   ```bash
   kubectl get services -n nv-ingest
   ```

3. **Manually test port-forward**:
   ```bash
   kubectl port-forward -n nv-ingest service/nv-ingest 7670:7670
   ```

4. **Check permissions** (if using sudo):
   - Ensure your user has sudo access
   - Or add user to k8s group (microk8s, k3s)

### Services not becoming ready

If `check_readiness()` times out:

1. **Check pod logs**:
   ```bash
   kubectl logs -n nv-ingest -l app=nv-ingest
   ```

2. **Check pod events**:
   ```bash
   kubectl describe pod -n nv-ingest
   ```

3. **Increase timeout**:
   ```yaml
   readiness_timeout: 1200  # 20 minutes
   ```

### Orphaned port-forward processes

If you see port conflicts on subsequent runs:

1. **Check for orphaned processes**:
   ```bash
   ps aux | grep port-forward
   ```

2. **Kill manually**:
   ```bash
   pkill -f "port-forward.*nv-ingest"
   ```

The manager automatically attempts cleanup, but manual intervention may be needed in rare cases.

## Future Enhancements

Potential improvements for future iterations:

1. ✅ ~~Add support for Kubernetes manifest files~~ (could be added as `KubernetesManager`)
2. ✅ ~~Implement kubectl-based readiness checks for Helm~~ (currently uses HTTP endpoint, could add `kubectl wait`)
3. ✅ ~~Add support for port-forwarding in Helm manager~~ (implemented with auto-restart)
4. ✅ ~~Add retry logic for transient failures~~ (implemented for port-forwards and pod readiness)
5. ✅ ~~Support for multiple service endpoints~~ (implemented with wildcard matching)
6. Add support for NodePort/LoadBalancer service types (alternative to port-forwarding)
7. Add Prometheus/Grafana port-forwards for observability
8. Support for custom health check endpoints per service
9. Integration with kubectl context switching for multi-cluster support
10. Support for Helm chart dependencies and pre-install hooks
