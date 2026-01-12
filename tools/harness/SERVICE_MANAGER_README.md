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
   - Supports custom values files and inline values
   - Maps profiles to Helm chart values
   - Waits for deployment readiness

4. **create_service_manager()**: Factory function that creates the appropriate manager based on config

## Configuration

Add the following to `test_configs.yaml`:

```yaml
active:
  # ... existing config ...
  
  # Deployment configuration
  deployment_type: docker-compose  # Options: docker-compose, helm
  
  # Docker Compose specific
  profiles:
    - retrieval
    - table-structure
  
  # Helm specific (only used when deployment_type: helm)
  helm_release: nv-ingest
  helm_namespace: default
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

### Helm

1. Update `test_configs.yaml` to use Helm:
   ```yaml
   active:
     deployment_type: helm
     helm_release: nv-ingest
     helm_namespace: default
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

## Future Enhancements

Potential improvements:
1. Add support for Kubernetes manifest files
2. Implement kubectl-based readiness checks for Helm
3. Add support for port-forwarding in Helm manager
4. Add retry logic for transient failures
5. Support for multiple service endpoints
