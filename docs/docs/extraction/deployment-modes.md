The NV-Ingest pipeline supports three distinct deployment modes:

### 1. **Compose (Containerized) Mode**

Compose mode utilizes a `docker-compose.yaml` file, orchestrating multiple Docker containers, each configured via environment variables. Key features include:

- **Isolation:** Each service (e.g., audio, paddle OCR, YOLOX) runs in its dedicated container.
- **Automatic Configuration**: Environment variables and endpoints are auto-configured, simplifying local and cloud deployments.
- **Resource Management:** GPU and system resources are explicitly allocated and managed by Docker.
- **Endpoints:** Uses internally hosted endpoints (e.g., `http://paddle:8000/v1/infer`) to interact with local microservices.
- **Example Configuration Snippet:**
  ```yaml
  environment:
    - AUDIO_GRPC_ENDPOINT=audio:50051
    - CUDA_VISIBLE_DEVICES=-1
    - YOLOX_HTTP_ENDPOINT=http://page-elements:8000/v1/infer
  ```

### 2. **Library (Single-Process) Mode**

The PipelineCreationSchema class is designed for scenarios where the entire NV-Ingest pipeline runs within a single Python process, connecting primarily to external or cloud-hosted services:

- **Unified Execution:** All pipeline components execute as a cohesive Python module, suitable for embedding within larger applications.
- **External API Integration:** Defaults to cloud-hosted NVIDIA endpoints (e.g., `https://ai.api.nvidia.com/v1`), reducing the need for local containers.
- **Configurability:** Environment variables enable configuration flexibility without container management complexity.
- **Protocol Selection:** HTTP is preferred over GRPC due to simplicity in external communications.
- **Examples:** Refer to `examples/launch_libmode_and_run_ingestor.py` and `examples/launch_libmode_service.py` for practical implementation examples.
- **CLI Integration:** You can use the `nv-ingest-cli` tool with a libmode deployment by adding `--client_type=simple --client_port=7671 --client_host=localhost`.

**Note:** Library mode is intended exclusively for prototyping, development, and local testing scenarios and is not recommended for production use.

### 3. **Helm (Kubernetes) Mode**

For full-scale production deployments, the NV-Ingest pipeline supports deployment via Helm charts in Kubernetes. Helm provides robust management of large-scale deployments, ensuring high availability, scalability, and streamlined upgrades.

- **Production-Grade:** Designed explicitly for large-scale and mission-critical production environments.
- **Scalable and Manageable:** Leverages Kubernetes orchestration capabilities for efficient resource allocation and reliability.

**Documentation:** Learn more at `helm/README.md`.

### Use-Case Comparison:
- **Compose Mode** is ideal for:
    - Full-stack development and moderate production environments.
    - Local testing where all microservices are readily available as containers.
    - Situations demanding precise control over local resources and infrastructure.

- **Library Mode** is suited for:
    - Rapid prototyping, lightweight development, and local testing.
    - Development scenarios where ease-of-setup and minimal overhead are priorities.
    - Leveraging cloud or external APIs to reduce local resource usage and complexity.

- **Helm Mode** is recommended for:
    - Full-scale, high-availability production deployments.
    - Environments requiring scalability, reliability, and robust deployment management.

These deployment modes provide flexibility, enabling optimized deployment tailored to the operational context, development needs, and infrastructure capabilities.
