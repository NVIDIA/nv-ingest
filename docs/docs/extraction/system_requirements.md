# System Requirements

## Hardware Requirements

The NV-Ingest full ingestion pipeline is designed to consume significant CPU and memory resources to achieve maximal parallelism. Resource usage will scale up to the limits of your deployed system.

### Recommended Production Deployment Specifications

- **System Memory**: At least 256 GB RAM
- **CPU Cores**: At least 32 CPU cores
- **GPU**: NVIDIA GPU with at least 24 GB VRAM (e.g., A100, V100, or equivalent)

**Note:** Using less powerful systems or lower resource limits is still viable, but performance will suffer.

### Resource Consumption Notes

- The pipeline performs runtime allocation of parallel resources based on system configuration
- Memory usage can reach up to the full system capacity for large document processing
- CPU utilization scales with the number of concurrent processing tasks
- GPU is required for image processing NIMs, embeddings, and other GPU-accelerated tasks

### Scaling Considerations

For production deployments processing large volumes of documents, consider:
- Higher memory configurations for processing large PDF files or image collections
- Additional CPU cores for improved parallel processing
- Multiple GPUs for distributed processing workloads

### Environment Requirements

Ensure your deployment environment meets these specifications before running the full NV-Ingest pipeline. Resource-constrained environments may experience performance degradation.
