# Prerequisites for NeMo Retriever Extraction

Before you begin using [NeMo Retriever extraction](overview.md), ensure the following software and hardware prerequisites are met.

For product naming, see [What is NeMo Retriever Extraction?](overview.md).



## Software Requirements

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Docker Buildx](https://docs.docker.com/build/concepts/overview/#buildx) `>= 0.17` (Compose 2.40+ enforces this)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA Driver >= `535`, CUDA >= `12.2`)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Conda Python environment and package manager](https://github.com/conda-forge/miniforge)


!!! note

    You install Python later.



## Hardware Requirements

The full ingestion pipeline is designed to consume significant CPU and memory resources to achieve maximal parallelism. 
Resource usage scales up to the limits of your deployed system.

For additional hardware details, refer to [Support Matrix](support-matrix.md).


### Recommended Production Deployment Specifications

- **System Memory**: At least 256 GB RAM
- **CPU Cores**: At least 32 CPU cores
- **GPU**: NVIDIA GPU with at least 24 GB VRAM (e.g., A100, V100, or equivalent)

!!! note

    Using less powerful systems or lower resource limits is still viable, but performance will suffer.


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

For guidance on choosing between static and dynamic scaling modes, and how to configure them in `docker-compose.yaml`, see [Scaling Modes](scaling-modes.md).



### Environment Requirements

Ensure your deployment environment meets these specifications before running the full pipeline. Resource-constrained environments may experience performance degradation.



## Related Topics

- [Support Matrix](support-matrix.md)
- [Troubleshooting](troubleshoot.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy With Helm](helm.md)
