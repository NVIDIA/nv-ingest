# Deployment options

Use this page to compare how you run NeMo Retriever — including when to use [NVIDIA-hosted NIMs](https://build.nvidia.com/) versus self-hosting on your own infrastructure.

## Compare deployment options

Use the sections below to pick documentation and deployment options that match your goal.

### I want to run locally or embed the library

1. [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
2. [Use the Python API](nemo-retriever-api-reference.md) or [Use the CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli) — install and run the [`nemo_retriever`](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever) package in your environment

### I want a standalone Docker service container

Build and run the NeMo Retriever service image with the [Docker service image guide](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docker.md). Use this for local service-container validation; use Helm for multi-service Kubernetes deployments.

### I want a Kubernetes / Helm deployment

1. [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
2. **NeMo Retriever Helm chart (supported):** [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md). Chart sources are in [`nemo_retriever/helm`](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/helm) on GitHub. Before you install, confirm persistent-volume binding and four allocatable GPU slots across eligible nodes, or GPU sharing, for the four default NIMServices. Refer to [Kubernetes Helm Storage Requirements](prerequisites-support-matrix.md#kubernetes-helm-storage-requirements) and [Kubernetes Helm GPU scheduling](prerequisites-support-matrix.md#kubernetes-helm-gpu-scheduling). When you change a NIM image repository or tag on an existing release, delete the `NIMCache` before you upgrade. Refer to [Changing a NIM image repository or tag](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#changing-nim-image-repository-or-tag).
3. **Published Library Helm charts (supported):** cluster install and upgrade procedures are covered in [About getting started](getting-started-about.md) — use alongside the NeMo Retriever chart README for your release
4. [Environment variables](environment-config.md) and [Troubleshoot](troubleshoot.md) as needed

The Helm chart uses `GET /v1/live` for startup and liveness probes and
`GET /v1/health` for readiness. Both endpoints are unauthenticated. In split
topology, `/v1/health` returns HTTP `503` when the required realtime or batch
worker is unavailable, so Kubernetes removes the gateway from Service endpoints.
In split topology, the realtime and batch init containers reach `/v1/live`
through the chart's internal gateway startup Service
(`<release>-nemo-retriever-gateway-startup`) so a clean install is not blocked
by that readiness gate. Refer to the Helm chart [health probe guidance](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#health-probes).
In split topology, the gateway authenticates public requests. When internal
service authentication is configured, the gateway uses it for restricted worker
handoffs instead of forwarding public credentials. Without internal service
authentication, the gateway forwards the configured public authentication header
to workers, which must share the same public-authentication configuration.

If you use `ServiceIngestor.vdb_upload()` sidecar metadata in split topology,
upload and reference the metadata while the singleton gateway remains running.
At ingest admission, the gateway binds the metadata to the work item. Workers
receive the bound attachment and do not resolve sidecar IDs.
Refer to [Sidecar metadata in split topology](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#sidecar-metadata-in-split-topology).
**Core NIMs for the default extraction pipeline:** `page_elements`, `table_structure`, `ocr`, and `vlm_embed` (`nemotron-3-embed-1b:2.2.2`). These four are enabled and auto-wired into the retriever service by default. The default embedder is text-only. **Nemotron Parse**, **Nemotron 3 Nano Omni**, the **VL reranker**, **Parakeet ASR**, and the **answer-generation LLM** (`answer_llm`, Nemotron 3.5 Lightning defaults) are optional and disabled by default. When you opt in, Omni captioning and VL reranking auto-wire into `nim_endpoints`; Parakeet ASR still needs an explicit `audioGrpcEndpoint`. Enabling Nemotron Parse in Helm also sets the default PDF extract method to `nemotron_parse`. Enabling the Omni caption key does not enable `POST /v1/answer`. For `/v1/answer`, enable `nimOperator.answer_llm` or point `serviceConfig.llm` at a supported OpenAI-compatible LLM or vision-language model (VLM), including Omni. For a minimal GPU footprint, leave optional keys disabled (refer to [Recommended minimal install](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.2/nemo_retriever/helm/README.md#recommended-minimal-install-26082)). Refer to [Pre-Requisites & Support Matrix — Default NIMs](prerequisites-support-matrix.md#default-helm-nims), [Answer generation](prerequisites-support-matrix.md#answer-generation), and [Default NVCF endpoints](prerequisites-support-matrix.md#default-nvcf-endpoints).

For audio and video extraction in Kubernetes, refer to [Audio and video](audio-video.md).

### I want examples and notebooks

1. [Jupyter Notebooks](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/README.md)

### I need API details and keys

1. [Get your API key](api-keys.md)
2. [API reference — PDF pre-splitting](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest) if applicable

### I am tuning performance or cost

1. [Evaluation and performance](evaluate-on-your-data.md)
2. [Throughput is dataset-dependent](multimodal-extraction.md#extraction-limitations-and-quality)
3. [Evaluate on your data](evaluate-on-your-data.md)

## When to use NVIDIA-hosted NIMs { #when-to-use-nvidia-hosted-nims }

[NVIDIA-hosted NIMs](https://build.nvidia.com/) run inference on NVIDIA-managed infrastructure. You call models with API keys (refer to [Get your API key](api-keys.md)) without operating GPU nodes yourself.

Consider hosted NIMs when:

- You want the fastest path to try models and iterate without installing drivers, containers, or the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html) on your own clusters.
- Latency to NVIDIA endpoints works for your region and use case.
- Your compliance and data policies allow document or query content in the hosted service (confirm with your security review).

**Also refer to:** [NVIDIA NIM catalog](https://build.nvidia.com/)

## When to self-host NIMs { #when-to-self-host-nims }

Self-hosted NIMs run on your GPUs or air-gapped hardware, typically with Kubernetes and the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html).

Consider self-hosting when:

- You need an air gap, strict data residency, or customer data must not leave your network.
- You run at large scale where dedicated capacity can cost less than hosted API usage.
- You must meet latency or locality requirements that hosted regions cannot satisfy.

**GPU sharing.** Combined core NIM VRAM fits on one A10G or better GPU, but the default Helm chart still requests four exclusive GPU slots. Time-slicing creates logical slots. It does not pin the four NIM pods onto one physical GPU. Refer to [Kubernetes Helm GPU scheduling](prerequisites-support-matrix.md#kubernetes-helm-gpu-scheduling) and [GPU scheduling prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#gpu-scheduling-prerequisite).

## Air-gapped and disconnected deployment { #air-gapped-deployment }

The **default document extraction pipeline** (page elements, table structure, OCR, and text embedding) runs disconnected when you mirror images and models into a private registry and configure the [NIM Operator for air-gapped environments](https://docs.nvidia.com/nim-operator/latest/air-gap.html).

On a staging host with internet access, pull from NGC, retag to your private registry, stage chart archives, then install in the enclave with registry overrides. Procedures, the chart image inventory, and Helm value patterns are in [Helm — Air-gapped deployment](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#air-gapped-deployment).

Include the default OpenTelemetry Collector, Zipkin, and split-mode BusyBox images from that inventory. Those images are not NIM Operator custom resources. The split-mode BusyBox image cannot be overridden through `values.yaml`.

!!! warning "Audio and video extraction"

    Audio and video workflows require `ffmpeg` and `ffprobe` on `PATH`; runtime package installation is not suitable for air-gapped clusters. Refer to [Audio and video](audio-video.md) and the Helm chart [air-gapped deployment](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#air-gapped-deployment) guide. Skip this if you do not use audio or video.

For offline image captioning, deploy the in-cluster [Nemotron 3 Nano Omni](prerequisites-support-matrix.md#image-captioning) NIM and point your pipeline caption endpoint at the in-cluster HTTP URL instead of `integrate.api.nvidia.com` or other hosted APIs.

**Related**

- [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) ([`nemo_retriever/helm`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/helm) on GitHub) — [air-gapped deployment](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#air-gapped-deployment)
- [About getting started](getting-started-about.md) (prerequisites through first deployment)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Audio and video](audio-video.md)
