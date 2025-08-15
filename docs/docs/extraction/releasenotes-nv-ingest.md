# Release Notes for NeMo Retriever Extraction

This documentation contains the release notes for [NeMo Retriever extraction](overview.md).

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.



## Release 25.8.0

The NeMo Retriever extraction 25.8.0 release adds new hardware and software support, and other improvements, including the following:

- Add functional support for [RTX Pro 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/).
- Add functional support for [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/).
- Add support for [nemoretriever-ocr-v1](https://build.nvidia.com/nvidia/nemoretriever-ocr-v1). For details, refer to [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md) and [NV-Ingest Helm Charts](https://github.com/nkmcalli/nv-ingest/tree/main/helm).
- Add support for [llama-3.2-nemoretriever-1b-vlm-embed-v1](https://build.nvidia.com/nvidia/llama-3_2-nemoretriever-1b-vlm-embed-v1).
- Add support for Llama Nemotron VLM 8b NIM for image captioning. For details, refer to [Extract Captions from Images](nv-ingest-python-api.md#extract-captions-from-images).
- Add support for custom vector database implementations. For details, refer to [Build a Custom Vector Database Operator](https://github.com/NVIDIA/nv-ingest/blob/main/examples/building_vdb_operator.ipynb).
- Add support for custom Lambda stages.  For details, refer to [Add User-defined Stages to Your NeMo Retriever Extraction Pipeline](user-defined-stages.md).
- Expanded documentation for [Library Mode](quickstart-library-mode.md).
- New documentation [Configure Ray Logging](ray-logging.md).
- New documentation [Use Multimodal Embedding](vlm-embed.md).
- Add support for Integer, float, boolean, and array in custom metadata during Milvus entity creation.
- Add support for running more than one VLM at a time by using Helm.  For details, refer to [NV-Ingest Helm Charts](https://github.com/nkmcalli/nv-ingest/tree/main/helm).

### Known Issues

The following are the know issues for this release:

- A10G and L40S are not supported. For details, refer to [Support Matrix](support-matrix.md).
- `nemoretriever-parse` is not supported on RTX Pro 6000 or B200. For details, refer to [Support Matrix](support-matrix.md).


### Breaking Changes

There are no breaking changes in this version.

### Upgrade

QA link: To upgrade the Helm Charts for this version, refer to [NV-Ingest Helm Charts](https://github.com/NVIDIA/nv-ingest/blob/release/main/helm/values.yaml).
GA link: To upgrade the Helm Charts for this version, refer to [NV-Ingest Helm Charts](https://github.com/NVIDIA/nv-ingest/blob/release/TODO/helm/values.yaml).



## Release 25.6.3

The NeMo Retriever extraction 25.6.3 release is a patch release 
that updates the client's dependency on [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) to the latest stable version.

Only the release branch and the `nv-ingest-client` package have been updated in 25.6.3. 
The previously released 25.6.2 container on NGC remains unchanged.



## Release 25.6.2

The NeMo Retriever extraction 25.06 release focuses on accuracy improvements and feature expansions, including the following:

- Improve reranker accuracy.
- Upgrade Python version from 3.10 to 3.12
- Helm deployment now has similar throughput performance to docker deployment.
- Add support for the latest version of the OpenAI API.
- Add MIG support. For details, see [Enable NVIDIA GPU MIG](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.2/helm/README.md#enable-nvidia-gpu-mig).
- Add time slicing support. For details, see [Enable GPU time-slicing](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.2/helm/README.md#enabling-gpu-time-slicing).
- Add support for RIVA NIM for optional audio extraction. For details, see [helm/values.yaml](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.2/helm/values.yaml).
- New notebook for [How to add metadata to your documents and filter searches](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.2/examples/metadata_and_filtered_search.ipynb).


### Breaking Changes

There are no breaking changes in this version.

### Upgrade

To upgrade the Helm Charts for this version, refer to [NV-Ingest Helm Charts](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.2/helm/values.yaml).



## Release 25.4.2

The NeMo Retriever extraction 25.04 release focuses on small bug fixes and improvements, including the following:

- Fixed a known issue where large text file ingestion failed.
- The REST service is now more resilient, and recovers from worker failures and connection errors.
- Various improvements on the client side to reduce retry rates, and improve overall quality of life.
- New notebook for [How to reindex a collection]( https://github.com/NVIDIA/nv-ingest/blob/release/25.4.2/examples/reindex_example.ipynb).
- Expanded chunking documentation. For more information, refer to [Split Documents](chunking.md).

### Breaking Changes

There are no breaking changes in this version.

### Upgrade

To upgrade the Helm Charts for this version, refer to [NV-Ingest Helm Charts](https://github.com/NVIDIA/nv-ingest/tree/release/25.4.2/helm).



## Release 25.3.0

The NeMo Retriever extraction 25.03 release includes accuracy improvements, feature expansions, and throughput improvements.

### New Features

- Consolidated NeMo Retriever extraction to run on a single GPU (H100, A100, L40S, or A10G). For details, refer to [Support Matrix](support-matrix.md).
- Added Library Mode for a lightweight no-GPU deployment that uses NIM endpoints hosted on build.nvidia.com. For details, refer to [Deploy Without Containers (Library Mode)](quickstart-library-mode.md).
- Added support for infographics extraction.
- Added support for RIVA NIM for Audio extraction (Early Access). For details, refer to [Audio Processing](audio.md).
- Added support for Llama-3.2 VLM for Image Captioning capability.
- docX, pptx, jpg, png support for image detection & extraction.
- Deprecated DePlot and CACHED NIMs.
<!-- - Integrated with nemoretriever-parse NIM for state-of-the-art text extraction -->
<!-- - Integrated with new NVIDIA NIMs -->
<!--   - Nemoretriever-table-structure-v1 -->
<!--   - Nemoretriever-graphic-elements-v1 -->
<!--   - Nemoretriever-page-elements-v2 -->



## Release 24.12.1

Cases where .split() tasks fail during ingestion are now fixed.



## Release 24.12

We currently do not support OCR-based text extraction. This was discovered in an unsupported use case and is not a product functionality issue.
