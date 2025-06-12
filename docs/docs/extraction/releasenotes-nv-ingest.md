# Release Notes for NeMo Retriever Extraction

This documentation contains the release notes for [NeMo Retriever extraction](overview.md).

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.

## Release 25.6.1

The NeMo Retriever extraction 25.06 release focuses on accuracy improvements and feature expansions, including the following:

- Improve reranker accuracy.
- Upgrade Python version from 3.10 to 3.12
- Helm deployment now has similar throughput performance to docker deployment.
- Add support for the latest version of the OpenAI API.
- Add MIG support. For details, see [Enable NVIDIA GPU MIG](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.1/helm/README.md#enable-nvidia-gpu-mig).
- Add time slicing support. For details, see [Enable GPU time-slicing](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.1/helm/README.md#enabling-gpu-time-slicing).
- Add support for RIVA NIM for optional audio extraction. For details, see [helm/values.yaml](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.1/helm/values.yaml).
- New notebook for [How to add metadata to your documents and filter searches](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.1/examples/metadata_and_filtered_search.ipynb).


### Breaking Changes

There are no breaking changes in this version.

### Upgrade

To upgrade the Helm Charts for this version, refer to [NV-Ingest Helm Charts](https://github.com/NVIDIA/nv-ingest/blob/release/25.6.1/helm/values.yaml).



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
