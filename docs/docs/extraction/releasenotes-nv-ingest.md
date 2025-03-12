# Release Notes for NV-Ingest


## Release 25.03

### Summary

The NV-Ingest 25.03 release includes accuracy improvements, feature expansions, and throughput improvements.


## New Features

- Consolidated NeMo Retriever Extraction to run on a single GPU
- Added Library Mode for a lightweight no-GPU deployment that uses NIM endpoints hosted on build.nvidia.com
- Deprecated DePlot and CACHED NIMs
- Added support for infographics extraction
- Added support for RIVA NIM for Audio extraction (Early Access) 
- Added support for Llama-3.2 VLM for Image Captioning capability
- docX, pptx, jpg, png support for image detection & extraction
<!-- - Integrated with nemoretriever-parse NIM for state-of-the-art text extraction -->
<!-- - Integrated with new NVIDIA NIMs -->
<!--   - Nemoretriever-table-structure-v1 -->
<!--   - Nemoretriever-graphic-elements-v1 -->
<!--   - Nemoretriever-page-elements-v2 -->




## Release 24.12.1

### Bug fixes

Cases where .split() tasks fail during ingestion are now fixed.


## Release 24.12

### Known Issues

We currently do not support OCR-based text extraction. This was discovered in an unsupported use case and is not a product functionality issue.
