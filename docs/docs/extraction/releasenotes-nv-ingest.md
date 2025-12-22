# Release Notes for NeMo Retriever Extraction

This documentation contains the release notes for [NeMo Retriever extraction](overview.md).

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.



## Release 26.01 (26.1.0)

The NeMo Retriever extraction 26.01 release adds new hardware and software support, and other improvements.

To upgrade the Helm Charts for this version, refer to [NV-Ingest Helm Charts](https://github.com/NVIDIA/nv-ingest/blob/release/26.1.0/helm/README.md).


### Highlights 

This release contains the following key changes:

- Add functional support for [H200 NVL](https://www.nvidia.com/en-us/data-center/h200/). For details, refer to [Support Matrix](support-matrix.md).
- Add support for the [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse/modelcard) model which replaces the [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse/modelcard) model. For details, refer to [Advanced Visual Parsing](advanced-visual-parsing.md).
- New documentation [Why Throughput Is Dataset-Dependent](throughput-is-dataset-dependent.md).
- New documentation [Add User-defined Stages](user-defined-stages.md).
- New documentation [Use the API (V2)](v2-api-guide.md).



### Fixed Known Issues

The following are the known issues that are fixed in this version:

- A10G support is restored. To use A10G hardware, use release 26.1.0 or later. For details, refer to [Support Matrix](support-matrix.md).
- L40S support is restored. To use L40S hardware, use release 26.1.0 or later. For details, refer to [Support Matrix](support-matrix.md).
- The page number field in the content metadata now starts at 1 instead of 0 so each page number is no longer off by one from what you would expect. For details, refer to [Content Metadata](content-metadata.md).




## All Known Issues

The following are the known issues for NeMo Retriever extraction:

- Advanced visual parsing is not supported on RTX Pro 6000, B200, or H200 NVL. For details, refer to [Advanced Visual Parsing](advanced-visual-parsing.md) and [Support Matrix](support-matrix.md).
- The NeMo Retriever extraction pipeline does not support ingestion of batches that include individual files greater than approximately 400MB.



## Release Notes for Previous Versions

| [25.9.0](https://docs.nvidia.com/nemo/retriever/25.9.0/extraction/releasenotes-nv-ingest/) 
| [25.6.3](https://docs.nvidia.com/nemo/retriever/25.6.3/extraction/releasenotes-nv-ingest/) 
| [25.6.2](https://docs.nvidia.com/nemo/retriever/25.6.2/extraction/releasenotes-nv-ingest/) 
| [25.4.2](https://docs.nvidia.com/nemo/retriever/25.4.2/extraction/releasenotes-nv-ingest/) 
| [25.3.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/) 
| [24.12.1](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/) 
| [24.12.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/) 
|



## Related Topics

<!-- REQUIRED: Fill in correct page titles and file names in the links -->
<!-- Generally, do not include pages in the Related Topics section     -->
<!--            that are already linked to earlier in the page.        -->
<!-- Here are some examples that you can leverage to get started       -->
<!-- START OF EXAMPLES -->
- [deployment-topic](link.md)
- [troubleshooting-topic](link.md)
- [related-topic-1](link.md)
- [related-topic-2](link.md)
<!-- END OF EXAMPLES -->
