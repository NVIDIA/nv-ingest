# Release Notes for NeMo Retriever Extraction

This documentation contains the release notes for [NeMo Retriever extraction](overview.md).

!!! note

    NeMo Retriever extraction is part of the NeMo Retriever Library.



## Release 26.01 (26.1.2)

The NeMo Retriever extraction 26.01 release adds new hardware and software support, and other improvements.

To upgrade the Helm Charts for this version, refer to [NeMo Retriever Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/release/26.1.2/helm/README.md).


### Highlights 

This release contains the following key changes:

- Added functional support for [H200 NVL](https://www.nvidia.com/en-us/data-center/h200/). For details, refer to [Support Matrix](support-matrix.md).
- All Helm deployments for Kubernetes now use [NVIDIA NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html). For details, refer to [NeMo Retriever Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/release/26.1.2/helm/README.md). 
- Updated RIVA NIM to version 1.4.0. For details, refer to [Extract Speech](audio.md).
- Updated VLM NIM to [nemotron-nano-12b-v2-vl](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl/modelcard). For details, refer to [Extract Captions from Images](nv-ingest-python-api.md#extract-captions-from-images).
- Added VLM caption prompt customization parameters, including reasoning control. For details, refer to [Caption Images and Control Reasoning](nv-ingest-python-api.md#caption-images-and-control-reasoning).
- Added support for the [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse/modelcard) model which replaces the [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse/modelcard) model. For details, refer to [Advanced Visual Parsing](nemoretriever-parse.md).
- Support is now deprecated for [paddleocr](https://build.nvidia.com/baidu/paddleocr/modelcard).
- The `meta-llama/Llama-3.2-1B` tokenizer is now pre-downloaded so that you can run token-based splitting without making a network request. For details, refer to [Split Documents](chunking.md).
- For scanned PDFs, added specialized extraction strategies. For details, refer to [PDF Extraction Strategies](nv-ingest-python-api.md#pdf-extraction-strategies).
- Added support for [LanceDB](https://lancedb.com/). For details, refer to [Upload to a Custom Data Store](data-store.md).
- The V2 API is now available and is the default processing pipeline. The response format remains backwards-compatible. You can enable the v2 API by using `message_client_kwargs={"api_version": "v2"}`.For details, refer to [API Reference](api-docs).
- Large PDFs are now automatically split into chunks and processed in parallel, delivering faster ingestion for long documents. For details, refer to [PDF Pre-Splitting](v2-api-guide.md).
- Issues maintaining extraction quality while processing very large files are now resolved with the V2 API. For details, refer to [V2 API Guide](v2-api-guide.md).
- Updated the embedding task to support embedding on custom content fields like the results of summarization functions. For details, refer to [Use the Python API](nv-ingest-python-api.md).
- User-defined function summarization is now using `nemotron-mini-4b-instruct` which provides significant speed improvements. For details, refer to [User-defined Functions](user-defined-functions.md) and [NeMo Retriever UDF Examples](https://github.com/NVIDIA/NeMo-Retriever/blob/release/26.1.2/examples/udfs/README.md).
- In the `Ingestor.extract` method, the defaults for `extract_text` and `extract_images` are now set to `true` for consistency with `extract_tables` and `extract_charts`. For details, refer to [Use the Python API](nv-ingest-python-api.md).
- The `table-structure` profile is no longer available. The table-structure profile is now part of the default profile. For details, refer to [Profile Information](quickstart-guide.md#profile-information).
- New documentation [Why Throughput Is Dataset-Dependent](throughput-is-dataset-dependent.md).
- New documentation [Add User-defined Stages](user-defined-stages.md).
- New documentation [Add User-defined Functions](user-defined-functions.md).
- New documentation [Resource Scaling Modes](scaling-modes.md).
- New documentation [NimClient Usage](nimclient.md).
- New documentation [Use the API (V2)](v2-api-guide.md).



### Fixed Known Issues

The following are the known issues that are fixed in this version:

- A10G support is restored. To use A10G hardware, use release 26.1.2 or later. For details, refer to [Support Matrix](support-matrix.md).
- L40S support is restored. To use L40S hardware, use release 26.1.2 or later. For details, refer to [Support Matrix](support-matrix.md).
- The page number field in the content metadata now starts at 1 instead of 0 so each page number is no longer off by one from what you would expect. For details, refer to [Content Metadata](content-metadata.md).
- Support for batches that include individual files greater than approximately 400MB is restored. This includes audio files and pdfs.



## All Known Issues

The following are the known issues for NeMo Retriever extraction:

- Advanced visual parsing is not supported on RTX Pro 6000, B200, or H200 NVL. For details, refer to [Advanced Visual Parsing](advanced-visual-parsing.md) and [Support Matrix](support-matrix.md).
- The Page Elements NIM (`nemoretriever-page-elements-v3:1.7.0`) may intermittently fail during inference under high-concurrency workloads. This happens when Triton’s dynamic batching combines requests that exceed the model’s maximum batch size, a situation more commonly seen in multi-GPU setups or large ingestion runs. In these cases, extraction fails for the impacted documents. A correction is planned for `nemoretriever-page-elements-v3:1.7.1`.


## Release Notes for Previous Versions

| [26.1.1](https://docs.nvidia.com/nemo/retriever/26.1.1/extraction/releasenotes-nv-ingest/)
| [25.9.0](https://docs.nvidia.com/nemo/retriever/25.9.0/extraction/releasenotes-nv-ingest/) 
| [25.6.3](https://docs.nvidia.com/nemo/retriever/25.6.3/extraction/releasenotes-nv-ingest/) 
| [25.6.2](https://docs.nvidia.com/nemo/retriever/25.6.2/extraction/releasenotes-nv-ingest/) 
| [25.4.2](https://docs.nvidia.com/nemo/retriever/25.4.2/extraction/releasenotes-nv-ingest/) 
| [25.3.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/) 
| [24.12.1](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/) 
| [24.12.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/) 
|



## Related Topics

- [Prerequisites](prerequisites.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy With Helm](helm.md)
