# Python API Reference

The [NeMo Retriever Library](overview.md) Python API provides a simple and flexible interface for processing and extracting information from various document types, including PDFs.

!!! note

    This library is the NeMo Retriever Library.

!!! tip

    There is a Jupyter notebook available to help you get started with the Python API. For more information, refer to [Python Client Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/python_client_usage.ipynb).


## Summary of Key Methods

The main class in the NeMo Retriever Library Python API is `Ingestor`.
The `Ingestor` class provides an interface for building, managing, and running data ingestion jobs, enabling for chainable task additions and job state tracking.

### Ingestor Methods

The following table describes methods of the `Ingestor` class.

| Method         | Description                       |
|----------------|-----------------------------------|
| `caption`      | Extract captions from images within the document. |
| `embed`        | Generate embeddings from extracted content. |
| `extract`      | Add an extraction task (text, tables, charts, infographics). |
| `files`        | Add document paths for processing. |
| `ingest`       | Submit jobs and retrieve results synchronously. |
| `load`         | Ensure files are locally accessible (downloads if needed). |
| `save_to_disk` | Save ingestion results to disk instead of memory. |
| `store`        | Persist extracted images/structured renderings to an fsspec-compatible backend. |
| `split`        | Split documents into smaller sections for processing. For more information, refer to [Split Documents](chunking.md). |
| `vdb_upload`   | Push extraction results to Milvus vector database. For more information, refer to [Data Upload](data-store.md). |


### Extract Method Options

The following table describes the `extract_method` options.

| Value                | Status       | Description                                      |
|----------------------|--------------|--------------------------------------------------|
| `audio`              | Current      | Extract information from audio files.            |
| `nemotron_parse`     | Current      | NVIDIA Nemotron Parse extraction.                |
| `ocr`                | Current      | Bypasses native text extraction and processes every page using the full OCR pipeline. Use this for fully scanned documents or when native text is corrupt. |
| `pdfium`             | Current      | Uses PDFium to extract native text. This is the default. This is the fastest method but does not capture text from scanned images/pages. |
| `pdfium_hybrid`      | Current      | A hybrid approach that uses PDFium for pages with native text and automatically switches to OCR for scanned pages. This offers a robust balance of speed and coverage for mixed documents. |
| `adobe`              | Deprecated   | Adobe PDF Services API extraction.               |
| `haystack`           | Deprecated   | Haystack-based extraction.                       |
| `llama_parse`        | Deprecated   | LlamaParse extraction.                           |
| `tika`               | Deprecated   | Apache Tika extraction.                          |
| `unstructured_io`    | Deprecated   | Unstructured.io API extraction.                  |
| `unstructured_local` | Deprecated   | Local Unstructured extraction.                   |


### Caption images and control reasoning

The caption task can call a vision-language model (VLM) with the following optional controls:

- `prompt` (string): User prompt for captioning. Defaults to `"Caption the content of this image:"`.
- `reasoning` (boolean): Enable reasoning mode. `True` enables reasoning, `False` disables it. Defaults to `None` (service default, typically disabled).

!!! note
    The `reasoning` parameter maps to the VLM's system prompt: `reasoning=True` sets the system prompt to `"/think"`, and `reasoning=False` sets it to `"/no_think"` per the [Nemotron Nano 12B v2 VL model card] (https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl/modelcard).

Example:
```python
from nemo_retriever.client.interface import Ingestor

ingestor = (
    Ingestor()
    .files("path/to/doc-with-images.pdf")
    .extract(extract_images=True)
    .caption(
        prompt="Caption the content of this image:",
        reasoning=True,  # Enable reasoning
    )
    .ingest()
)
```



## Track Job Progress

For large document batches, you can enable a progress bar by setting `show_progress` to true.
Use the following code.

```python
# Return only successes
results = ingestor.ingest(show_progress=True)

print(len(results), "successful documents")
```



## Capture Job Failures

You can capture job failures by setting `return_failures` to true.
Use the following code.

```python
# Return both successes and failures
results, failures = ingestor.ingest(show_progress=True, return_failures=True)

print(f"{len(results)} successful docs; {len(failures)} failures")

if failures:
    print("Failures:", failures[:1])
```

When you use the `vdb_upload` method, uploads are performed after ingestion completes.
The behavior of the upload depends on the following values of `return_failures`:

- **False** – If any job fails, the `ingest` method raises a runtime error and does not upload any data (all-or-nothing data upload). This is the default setting.
- **True** – If any jobs succeed, the results from those jobs are uploaded, and no errors are raised (partial data upload). The `ingest` method returns a failures object that contains the details for any jobs that failed. You can inspect the failures object and selectively retry or remediate the failed jobs.


The following example uploads data to Milvus and returns any failures.

```python
ingestor = (
    Ingestor(client=client)
    .files(["/path/doc1.pdf", "/path/doc2.pdf"])
    .extract()
    .embed()
    .vdb_upload(collection_name="my_collection", milvus_uri="milvus.db")
)

# Use for large batches where you want successful chunks/pages to be committed, while collecting detailed diagnostics for failures.
results, failures = ingestor.ingest(return_failures=True)

print(f"Uploaded {len(results)} successful docs; {len(failures)} failures")

if failures:
    print("Failures:", failures[:1])
```



## Quick Start: Extracting PDFs

The following example demonstrates how to initialize `Ingestor`, load a PDF file, and extract its contents.
The `extract` method enables different types of data to be extracted.

### Extract a Single PDF

Use the following code to extract a single PDF file.

```python
from nemo_retriever.client.interface import Ingestor

# Initialize Ingestor with a local PDF file
ingestor = Ingestor().files("path/to/document.pdf")

# Extract text, tables, and images
result = ingestor.extract().ingest()

print(result)
```

### Extract Multiple PDFs

Use the following code to process multiple PDFs at one time.

```python
ingestor = Ingestor().files(["path/to/doc1.pdf", "path/to/doc2.pdf"])

# Extract content from all PDFs
result = ingestor.extract().ingest()

for doc in result:
    print(doc)
```

### Extract Specific Elements from PDFs

By default, the `extract` method extracts all supported content types.
You can customize the extraction behavior by using the following code.

```python
ingestor = ingestor.extract(
    extract_text=True,  # Extract text
    text_depth="page",
    extract_tables=False,  # Skip table extraction
    extract_charts=True,  # Extract charts
    extract_infographics=True,  # Extract infographic images
    extract_images=False  # Skip image extraction
)
```

### Extract Non-standard Document Types

Use the following code to extract text from `.md`, `.sh`, and `.html` files.

```python
ingestor = Ingestor().files(["path/to/doc1.md", "path/to/doc2.html"])

ingestor = ingestor.extract(
    extract_text=True,  # Only extract text
    extract_tables=False,
    extract_charts=False,
    extract_infographics=False,
    extract_images=False
)

result = ingestor.ingest()
```


### Extract with Custom Document Type

Use the following code to specify a custom document type for extraction.

```python
ingestor = ingestor.extract(document_type="pdf")
```



### Extract Office Documents (DOCX and PPTX)

The NeMo Retriever Library offers the following two extraction methods for Microsoft Office documents (.docx and .pptx), to balance performance and layout fidelity:

- Native extraction
- Render as PDF

#### Native Extraction (Default)

The default methods (`python_docx` and `python_pptx`) extract content directly from the file structure.
This is generally faster, but you might lose some visual layout information.

```python
# Uses default native extraction
ingestor = Ingestor().files(["report.docx", "presentation.pptx"]).extract()
```

#### Render as PDF

The `render_as_pdf` method uses [LibreOffice](https://www.libreoffice.org/) to convert the document to a PDF before extraction.
We recommend this approach when preserving the visual layout is critical, or when you need to extract visual elements, such as tables and charts, that are better detected by using computer vision on a rendered page.

```python
ingestor = Ingestor().files(["report.docx", "presentation.pptx"])

ingestor = ingestor.extract(
    extract_text=True,
    extract_tables=True,
    extract_charts=True,
    extract_infographics=True,
    extract_method="render_as_pdf"  # Convert to PDF first for improved visual extraction
)
```



### PDF Extraction Strategies

The NeMo Retriever Library offers specialized strategies for PDF processing to handle various document qualities.
You can select the strategy by using the following `extract_method` parameter values.
For the full list of `extract_method` options, refer to [Extract Method Options](#extract-method-options).

- **ocr** – Bypasses native text extraction and processes every page using the full OCR pipeline. Use this for fully scanned documents or when native text is corrupt.
- **pdfium** – Uses PDFium to extract native text. This is the default. This is the fastest method but does not capture text from scanned images/pages.
- **pdfium_hybrid** – A hybrid approach that uses PDFium for pages with native text and automatically switches to OCR for scanned pages. This offers a robust balance of speed and coverage for mixed documents.

```python
ingestor = Ingestor().files("mixed_content.pdf")

# Use hybrid mode for mixed digital/scanned PDFs
ingestor = ingestor.extract(
    document_type="pdf",
    extract_method="pdfium_hybrid",
)
results = ingestor.ingest()
```



## Work with Large Datasets: Save to Disk

By default, the NeMo Retriever Library stores the results from every document in system memory (RAM).
When you process a very large dataset with thousands of documents, you might encounter an Out-of-Memory (OOM) error.
The `save_to_disk` method configures the extraction pipeline to write the output for each document to a separate JSONL file on disk.


### Basic Usage: Save to a Directory

To save results to disk, simply chain the `save_to_disk` method to your ingestion task.
By using `save_to_disk` the `ingest` method returns a list of `LazyLoadedList` objects,
which are memory-efficient proxies that read from the result files on disk.

In the following example, the results are saved to a directory named `my_ingest_results`.
You are responsible for managing the created files.

```python
ingestor = Ingestor().files("large_dataset/*.pdf")

# Use save_to_disk to configure the ingestor to save results to a specific directory.
# Set cleanup=False to ensure that the directory is not deleted by any automatic process.
ingestor.save_to_disk(output_directory="./my_ingest_results", cleanup=False)  # Offload results to disk to prevent OOM errors

# 'results' is a list of LazyLoadedList objects that point to the new jsonl files.
results = ingestor.extract().ingest()

print("Ingestion results saved in ./my_ingest_results")
# You can now iterate over the results or inspect the files directly.
```

### Managing Disk Space with Automatic Cleanup

When you use `save_to_disk`, the NeMo Retriever Library creates intermediate files.
For workflows where these files are temporary, the NeMo Retriever Library provides two automatic cleanup mechanisms.

- **Directory Cleanup with Context Manager** — While not required for general use, the Ingestor can be used as a context manager (`with` statement). This enables the automatic cleanup of the entire output directory when `save_to_disk(cleanup=True)` is set (which is the default).

- **File Purge After VDB Upload** – The `vdb_upload` method includes a `purge_results_after_upload: bool = True` parameter (the default). After a successful VDB upload, this feature deletes the individual `.jsonl` files that were just uploaded.

You can also configure the output directory by using the `NV_INGEST_CLIENT_SAVE_TO_DISK_OUTPUT_DIRECTORY` environment variable.


#### Example (Fully Automatic Cleanup)

Fully Automatic cleanup is the recommended pattern for ingest-and-upload workflows where the intermediate files are no longer needed.
The entire process is temporary, and no files are left on disk.
The following example includes automatic file purge.

```python
# After the 'with' block finishes,
# the temporary directory and all its contents are automatically deleted.

with (
    Ingestor()
    .files("/path/to/large_dataset/*.pdf")
    .extract()
    .embed()
    .save_to_disk()  # cleanup=True is the default, enables directory deletion on exit
    .vdb_upload()  # purge_results_after_upload=True is the default, deletes files after upload
) as ingestor:
    results = ingestor.ingest()

```


#### Example (Preserve Results on Disk)

In scenarios where you need to inspect or use the intermediate `jsonl` files, you can disable the cleanup features.
The following example disables automatic file purge.

```python
# After the 'with' block finishes,
# the './permanent_results' directory and all jsonl files are preserved for inspection or other uses.

with (
    Ingestor()
    .files("/path/to/large_dataset/*.pdf")
    .extract()
    .embed()
    .save_to_disk(output_directory="./permanent_results", cleanup=False)  # Specify a directory and disable directory-level cleanup
    .vdb_upload(purge_results_after_upload=False)  # Disable automatic file purge after the VDB upload
) as ingestor:
    results = ingestor.ingest()
```



## Extract Captions from Images

The `caption` method generates image captions by using a VLM.
You can use this to generate descriptions of unstructured images, infographics, and other visual content extracted from documents.

!!! note

    To use the `caption` option, enable the `vlm` profile when you start the NeMo Retriever Library services. The default model used by `caption` is `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`. For more information, refer to [Profile Information in the Quickstart Guide](quickstart-guide.md#profile-information).

### Basic Usage

!!! tip

    You can configure and use other vision language models for image captioning by specifying a different `model_name` and `endpoint_url` in the `caption` method. Choose a VLM that best fits your specific use case requirements.

```python
ingestor = ingestor.caption()
```

To specify a different API endpoint, pass additional parameters to `caption`.

```python
ingestor = ingestor.caption(
    endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
    model_name="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    api_key="nvapi-"
)
```

### Captioning Infographics

Infographics are complex visual elements that combine text, charts, diagrams, and images to convey information.
VLMs are particularly effective at generating descriptive captions for infographics because they can understand and summarize the visual content.

The following example extracts and captions infographics from a document:

```python
ingestor = (
    Ingestor()
    .files("document_with_infographics.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,  # Extract infographics for captioning
        extract_images=False,
    )
    .caption(
        prompt="Describe the content and key information in this infographic:",
        reasoning=True,  # Enable reasoning for more detailed captions
    )
)
results = ingestor.ingest()
```

!!! tip

    For more information about working with infographics and multimodal content, refer to [Use Multimodal Embedding](vlm-embed.md).

### Caption Images and Control Reasoning

The caption task can call a VLM with optional prompt and system prompt overrides:

- `caption_prompt` (user prompt): defaults to `"Caption the content of this image:"`.
- `caption_system_prompt` (system prompt): defaults to `"/no_think"` (reasoning off). Set to `"/think"` to enable reasoning per the Nemotron Nano 12B v2 VL model card.

Example:
```python
from nemo_retriever.client.interface import Ingestor

ingestor = (
    Ingestor()
    .files("path/to/doc-with-images.pdf")
    .extract(extract_images=True)
    .caption(
        prompt="Caption the content of this image:",
        system_prompt="/think",  # or "/no_think"
    )
    .ingest()
)
```



## Extract Embeddings

The `embed` method in the NeMo Retriever Library generates text embeddings for document content.

```python
ingestor = ingestor.embed()
```

!!! note

    By default, `embed` uses the [llama-3.2-nv-embedqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2) model.

To use a different embedding model, such as [nv-embedqa-e5-v5](https://build.nvidia.com/nvidia/nv-embedqa-e5-v5), specify a different `model_name` and `endpoint_url`.

```python
ingestor = ingestor.embed(
    endpoint_url="https://integrate.api.nvidia.com/v1",
    model_name="nvidia/nv-embedqa-e5-v5",
    api_key="nvapi-"
)
```

## Store Extracted Images

The `store` method exports decoded images (unstructured images as well as structured renderings such as tables and charts) to any fsspec-compatible URI so you can inspect or serve the generated visuals.

```python
ingestor = ingestor.store(
    structured=True,   # persist table/chart renderings
    images=True,       # persist unstructured images
    storage_uri="file:///workspace/data/artifacts/store/images",  # Supports file://, s3://, etc.
    public_base_url="https://assets.example.com/images"  # Optional CDN/base URL for download links
)
```

### Store Method Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `structured` | bool | Persist table and chart renderings. Default: `False` |
| `images` | bool | Persist unstructured images extracted from documents. Default: `False` |
| `storage_uri` | str | fsspec-compatible URI (`file://`, `s3://`, `gs://`, etc.). Defaults to server-side `IMAGE_STORAGE_URI` environment variable. |
| `public_base_url` | str | Optional HTTP(S) base URL for serving stored images. When set, metadata includes public download links. |

### Supported Storage Backends

The `store` task uses [fsspec](https://filesystem-spec.readthedocs.io/) for storage, supporting multiple backends:

| Backend | URI Format | Example |
|---------|------------|---------|
| Local filesystem | `file://` | `file:///workspace/data/images` |
| Amazon S3 | `s3://` | `s3://my-bucket/extracted-images` |
| Google Cloud Storage | `gs://` | `gs://my-bucket/images` |
| Azure Blob Storage | `abfs://` | `abfs://container@account.dfs.core.windows.net/images` |
| MinIO (S3-compatible) | `s3://` | `s3://nemo-retriever/artifacts/store/images` (default) |

!!! tip

    `storage_uri` defaults to the server-side `IMAGE_STORAGE_URI` environment variable (commonly `s3://nemo-retriever/...`). If you change that variable—for example to a host-mounted `file://` path—restart the NeMo Retriever Library runtime so the container picks up the new value.

When `public_base_url` is provided, the metadata returned from `ingest()` surfaces that HTTP(S) link while still recording the underlying storage URI. Leave it unset when the storage endpoint itself is already publicly reachable.

### Docker Volume Mounts for Local Storage

When running the NeMo Retriever Library via Docker and using `file://` storage URIs, the path must be within a mounted volume for files to persist on the host machine.

By default, the `docker-compose.yaml` mounts a single volume:

```yaml
volumes:
  - ${DATASET_ROOT:-./data}:/workspace/data
```

This means:

| Container Path | Host Path | Works with `file://`? |
|----------------|-----------|----------------------|
| `/workspace/data/...` | `${DATASET_ROOT}/...` (default: `./data/...`) | ✅ Yes |
| `/tmp/...` | (container only) | ❌ No - files lost on restart |
| `/raid/custom/path` | (container only) | ❌ No - path not mounted |

**Example: Save to host filesystem**

```python
# Files save to ./data/artifacts/images on the host
ingestor = ingestor.store(
    structured=True,
    images=True,
    storage_uri="file:///workspace/data/artifacts/images"
)
```

**Example: Use a custom host directory**

```bash
# Set DATASET_ROOT before starting services
export DATASET_ROOT=/raid/my-project/nemo-retriever-data
docker compose up -d
```

```python
# Now /workspace/data maps to /raid/my-project/nemo-retriever-data
ingestor = ingestor.store(
    structured=True,
    images=True,
    storage_uri="file:///workspace/data/extracted-images"
)
# Files save to /raid/my-project/nemo-retriever-data/extracted-images on host
```

For more information on environment variables, refer to [Environment Variables](environment-config.md).



## Extract Audio

Use the following code to extract mp3 audio content.

```python
from nemo_retriever.client import Ingestor

ingestor = Ingestor().files("audio_file.mp3")

ingestor = ingestor.extract(
        document_type="mp3",
        extract_text=True,
        extract_tables=False,
        extract_charts=False,
        extract_images=False,
        extract_infographics=False,
    ).split(
        tokenizer="meta-llama/Llama-3.2-1B",
        chunk_size=150,
        chunk_overlap=0,
        params={"split_source_types": ["mp3"], "hf_access_token": "hf_***"}
    )

results = ingestor.ingest()
```



## Related Topics

- [Split Documents](chunking.md)
- [Troubleshoot NeMo Retriever Library](troubleshoot.md)
- [Advanced Visual Parsing](nemoretriever-parse.md)
- [Use the NeMo Retriever Library with Riva for Audio Processing](audio.md)
- [Use Multimodal Embedding](vlm-embed.md)
