# Use the NV-Ingest Python API

The [NV-Ingest](overview.md) Python API provides a simple and flexible interface for processing and extracting information from various document types, including PDFs.

!!! tip

    There is a Jupyter notebook available to help you get started with the Python API. For more information, refer to [Python Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/python_client_usage.ipynb).

## Summary of Key Methods

The main class in the nv-ingest API is `Ingestor`. 
The `Ingestor` class provides an interface for building, managing, and running data ingestion jobs, enabling for chainable task additions and job state tracking. 
The following table describes methods of the `Ingestor` class.

| Method       | Description                       |
|--------------|-----------------------------------|
| `caption`    | Extract captions from images within the document. |
| `embed`      | Generate embeddings from extracted content. |
| `extract`    | Add an extraction task (text, tables, charts). |
| `files`      | Add document paths for processing. |
| `ingest`     | Submit jobs and retrieve results synchronously. |
| `load`       | Ensure files are locally accessible (downloads if needed). |
| `split`      | Split documents into smaller sections for processing. For more information, refer to [Split Documents](chunking.md). |
| `vdb_upload` | Pushes extraction results to Milvus vector database. For more information, refer to [Data Upload](data-store.md). |



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

- **return_failures=False** – If any job fails, the `ingest` method raises a runtime error and does not upload any data (all-or-nothing data upload). This is the default setting.
- **return_failures=True** – If any jobs succeed, the results from those jobs are uploaded, and no errors are raised (partial data upload). The `ingest()` method returns a failures object that contains the details for any jobs that failed. You can inspect the failures object and selectively retry or remediate the failed jobs.


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
from nv_ingest_client.client.interface import Ingestor

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



## Extract Captions from Images

The `caption` method generates image captions by using a vision-language model. 
This can be used to describe images extracted from documents.

!!! note

    The default model used by `caption` is `meta/llama-3.2-11b-vision-instruct`.

```python
ingestor = ingestor.caption()
```

To specify a different API endpoint, pass additional parameters to `caption`.

```python
ingestor = ingestor.caption(
    endpoint_url="https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions",
    model_name="meta/llama-3.2-11b-vision-instruct",
    api_key="nvapi-"
)
```



## Extract Embeddings

The `embed` method in NV-Ingest generates text embeddings for document content.

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



## Related Topics

- [Split Documents](chunking.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
- [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md)
- [Use NeMo Retriever Extraction with Riva for Audio Processing](nemoretriever-parse.md)
- [Use Multimodal Embedding](vlm-embed.md)
