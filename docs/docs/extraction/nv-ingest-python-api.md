# Use the NV-Ingest Python API

The [NV-Ingest](overview.md) Python API provides a simple and flexible interface for processing and extracting information from various document types, including PDFs.

!!! tip

    There is a Jupyter notebook available to help you get started with the Python API. For more information, refer to [NV-Ingest: Python Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/python_client_usage.ipynb).

## Summary of Key Methods

| Method | Description |
| ------ | ----------- |
| `files` | Add document paths for processing. |
| `load` | Ensure files are locally accessible (downloads if needed). |
| `extract` | Add an extraction task (text, tables, charts). |
| `split` | Split documents into smaller sections for processing.
| `embed` | Generate embeddings from extracted content. |
| `caption` | Extract captions from images within the document. |
| `ingest` | Submit jobs and retrieve results synchronously. |

## Quick Start: Extracting PDFs

The following example demonstrates how to initialize `Ingestor`, load a PDF file, and extract its contents.
The `extract` method enables different types of data to be extracted.

### Extraction of a Single PDF

Use the following code to extract a single PDF file.

```python
from nv_ingest_client.client.interface import Ingestor

# Initialize Ingestor with a local PDF file
ingestor = Ingestor().files("path/to/document.pdf")

# Extract text, tables, and images
result = ingestor.extract().ingest()

print(result)
```

### Use the Python API to Extract PDFs

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
    extract_tables=False,  # Skip table extraction
    extract_charts=True,  # Extract charts
    extract_infographics=True,  # Extract infographic images
    extract_images=False,  # Skip image extraction
)
```

### Extract with Custom Document Type

Use the following code to specify a custom document type for extraction.

```python
ingestor = ingestor.extract(document_type="pdf")
```

### Track Job Progress

For large document batches, you can enable a progress bar by setting `show_progress` to true. 
Use the following code.

```python
result = ingestor.extract().ingest(show_progress=True)
```

## Using the Python API to Split Documents

Splitting, also known as chunking, breaks large documents or text into smaller, manageable sections to improve retrieval efficiency.
Use the `split` method to chunk large documents into smaller sections before processing as shown in the following code.

```python
ingestor = ingestor.split(
    tokenizer="meta-llama/Llama-3.2-1B",
    chunk_size=16,
    chunk_overlap=1,
    params={"split_source_types": ["text", "PDF"], "hf_access_token": "hf_***"},
)
```

!!! note

  The default tokenizer (`"meta-llama/Llama-3.2-1B"`) requires a [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens).
  You must set `"hf_access_token": "hf_***"` to authenticate.

If you prefer to use a different tokenizer, such as `"intfloat/e5-large-unsupervised"`, you can modify the `split` call as shown following.

```python
ingestor = ingestor.split(
    tokenizer="intfloat/e5-large-unsupervised",
    chunk_size=1024,
    chunk_overlap=150,
)
```

## Extracting Image Captions from Images

The `.caption()` method generates image captions using a vision-language model.
This can be used to describe images extracted from documents.

!!! note

    The default captioning model used by `.caption()` is `"meta/llama-3.2-11b-vision-instruct"`.

```python
ingestor = ingestor.caption()
```

To specify a different API endpoint, pass additional parameters to `.caption()`.
```python
ingestor = ingestor.caption(
    endpoint_url="https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions",
    model_name="meta/llama-3.2-11b-vision-instruct",
    api_key="nvapi-",
)
```

## Extracting Embeddings with NV-Ingest

The `.embed()` method in NV-Ingest generates text embeddings for document content.
```python
ingestor = ingestor.embed()
```

!!! note

    By default, `.embed()` uses the [llama-3.2-nv-embedqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2) model.

If you prefer to use a different embedding model, such as [nv-embedqa-e5-v5](https://build.nvidia.com/nvidia/nv-embedqa-e5-v5), specify a different `model_name` and `endpoint_url`.
```python
ingestor = ingestor.embed(
    endpoint_url="https://integrate.api.nvidia.com/v1",
    model_name="nvidia/nv-embedqa-e5-v5",
    api_key="nvapi-",
)
```
