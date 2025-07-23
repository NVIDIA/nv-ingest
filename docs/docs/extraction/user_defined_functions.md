# User-Defined Functions (UDFs) Guide

User-Defined Functions (UDFs) allow you to inject custom processing logic into the NV-Ingest pipeline at specific stages. This guide covers how to write, validate, and submit UDFs using both the CLI and the Python client interface.

## Quickstart

### 1. Write Your UDF

Create a Python function that accepts an `IngestControlMessage` and returns a modified `IngestControlMessage`:

```python
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def my_custom_processor(control_message: IngestControlMessage) -> IngestControlMessage:
    """Add custom metadata to all documents."""
    # Get the DataFrame payload
    df = control_message.payload()
    
    # Add custom metadata to each row
    for idx, row in df.iterrows():
        if 'metadata' in row and isinstance(row['metadata'], dict):
            # Add your custom field
            df.at[idx, 'metadata']['custom_field'] = 'my_custom_value'
    
    # Update the payload with modified DataFrame
    control_message.payload(df)
    return control_message
```

### 2. Submit via CLI

Save your function to a file and submit it to run before a specific pipeline stage:

```bash
# Submit UDF to run before the text embedding stage
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "my_file.py:my_custom_processor", "target_stage": "text_embedder", "run_before": true}'

# Submit UDF to run after the text embedding stage
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "my_file.py:my_custom_processor", "target_stage": "text_embedder", "run_after": true}'
```

### 3. Submit via Python Client

```python
from nv_ingest_client.client.interface import Ingestor

# Create an Ingestor instance with default client
ingestor = Ingestor()

# Add documents and configure UDF to run before text embedding
results = ingestor.files("/path/to/document.pdf") \
    .extract() \
    .udf(
        udf_function="my_file.py:my_custom_processor",
        target_stage="text_embedder",
        run_before=True
    ) \
    .embed() \
    .store() \
    .ingest()

# Alternative: UDF to run after text embedding stage
results = ingestor.files("/path/to/document.pdf") \
    .extract() \
    .embed() \
    .udf(
        udf_function="my_file.py:my_custom_processor", 
        target_stage="text_embedder",
        run_after=True
    ) \
    .store() \
    .ingest()

# Using phase-based targeting (legacy approach)
results = ingestor.files("/path/to/document.pdf") \
    .extract() \
    .udf(
        udf_function="my_file.py:my_custom_processor",
        phase="embed"  # or phase=4
    ) \
    .embed() \
    .store() \
    .ingest()
```

---

## Comprehensive Documentation

### Understanding IngestControlMessage (ICM)

The `IngestControlMessage` is the primary data structure that flows through the NV-Ingest pipeline. Your UDF receives an ICM and must return a (potentially modified) ICM.

#### Key ICM Methods

```python
# Get the pandas DataFrame payload
df = control_message.payload()

# Update the payload with a modified DataFrame
control_message.payload(modified_df)

# Access metadata
metadata = control_message.get_metadata("some_key")
control_message.set_metadata("some_key", "some_value")

# Get tasks (advanced usage)
tasks = control_message.get_tasks()
```

### Understanding the DataFrame Payload

The DataFrame payload contains the extracted content and metadata for processing. Each row represents a piece of content (text, image, table, etc.).

#### Core DataFrame Columns

| Column | Type | Description |
|--------|------|-------------|
| `document_type` | `str` | Type of document (pdf, docx, txt, etc.) |
| `source_type` | `str` | Source type identifier |
| `source_file` | `str` | Path or identifier of the source file |
| `id` | `str` | Unique identifier for this content piece |
| `metadata` | `dict` | Rich metadata structure (see below) |
| `content` | `str` | The actual extracted content |

#### Example DataFrame Access

```python
def process_text_content(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    for idx, row in df.iterrows():
        # Access core fields
        doc_type = row['document_type']
        content = row['content']
        metadata = row['metadata']
        
        # Modify content or metadata
        if doc_type == 'pdf' and len(content) > 1000:
            # Add a summary for long PDF content
            df.at[idx, 'metadata']['content_summary'] = content[:200] + "..."
    
    control_message.payload(df)
    return control_message
```

### Metadata Structure

The `metadata` field in each DataFrame row contains a rich, nested structure with information about the content. The metadata follows a standardized schema with the following top-level keys:

> **⚠️ Important Notes:**
> - **Full Modification Access**: UDFs can modify any field defined by the MetadataSchema. However, corrupting or removing data expected by downstream stages may cause job failures.
> - **Custom Content Field**: The top-level `metadata` object includes an unvalidated `custom_content` dictionary where you can place any valid, JSON-serializable data for your custom use cases.

#### Top-Level Metadata Structure

```python
{
    "content": "The extracted text content",
    "custom_content": {},  # Your custom JSON-serializable data goes here
    "content_metadata": {
        "type": "text|image|audio|structured",
        "page_number": 1,
        "description": "Content description",
        "hierarchy": {...},  # Content hierarchy information
        "subtype": "",
        # ... other content-specific fields
    },
    "source_metadata": {
        "source_id": "unique_source_identifier",
        "source_name": "document.pdf",
        "source_type": "pdf",
        "source_location": "/path/to/document.pdf",
        "collection_id": "",
        "date_created": "2024-01-01T00:00:00",
        "last_modified": "2024-01-01T00:00:00",
        "summary": "",
        "partition_id": -1,
        "access_level": 0
    },
    "text_metadata": {  # Present when content_metadata.type == "text"
        "text_type": "document",
        "summary": "",
        "keywords": "",
        "language": "unknown",
        "text_location": [0, 0, 0, 0],
        "text_location_max_dimensions": [0, 0, 0, 0]
    },
    "image_metadata": {  # Present when content_metadata.type == "image"
        "image_type": "png",
        "structured_image_type": "none",
        "caption": "",
        "text": "",
        "image_location": [0, 0, 0, 0],
        "image_location_max_dimensions": [0, 0],
        "uploaded_image_url": "",
        "width": 0,
        "height": 0
    },
    "audio_metadata": {  # Present when content_metadata.type == "audio"
        "audio_type": "wav",
        "audio_transcript": ""
    },
    "error_metadata": null,  # Contains error information if processing failed
    "debug_metadata": {}     # Arbitrary debug information
}
```

#### Example Metadata Manipulation

```python
def enhance_metadata(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    for idx, row in df.iterrows():
        metadata = row['metadata']
        
        # Add custom fields to source metadata
        if 'source_metadata' in metadata:
            metadata['source_metadata']['custom_processing_date'] = datetime.now().isoformat()
            metadata['source_metadata']['custom_processor_version'] = "1.0.0"
        
        # Add content-specific enhancements
        if metadata.get('content_metadata', {}).get('type') == 'text':
            content = metadata.get('content', '')
            # Add word count
            metadata['text_metadata']['word_count'] = len(content.split())
            
        # Update the row
        df.at[idx, 'metadata'] = metadata
    
    control_message.payload(df)
    return control_message
```

> **📖 For detailed metadata schema documentation, see:** [metadata_documentation.md](metadata_documentation.md)

### UDF Targeting

UDFs can be executed at different stages of the pipeline by specifying the `target_stage` parameter. The following stages are available in the default pipeline configuration:

#### Available Pipeline Stages

**Pre processing Stages (Phase 0):**
- `metadata_injector` - Metadata injection stage

**Extraction Stages (Phase 1):**
- `pdf_extractor` - PDF content extraction
- `audio_extractor` - Audio content extraction  
- `docx_extractor` - DOCX document extraction
- `pptx_extractor` - PowerPoint presentation extraction
- `image_extractor` - Image content extraction
- `html_extractor` - HTML document extraction
- `infographic_extractor` - Infographic content extraction
- `table_extractor` - Table structure extraction
- `chart_extractor` - Chart and graphic extraction

**Mutation Stages (Phase 3):**
- `image_filter` - Image filtering and validation
- `image_dedup` - Image deduplication

**Transform Stages (Phase 4):**
- `text_splitter` - Text chunking and splitting
- `image_caption` - Image captioning and description
- `text_embedder` - Text embedding generation

**Storage Stages (Phase 5):**
- `image_storage` - Image storage and management
- `embedding_storage` - Embedding storage and indexing
- `broker_response` - Response message handling
- `otel_tracer` - OpenTelemetry tracing

> **Note:** For the complete and up-to-date list of pipeline stages, see the [default_pipeline.yaml](../../../config/default_pipeline.yaml) configuration file.

#### Target Stage Selection Examples

```bash
# CLI examples for different target stages
nv-ingest-cli --doc file.pdf --task 'udf:{"udf_function": "processor.py:validate_input", "target_stage": "pdf_extractor", "run_before": true}'
nv-ingest-cli --doc file.pdf --task 'udf:{"udf_function": "processor.py:extract_custom", "target_stage": "text_embedder", "run_after": true}'
nv-ingest-cli --doc file.pdf --task 'udf:{"udf_function": "processor.py:enhance_output", "target_stage": "embedding_storage", "run_before": true}'
```

```python
# Python client examples
ingestor = Ingestor()

ingestor.udf(udf_function="processor.py:validate_input", target_stage="pdf_extractor", run_before=True
    ).udf(udf_function="processor.py:extract_custom", target_stage="text_embedder", run_after=True
    ).udf(udf_function="processor.py:enhance_output", target_stage="embedding_storage", run_before=True)
```

### UDF Function Requirements

#### Signature Requirements

Your UDF function **must**:

1. Accept exactly one parameter named `control_message` with type annotation `IngestControlMessage`
2. Return an `IngestControlMessage`
3. Have proper type annotations

```python
# ✅ Correct signature
def my_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    # Your processing logic
    return control_message

# ❌ Incorrect - missing type annotations
def my_udf(control_message):
    return control_message

# ❌ Incorrect - wrong parameter name
def my_udf(message: IngestControlMessage) -> IngestControlMessage:
    return message

# ❌ Incorrect - multiple parameters
def my_udf(control_message: IngestControlMessage, config: dict) -> IngestControlMessage:
    return control_message
```

#### Essential Patterns

**Always return the control message:**
```python
def my_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    # ... your processing ...
    return control_message  # Don't forget this!
```

**Update the payload after DataFrame modifications:**
```python
def my_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    # Modify the DataFrame
    df['new_column'] = 'new_value'
    
    # IMPORTANT: Update the payload
    control_message.payload(df)
    
    return control_message
```

### UDF Function Specification Formats

#### File Path with Function Name
```python
# File: /path/to/my_processors.py
def process_documents(control_message: IngestControlMessage) -> IngestControlMessage:
    # ... processing logic ...
    return control_message
```

```bash
# CLI usage
--task 'udf:{"udf_function": "/path/to/my_processors.py:process_documents"}'
```

#### Import Path Format
```python
# File: my_package/processors.py
def advanced_processor(control_message: IngestControlMessage) -> IngestControlMessage:
    # ... processing logic ...
    return control_message
```

```bash
# CLI usage (if my_package is in Python path)
--task 'udf:{"udf_function": "my_package.processors:advanced_processor"}'
```

### Error Handling

The NV-Ingest system automatically catches all exceptions that occur within UDF execution. If your UDF fails for any reason, the system will:

1. Annotate the job with appropriate error information
2. Mark the job as failed
3. Return the failed job to you with error details
4. Failures that are not caught by the system, or unhandled exceptions (segfaults) from acceleration libraries may leave the pipeline in an unstable state 

You do not need to implement extensive error handling within your UDF - focus on your core processing logic and let the system handle failures gracefully.

### Debugging and Testing

#### Local Testing
```python
# Create a test script to validate your UDF
import pandas as pd
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def test_my_udf():
    # Create test data
    test_data = pd.DataFrame({
        'document_type': ['pdf'],
        'source_type': ['file'],
        'source_file': ['test.pdf'],
        'id': ['test_id_1'],
        'content': ['This is test content'],
        'metadata': [{'content': 'This is test content', 'source_metadata': {}}]
    })
    
    # Create test control message
    control_message = IngestControlMessage()
    control_message.payload(test_data)
    
    # Test your UDF
    result = my_custom_processor(control_message)
    
    # Validate results
    result_df = result.payload()
    print("Result DataFrame:")
    print(result_df)
    
    return result

if __name__ == "__main__":
    test_my_udf()
```

#### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `TypeError: Parameter must be annotated` | Missing type annotations | Add proper type annotations to function signature |
| `ValueError: UDF must return IngestControlMessage` | Not returning control message | Always return the control_message parameter |
| `Payload cannot be None` | Not setting payload after modification | Call `control_message.payload(df)` after DataFrame changes |
| Function not found | Incorrect function name or path | Verify file path and function name are correct |

### CLI Reference

#### Basic UDF Submission
```bash
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "path/to/udf.py:function_name", "target_stage": "text_extractor", "run_before": true}'
```

#### Multiple UDFs
```bash
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "validator.py:validate_input", "target_stage": "text_extractor", "run_before": true}' \
    --task 'udf:{"udf_function": "enhancer.py:enhance_content", "target_stage": "text_embedder", "run_after": true}'
```

---

## Conclusion

UDFs provide a powerful way to customize the NV-Ingest pipeline for your specific needs. Start with simple metadata enhancements and gradually build more sophisticated processing logic as needed. Remember to follow the signature requirements, handle errors gracefully, and test your UDFs thoroughly before deployment.

For more information about the metadata structure and available fields, see the [metadata documentation](metadata_documentation.md).
