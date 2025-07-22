# User-Defined Functions (UDFs) Guide

User-Defined Functions (UDFs) allow you to inject custom processing logic into the NV-Ingest pipeline at specific phases. This guide covers how to write, validate, and submit UDFs using both the CLI and the Python client interface.

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

Save your function to a file and submit it:

```bash
# Submit UDF for the "response" phase
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "my_file.py:my_custom_processor", "phase": "response"}'
```

### 3. Submit via Python Client

```python
from nv_ingest_client.client import NvIngestClient

client = NvIngestClient()
client.add_document("/path/to/document.pdf")
client.udf(
    udf_function="my_file.py:my_custom_processor",
    phase="response"
)
results = client.ingest()
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

### Pipeline Phases

UDFs can be executed at different phases of the pipeline:

| Phase | Value | Description | Use Cases |
|-------|-------|-------------|-----------|
| `pre_processing` | 0 | Before content extraction | Input validation, preprocessing |
| `extraction` | 1 | During content extraction | Custom extraction logic |
| `post_processing` | 2 | After extraction, before mutation | Content validation, filtering |
| `mutation` | 3 | Content transformation phase | Content modification, enhancement |
| `transform` | 4 | Embedding and analysis phase | Feature extraction, analysis |
| `response` | 5 | Final processing before output | Output formatting, final metadata |

#### Phase Selection Examples

```bash
# CLI examples for different phases
nv-ingest-cli --doc file.pdf --task 'udf:{"udf_function": "processor.py:validate_input", "phase": "pre_processing"}'
nv-ingest-cli --doc file.pdf --task 'udf:{"udf_function": "processor.py:extract_custom", "phase": "extraction"}'  
nv-ingest-cli --doc file.pdf --task 'udf:{"udf_function": "processor.py:enhance_output", "phase": "response"}'
```

```python
# Python client examples
client.udf(udf_function="processor.py:validate_input", phase="pre_processing")
client.udf(udf_function="processor.py:extract_custom", phase="extraction")
client.udf(udf_function="processor.py:enhance_output", phase="response")
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

### Advanced Examples

#### Content Filtering UDF
```python
def filter_short_content(control_message: IngestControlMessage) -> IngestControlMessage:
    """Remove content that's too short to be useful."""
    df = control_message.payload()
    
    # Filter out rows with content shorter than 50 characters
    mask = df['content'].str.len() >= 50
    filtered_df = df[mask].copy()
    
    control_message.payload(filtered_df)
    return control_message
```

#### Content Enhancement UDF
```python
import re
from datetime import datetime

def enhance_document_metadata(control_message: IngestControlMessage) -> IngestControlMessage:
    """Add custom metadata based on content analysis."""
    df = control_message.payload()
    
    for idx, row in df.iterrows():
        content = row['content']
        metadata = row['metadata']
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        
        # Extract phone numbers (simple pattern)
        phones = re.findall(r'\b\d{3}-\d{3}-\d{4}\b', content)
        
        # Add custom metadata
        if 'debug_metadata' not in metadata:
            metadata['debug_metadata'] = {}
            
        metadata['debug_metadata'].update({
            'extracted_emails': emails,
            'extracted_phones': phones,
            'content_length': len(content),
            'word_count': len(content.split()),
            'processing_timestamp': datetime.now().isoformat(),
            'contains_contact_info': len(emails) > 0 or len(phones) > 0
        })
        
        df.at[idx, 'metadata'] = metadata
    
    control_message.payload(df)
    return control_message
```

#### Multi-Phase Processing
```python
# Phase 1: Validation (pre_processing)
def validate_documents(control_message: IngestControlMessage) -> IngestControlMessage:
    """Validate document structure and add validation metadata."""
    df = control_message.payload()
    
    for idx, row in df.iterrows():
        metadata = row['metadata']
        
        # Add validation metadata
        if 'debug_metadata' not in metadata:
            metadata['debug_metadata'] = {}
            
        metadata['debug_metadata']['validation'] = {
            'has_content': len(row['content'].strip()) > 0,
            'has_source_file': bool(row.get('source_file')),
            'metadata_complete': 'source_metadata' in metadata,
            'validated_at': datetime.now().isoformat()
        }
        
        df.at[idx, 'metadata'] = metadata
    
    control_message.payload(df)
    return control_message

# Phase 2: Final Processing (response)
def finalize_output(control_message: IngestControlMessage) -> IngestControlMessage:
    """Add final metadata and prepare for output."""
    df = control_message.payload()
    
    for idx, row in df.iterrows():
        metadata = row['metadata']
        
        # Add final processing metadata
        if 'debug_metadata' not in metadata:
            metadata['debug_metadata'] = {}
            
        metadata['debug_metadata']['final_processing'] = {
            'pipeline_complete': True,
            'final_content_length': len(row['content']),
            'completed_at': datetime.now().isoformat()
        }
        
        df.at[idx, 'metadata'] = metadata
    
    control_message.payload(df)
    return control_message
```

### Error Handling and Best Practices

#### Robust Error Handling
```python
import logging

logger = logging.getLogger(__name__)

def robust_processor(control_message: IngestControlMessage) -> IngestControlMessage:
    """Example of robust UDF with proper error handling."""
    try:
        df = control_message.payload()
        
        for idx, row in df.iterrows():
            try:
                # Your processing logic here
                content = row['content']
                metadata = row['metadata']
                
                # Safe metadata access
                if isinstance(metadata, dict):
                    if 'debug_metadata' not in metadata:
                        metadata['debug_metadata'] = {}
                    
                    metadata['debug_metadata']['processed'] = True
                    df.at[idx, 'metadata'] = metadata
                else:
                    logger.warning(f"Row {idx}: metadata is not a dict, skipping")
                    
            except Exception as row_error:
                logger.error(f"Error processing row {idx}: {row_error}")
                # Continue processing other rows
                continue
        
        control_message.payload(df)
        
    except Exception as e:
        logger.error(f"Critical error in UDF: {e}")
        # Return original message if processing fails
        
    return control_message
```

#### Performance Best Practices

1. **Minimize DataFrame copies:**
```python
# ✅ Good - modify in place
def efficient_processor(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    # Use .at[] for single value assignment
    for idx, row in df.iterrows():
        df.at[idx, 'new_field'] = compute_value(row)
    
    control_message.payload(df)
    return control_message
```

2. **Use vectorized operations when possible:**
```python
# ✅ Good - vectorized operation
def vectorized_processor(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    # Vectorized string operation
    df['content_length'] = df['content'].str.len()
    
    control_message.payload(df)
    return control_message
```

3. **Cache expensive computations:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(text):
    # Some expensive processing
    return result

def cached_processor(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    for idx, row in df.iterrows():
        # Use cached computation
        result = expensive_computation(row['content'])
        df.at[idx, 'computed_field'] = result
    
    control_message.payload(df)
    return control_message
```

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
    --task 'udf:{"udf_function": "path/to/udf.py:function_name"}'
```

#### Multiple UDFs
```bash
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "validator.py:validate_input", "phase": "pre_processing"}' \
    --task 'udf:{"udf_function": "enhancer.py:enhance_content", "phase": "response"}'
```

### Python Client Reference

#### Basic Usage
```python
from nv_ingest_client.client import NvIngestClient

client = NvIngestClient()
client.add_document("/path/to/document.pdf")
client.udf(
    udf_function="/path/to/processor.py:my_function"
)
results = client.ingest()
```

#### Advanced Usage with Multiple UDFs
```python
client = NvIngestClient()
client.add_document("/path/to/document.pdf")

# Add validation UDF
client.udf(
    udf_function="validators.py:validate_structure", 
    phase="pre_processing"
)

# Add enhancement UDF  
client.udf(
    udf_function="enhancers.py:add_metadata",
    phase="response"
)

# Process with progress tracking
results = client.ingest(show_progress=True)
```

---

## Performance and Caching

The NV-Ingest system includes automatic LRU caching for UDF functions. Identical UDF code is compiled and validated only once, then cached for subsequent use. This provides significant performance benefits for repeated UDF usage without requiring any changes to your UDF code.

## Conclusion

UDFs provide a powerful way to customize the NV-Ingest pipeline for your specific needs. Start with simple metadata enhancements and gradually build more sophisticated processing logic as needed. Remember to follow the signature requirements, handle errors gracefully, and test your UDFs thoroughly before deployment.

For more information about the metadata structure and available fields, see the [metadata documentation](metadata_documentation.md).
