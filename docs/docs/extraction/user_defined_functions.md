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

The CLI supports all UDF function specification formats. Here are examples of each:

#### Inline Function String
```bash
# Submit inline UDF function
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "def my_processor(control_message): print(\"Processing...\"); return control_message", "udf_function_name": "my_processor", "target_stage": "text_embedder", "run_before": true}'
```

#### Module Path with Colon (Recommended)
```bash
# Submit UDF from importable module (preserves all imports and context)
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "my_package.processors:enhance_metadata", "target_stage": "text_embedder", "run_after": true}'
```

#### File Path
```bash
# Submit UDF from file path
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "my_file.py:my_custom_processor", "target_stage": "text_embedder", "run_before": true}'
```

#### Legacy Import Path (Limited)
```bash
# Submit UDF using legacy dot notation (function only, no imports)
nv-ingest-cli \
    --doc /path/to/document.pdf \
    --output-directory ./output \
    --task 'udf:{"udf_function": "my_package.processors.basic_processor", "target_stage": "text_embedder", "run_after": true}'
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
    .ingest()

# Using phase-based targeting (legacy approach)
results = ingestor.files("/path/to/document.pdf") \
    .extract() \
    .udf(
        udf_function="my_file.py:my_custom_processor",
        phase="embed"  # or phase=4
    ) \
    .embed() \
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

**Post-processing Stages (Phase 2):**
> **Note:** There are currently no Phase 2 stages in the default pipeline. This phase is reserved for future use and may include stages for content validation, quality assessment, or intermediate processing steps between extraction and mutation phases.

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

NV-Ingest supports four different formats for specifying UDF functions:

### 1. Inline Function String
Define your function directly as a string:

```python
udf_function = """
def my_custom_processor(control_message):
    # Your processing logic here
    payload = control_message.payload()
    # Modify the payload as needed
    return control_message
"""
```

### 2. Module Path with Colon (Recommended)
Reference a function from an importable module while preserving all imports and context:

```python
# Format: 'module.submodule:function_name'
udf_function = "my_package.processors.text_utils:enhance_metadata"
```

**Benefits:**
- ✅ Preserves all module-level imports (`import pandas as pd`, etc.)
- ✅ Includes helper functions and variables the UDF depends on
- ✅ Maintains full execution context
- ✅ Best for complex UDFs with dependencies

### 3. File Path
Reference a function from a specific Python file:

```python
# With function name: 'path/to/file.py:function_name'
udf_function = "/path/to/my_udfs.py:my_custom_processor"

# Without function name (assumes 'process' function)
udf_function = "/path/to/my_udfs.py"
```

**Benefits:**
- ✅ Preserves all file-level imports and context
- ✅ Works with files not in Python path
- ✅ Good for standalone UDF files

### 4. Legacy Import Path (Limited)
Reference a function using dot notation (legacy format):

```python
# Format: 'module.submodule.function_name'
udf_function = "my_package.processors.text_utils.enhance_metadata"
```

**Limitations:**
- ⚠️ Only extracts the function source code
- ⚠️ Does NOT include module imports or dependencies
- ⚠️ May fail if function depends on imports
- ⚠️ Use format #2 instead for better reliability

## Recommendation

**Use format #2 (Module Path with Colon)** for most use cases as it provides the best balance of functionality and reliability by preserving the complete execution context your UDF needs.

### UDF Function Specification Examples

```bash
# CLI usage
--task 'udf:{"udf_function": "path/to/my_processors.py:process_documents"}'
```

```python
# Python client usage
ingestor.udf(udf_function="my_package.processors.text_utils:enhance_metadata")
```

## Integrating with NVIDIA NIMs

NVIDIA Inference Microservices (NIMs) provide powerful AI capabilities that can be seamlessly integrated into your UDFs. The `NimClient` class offers a unified interface for connecting to and using NIMs within the NV-Ingest pipeline.

### Quick NIM Integration

```python
from nv_ingest_api.internal.primitives.control_message import IngestControlMessage
from nv_ingest_api.util.nim import create_inference_client
from nv_ingest_api.internal.primitives.nim.model_interface.vlm import VLMModelInterface
import os

def document_analysis_with_nim(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF that uses a NIM to analyze document content."""
    
    # Create NIM client for text analysis
    model_interface = VLMModelInterface()
    client = create_inference_client(
        endpoints=(
            os.getenv("ANALYSIS_NIM_GRPC", "grpc://analysis-nim:8001"),
            os.getenv("ANALYSIS_NIM_HTTP", "http://analysis-nim:8000")
        ),
        model_interface=model_interface,
        auth_token=os.getenv("NGC_API_KEY"),
        infer_protocol="http"
    )
    
    df = control_message.payload()
    
    for idx, row in df.iterrows():
        if row.get("content"):
            try:
                # Perform NIM inference
                results = client.infer(
                    data={
                        "base64_images": [row.get("image_data", "")],
                        "prompt": f"Analyze this document: {row['content'][:500]}"
                    },
                    model_name="llava-1.5-7b-hf"
                )
                
                # Add analysis to DataFrame
                df.at[idx, "nim_analysis"] = results[0] if results else "No analysis"
                
            except Exception as e:
                print(f"NIM inference failed: {e}")
                df.at[idx, "nim_analysis"] = "Analysis failed"
    
    control_message.payload(df)
    return control_message
```

### Environment Configuration

Set these environment variables for your NIM endpoints:

```bash
# NIM service endpoints
export ANALYSIS_NIM_GRPC="grpc://your-nim-service:8001"
export ANALYSIS_NIM_HTTP="http://your-nim-service:8000"

# Authentication (if required)
export NGC_API_KEY="your-ngc-api-key"
```

### Available NIM Interfaces

NV-Ingest provides several pre-built model interfaces:

- **VLMModelInterface**: Vision-Language Models for image analysis and captioning
- **EmbeddingModelInterface**: Text embedding generation
- **OCRModelInterface**: Optical Character Recognition
- **YoloxModelInterface**: Object detection and page element extraction

### Creating Custom NIM Integrations

For detailed guidance on creating custom NIM integrations, including:

- Custom ModelInterface implementation
- Protocol handling (gRPC vs HTTP)
- Batch processing optimization
- Error handling and debugging
- Performance best practices

See the comprehensive [**NimClient Usage Guide**](nimclient_usage.md).

### Error Handling

The NV-Ingest system automatically catches all exceptions that occur within UDF execution. If your UDF fails for any reason, the system will:

1. Annotate the job with appropriate error information
2. Mark the job as failed
3. Return the failed job to you with error details
4. Failures that are not caught by the system, or unhandled exceptions (segfaults) from acceleration libraries may leave the pipeline in an unstable state 

You do not need to implement extensive error handling within your UDF - focus on your core processing logic and let the system handle failures gracefully.

### Performance Considerations

UDFs execute within the NV-Ingest pipeline and can significantly impact overall system performance and stability. Understanding these considerations is crucial for maintaining optimal pipeline throughput and reliability.

#### Pipeline Impact

**Global Slowdown on Congested Stages:**
- UDFs run synchronously within pipeline stages, blocking other processing until completion
- Heavy-weight UDFs on high-traffic stages (e.g., `text_embedder`, `pdf_extractor`) can create bottlenecks
- A single slow UDF can reduce throughput for the entire pipeline
- Consider the stage's typical workload when designing UDF complexity

**Stage Selection Strategy:**
```python
# ❌ Avoid heavy processing on high-throughput stages
ingestor.udf(
    udf_function="heavy_ml_processing.py:complex_analysis",
    target_stage="text_embedder",  # High-traffic stage - will slow everything down
    run_before=True
)

# ✅ Better: Use less congested stages or run after processing
ingestor.udf(
    udf_function="heavy_ml_processing.py:complex_analysis", 
    target_stage="embedding_storage",  # Lower-traffic stage
    run_before=True
)
```

#### Memory Management

**Memory Consumption:**
- UDFs share memory with the pipeline worker processes
- Excessive memory usage can trigger out-of-memory (OOM) kills
- Large DataFrame modifications can cause memory spikes
- Memory leaks in UDFs accumulate over time and destabilize workers

**Best Practices:**
```python
# ❌ Memory-intensive operations
def memory_heavy_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    # Creates large temporary objects
    large_temp_data = [expensive_computation(row) for row in df.itertuples()]
    
    # Multiple DataFrame copies
    df_copy1 = df.copy()
    df_copy2 = df.copy()
    df_copy3 = df.copy()
    
    return control_message

# ✅ Memory-efficient approach
def memory_efficient_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    try:
        # Load model once and reuse (consider caching)
        model = get_cached_model()
        
        # Batch processing with error handling
        batch_results = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            # Process chunk in-place
            for idx in chunk.index:
                df.at[idx, 'new_field'] = efficient_computation(df.at[idx, 'content'])
        
        df['result'] = batch_results
        
    except Exception as e:
        logger.error(f"UDF failed: {e}")
        # Return original message on failure
        return control_message
    finally:
        # Explicit cleanup if needed
        cleanup_resources()
    
    control_message.payload(df)
    return control_message
```

#### Computational Complexity

**CPU-Intensive Operations:**
- Complex algorithms can monopolize CPU resources
- Long-running computations block the pipeline stage
- Consider computational complexity relative to document size

**I/O Operations:**
- File system access, network requests, and database queries add latency
- Synchronous I/O blocks the entire pipeline stage
- External service dependencies introduce failure points

```python
# ❌ Blocking I/O in UDF
def blocking_io_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    for idx, row in df.iterrows():
        # Blocks pipeline for each external call
        result = requests.get(f"https://api.example.com/process/{row['id']}")
        df.at[idx, 'external_data'] = result.json()
    
    control_message.payload(df)
    return control_message

# ✅ Batch processing with timeouts
def efficient_io_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    df = control_message.payload()
    
    # Batch requests and set reasonable timeouts
    ids = df['id'].tolist()
    try:
        response = requests.post(
            "https://api.example.com/batch_process",
            json={"ids": ids},
            timeout=5.0  # Prevent hanging
        )
        results = response.json()
        
        # Update DataFrame efficiently
        df['external_data'] = df['id'].map(results.get)
        
    except requests.RequestException as e:
        logger.warning(f"External API failed: {e}")
        df['external_data'] = None  # Graceful fallback
    
    control_message.payload(df)
    return control_message
```

#### System Stability Risks

**Segmentation Faults:**
- Native library crashes (C extensions) can kill worker processes
- Segfaults may leave the pipeline in an unstable state
- Worker restarts cause job failures and processing delays

**Resource Exhaustion:**
- File descriptor leaks from unclosed resources
- Thread pool exhaustion from concurrent operations

**Common Stability Issues:**
```python
# ❌ Potential stability risks
def risky_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF with potential stability risks."""
    logger = logging.getLogger(__name__)
    
    try:
        df = control_message.get_payload()
        logger.info(f"Processing {len(df)} documents")
        
        # Load model repeatedly (memory intensive)
        model = load_large_ml_model()
        
        # Native library calls without error handling
        for idx, row in df.iterrows():
            result = unsafe_native_function(row['content'])  # Could segfault
            df.at[idx, 'result'] = result
        
        logger.info("UDF processing completed successfully")
        control_message.payload(df)
        return control_message
        
    except Exception as e:
        logger.error(f"UDF failed: {e}", exc_info=True)
        # Return original message on failure
        return control_message
    finally:
        # No explicit cleanup
        pass

# ✅ Stable approach with resource management
def stable_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF with proper resource management."""
    logger = logging.getLogger(__name__)
    
    try:
        df = control_message.get_payload()
        logger.info(f"Processing {len(df)} documents")
        
        # Load model once and reuse (consider caching)
        model = get_cached_model()
        
        # Batch processing with error handling
        batch_results = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            # Process chunk in-place
            for idx in chunk.index:
                df.at[idx, 'new_field'] = efficient_computation(df.at[idx, 'content'])
        
        df['result'] = batch_results
        
        logger.info("UDF processing completed successfully")
        control_message.payload(df)
        return control_message
        
    except Exception as e:
        logger.error(f"UDF failed: {e}", exc_info=True)
        # Return original message on failure
        return control_message
    finally:
        # Explicit cleanup if needed
        cleanup_resources()
```

#### Performance Monitoring

**Key Metrics to Monitor:**
- UDF execution time per document
- Memory usage during UDF execution
- Pipeline stage throughput before/after UDF deployment
- Worker process restart frequency
- Job failure rates

**Profiling UDFs:**
```python
import time
import psutil
import logging

def profiled_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF with profiling."""
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    df = control_message.payload()
    
    # Your UDF logic here
    # ... processing ...
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    execution_time = end_time - start_time
    memory_delta = end_memory - start_memory
    
    if execution_time > 5.0:  # Log slow UDFs
        logger.warning(f"Slow UDF execution: {execution_time:.2f}s")
    
    if memory_delta > 100:  # Log high memory usage
        logger.warning(f"High memory usage: {memory_delta:.2f}MB")
    
    control_message.payload(df)
    return control_message
```

#### Recommendations

**Development Guidelines:**
1. **Profile Early:** Test UDFs with realistic data volumes
2. **Optimize for Stage:** Consider the target stage's typical workload
3. **Limit Complexity:** Keep UDFs focused and lightweight
4. **Handle Errors:** Implement graceful fallbacks for external dependencies
5. **Monitor Impact:** Track pipeline performance after UDF deployment

**Production Deployment:**
1. **Gradual Rollout:** Deploy UDFs to a subset of documents first
2. **Resource Limits:** Set appropriate memory and CPU limits
3. **Monitoring:** Implement alerting for performance degradation
4. **Rollback Plan:** Have a quick way to disable problematic UDFs

**When to Avoid UDFs:**
- For simple metadata additions that could be done post-processing
- When external service dependencies are unreliable
- For computationally expensive operations that could be batched offline
- When the processing logic changes frequently

Remember: UDFs are powerful but should be used judiciously. Poor UDF design can significantly impact the entire pipeline's performance and stability.

### Debugging and Testing

## Global UDF Control

You can globally disable all UDF processing using an environment variable:

```bash
# Disable all UDF execution across the entire pipeline
export INGEST_DISABLE_UDF_PROCESSING=1
```

**When to Use:**
Setting `INGEST_DISABLE_UDF_PROCESSING` to any non-empty value will disable all UDF processing across the entire pipeline. This is useful for:

- **Debugging:** Isolate pipeline issues by removing UDF interference
- **Performance Testing:** Measure baseline pipeline throughput without UDF overhead
- **Emergency Situations:** Quickly disable UDFs causing instability or crashes
- **Maintenance:** Temporary bypass during troubleshooting or system updates
- **Rollback:** Quick way to disable problematic UDFs in production

**Behavior:**
When disabled, all UDF tasks remain in control messages but are not executed. The pipeline runs normally without any UDF processing overhead, allowing you to verify that issues are UDF-related.

```bash
# Examples of values that disable UDF processing
export INGEST_DISABLE_UDF_PROCESSING=1
export INGEST_DISABLE_UDF_PROCESSING=true
export INGEST_DISABLE_UDF_PROCESSING=disable
export INGEST_DISABLE_UDF_PROCESSING=any_non_empty_value

# UDF processing is enabled (default behavior)
unset INGEST_DISABLE_UDF_PROCESSING
# OR
export INGEST_DISABLE_UDF_PROCESSING=""
```

## Testing UDFs

When developing and testing UDFs, consider these approaches:

### Local Testing

Test your UDF functions in isolation before deploying them to the pipeline:

```python
import pandas as pd
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def test_my_udf():
    # Create test data
    test_df = pd.DataFrame({
        'content': ['test document 1', 'test document 2'],
        'metadata': [{'source': 'test1'}, {'source': 'test2'}]
    })
    
    # Create control message
    control_message = IngestControlMessage()
    control_message.payload(test_df)
    
    # Test your UDF
    result = my_custom_processor(control_message)
    
    # Verify results
    result_df = result.get_payload()
    print(result_df)
    assert 'custom_field' in result_df.iloc[0]['metadata']

# Run the test
test_my_udf()
```

### Pipeline Integration Testing

Test UDFs in a controlled pipeline environment:

1. **Start with small datasets** to verify basic functionality
2. **Use the disable flag** to compare pipeline behavior with/without UDFs
3. **Monitor resource usage** during UDF execution
4. **Test error scenarios** to ensure graceful failure handling

### Common Debugging Techniques

```python
import logging

def debug_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF with comprehensive debugging."""
    logger = logging.getLogger(__name__)
    
    try:
        df = control_message.get_payload()
        logger.info(f"Processing {len(df)} documents")
        
        # Log input data structure
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"Sample row: {df.iloc[0].to_dict()}")
        
        # Your processing logic here
        for idx, row in df.iterrows():
            logger.debug(f"Processing row {idx}: {row.get('content', '')[:50]}...")
            # ... your logic ...
        
        logger.info("UDF processing completed successfully")
        control_message.payload(df)
        return control_message
        
    except Exception as e:
        logger.error(f"UDF failed: {e}", exc_info=True)
        # Return original message on failure
        return control_message
```
