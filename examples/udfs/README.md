# NV-Ingest UDF Examples

User-Defined Functions (UDFs) let you inject custom processing logic into the NV-Ingest pipeline at specific stages. This directory contains practical examples to get you started.

## Structural Text Splitter Example

**Problem**: You have markdown documents with hierarchical structure (headers like `#`, `##`, `###`) that you want to ingest. The default text splitter doesn't preserve this structure, losing important document organization.

**Solution**: The `structural_split_udf.py` splits documents at header boundaries while preserving hierarchical metadata.

### Quick Start

1. **Understand the UDF Pattern**
   ```python
   def my_udf(control_message: IngestControlMessage) -> IngestControlMessage:
       df = control_message.payload()  # Get documents as DataFrame
       # Process documents (modify df)
       control_message.payload(df)     # Return updated documents
       return control_message
   ```

2. **Use with CLI**
   ```bash
   nv-ingest-cli \
     --doc './my_markdown_docs/' \
     --task='extract' \
     --task='udf:{"udf_function": "./examples/udfs/structural_split_udf.py:structural_split", "target_stage": "text_splitter", "run_before": true}' \
     --task='embed' \
     --output_directory=./output
   ```

3. **Use with Python API**
   ```python
   from nv_ingest_client.client.interface import Ingestor

   ingestor = Ingestor()
   results = ingestor.files("./my_markdown_docs/") \
       .extract() \
       .udf(
           udf_function="./examples/udfs/structural_split_udf.py:structural_split",
           target_stage="text_splitter", 
           run_before=True
       ) \
       .embed() \
       .ingest()
   ```

### What You Get

- **Input**: 1 document with markdown content  
- **Output**: N documents split at header boundaries
- **Enhanced Metadata**: Each chunk includes hierarchical information like header level, chunk index, and splitting method

### Implementation Details

The UDF processes documents by:
1. Filtering for text documents with `source_type: "text"`
2. Extracting content from document metadata
3. Splitting text at markdown header boundaries (`#`, `##`, etc.)
4. Creating new document rows for each chunk with enriched metadata
5. Returning the updated DataFrame with all chunks

### Customization

Adapt the pattern for your needs:
```python
# Split on custom patterns
markdown_headers = ["#", "##", "===", "---"]

# Process different document types  
if row.get("document_type") == "html":
    # Handle HTML headers
    
# Add custom metadata
metadata["custom_content"]["document_category"] = detect_category(content)
```

## LLM Content Summarizer Example

**Problem**: You want to generate AI-powered summaries of your text content after ingesting PDFs or other documents. This helps with downstream processing, search relevance, and content understanding.

**Solution**: The `llm_summarizer_udf.py` uses OpenAI-compatible APIs (including NVIDIA NIMs) to generate concise summaries of text chunks.

### Quick Start

1. **Set up environment**
   ```bash
   export NVIDIA_API_KEY="your-nvidia-api-key"
   export LLM_SUMMARIZATION_MODEL="nvidia/llama-3.1-nemotron-70b-instruct"
   ```

2. **Use with CLI**
   ```bash
   nv-ingest-cli \
     --doc './my_pdfs/' \
     --task='extract' \
     --task='split' \
     --task='udf:{"udf_function": "./examples/udfs/llm_summarizer_udf.py:content_summarizer", "target_stage": "text_splitter", "run_after": true}' \
     --task='embed' \
     --output_directory=./output
   ```

3. **Use with Python API**
   ```python
   from nv_ingest_client.client.interface import Ingestor

   ingestor = Ingestor()
   results = ingestor.files("./my_pdfs/") \
       .extract() \
       .split() \
       .udf(
           udf_function="./examples/udfs/llm_summarizer_udf.py:content_summarizer",
           target_stage="text_splitter", 
           run_after=True
       ) \
       .embed() \
       .ingest()
   ```

### What You Get

- **Input**: Text chunks from document splitting  
- **Output**: Same chunks enhanced with AI-generated summaries
- **Enhanced Metadata**: Each chunk includes LLM summary, model info, processing timestamp, and content statistics

### Example Output Metadata

```python
{
  "metadata": {
    "content": "Original text content...",
    "custom_content": {
      "llm_summary": {
        "summary": "This section discusses machine learning algorithms and their applications in modern technology, focusing on neural networks and supervised learning methods.",
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "timestamp": "2024-01-01T12:00:00Z",
        "content_length": 1234,
        "summary_length": 156,
        "summarization_method": "llm_api"
      }
    }
  }
}
```

### Advanced Usage

**Combine with Structural Splitting**:
```bash
nv-ingest-cli \
  --doc './markdown_docs/' \
  --task='extract' \
  --task='udf:{"udf_function": "./examples/udfs/structural_split_udf.py:structural_split", "target_stage": "text_splitter", "run_before": true}' \
  --task='split' \
  --task='udf:{"udf_function": "./examples/udfs/llm_summarizer_udf.py:content_summarizer", "target_stage": "text_splitter", "run_after": true}' \
  --task='embed' \
  --output_directory=./output
```

**Configuration Options**:
- `LLM_SUMMARIZATION_MODEL`: Choose different models
- `LLM_MIN_CONTENT_LENGTH`: Skip very short content (default: 100)
- `LLM_MAX_CONTENT_LENGTH`: Limit input size (default: 8000)
- `LLM_SUMMARIZATION_TIMEOUT`: API timeout seconds (default: 60)

**More Examples**: See `example_usage_llm_summarizer.py` for comprehensive usage patterns including batch processing of 20+ PDFs.

## Resources

- **Comprehensive Guide**: [User-Defined Functions](../../docs/docs/extraction/user_defined_functions.md)
- **Pipeline Stages**: [User-Defined Stages](../../docs/docs/extraction/user-defined-stages.md)  
- **Metadata Schema**: [Content Metadata](../../docs/docs/extraction/content-metadata.md)

## Troubleshooting

**UDF not executing?**
- Check function signature matches exactly: `def my_udf(control_message: IngestControlMessage) -> IngestControlMessage`
- Verify file path is accessible in container
- Use `INGEST_DISABLE_UDF_PROCESSING=""` to ensure UDFs are enabled

**LLM Summarization Issues?**
- Verify `NVIDIA_API_KEY` is set and valid
- Check API connectivity: `curl -H "Authorization: Bearer $NVIDIA_API_KEY" https://integrate.api.nvidia.com/v1/models`
- Monitor logs for API timeout or rate limit errors
- Adjust `LLM_SUMMARIZATION_TIMEOUT` for slow API responses
- Use smaller `LLM_MAX_CONTENT_LENGTH` if getting payload errors

**Performance issues?**
- Profile with small document batches first
- Consider running UDF on less congested pipeline stages (try `embedding_storage` instead of `text_splitter`)
- Optimize regex patterns and DataFrame operations
- For LLM UDFs: increase `LLM_MIN_CONTENT_LENGTH` to skip short content
- Monitor API costs and rate limits for large document collections
