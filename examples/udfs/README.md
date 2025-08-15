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

## Resources

- **Comprehensive Guide**: [User-Defined Functions](../../docs/docs/extraction/user_defined_functions.md)
- **Pipeline Stages**: [User-Defined Stages](../../docs/docs/extraction/user-defined-stages.md)  
- **Metadata Schema**: [Content Metadata](../../docs/docs/extraction/content-metadata.md)

## Troubleshooting

**UDF not executing?**
- Check function signature matches exactly: `def my_udf(control_message: IngestControlMessage) -> IngestControlMessage`
- Verify file path is accessible in container
- Use `INGEST_DISABLE_UDF_PROCESSING=""` to ensure UDFs are enabled

**Performance issues?**
- Profile with small document batches first
- Consider running UDF on less congested pipeline stages
- Optimize regex patterns and DataFrame operations
