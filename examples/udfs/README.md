# NV-Ingest UDF Examples

User-Defined Functions (UDFs) let you inject custom processing logic into the NV-Ingest pipeline at specific stages. This directory demonstrates different UDF organizational patterns:

## 📁 UDF Organization

### Examples (`examples/udfs/`) - Learning & Reference
- **Purpose**: Core examples for learning UDF development patterns
- **Audience**: Developers learning to write UDFs
- **Characteristics**: Self-contained, educational
- **Example**: `structural_split_udf.py` - demonstrates text processing patterns for specific files

### API UDFs (`api/src/udfs/`) - Production-Ready Components  
- **Purpose**: Reusable UDFs that are part of the API library
- **Audience**: Production deployments, advanced users
- **Characteristics**: Robust error handling, configurable, tested for production use
- **Example**: `llm_summarizer_udf.py` - production-grade AI summarization

> **💡 Placement Guide**: Put learning examples in `examples/udfs/`, put reusable production UDFs in `api/src/udfs/`

## Markdown Document Splitter Example

**Problem**: You have **markdown documents** with hierarchical structure (headers like `#`, `##`, `###`) that you want to ingest. The default text splitter doesn't preserve this structure, losing important document organization.

**Solution**: The `structural_split_udf.py` splits **markdown documents** at header boundaries (`#`, `##`, `###`, etc.) while preserving hierarchical metadata.

> **Works with**: Native markdown files (`.md`), text files containing markdown syntax
> 
> **Does not work with**: PDFs, Word docs, HTML files, or other formats that don't use markdown header syntax (`#`, `##`, `###`). These documents contain plain text without markdown formatting after extraction.

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
   # For markdown files (.md) - process directly as text
   nv-ingest-cli \
     --doc './my_markdown_docs/' \
     --task='extract:{"document_type":"text", "extract_text":true}' \
     --task='udf:{"udf_function": "./examples/udfs/structural_split_udf.py:structural_split", "target_stage": "text_splitter", "run_before": true}' \
     --task='embed' \
     --output_directory=./output
   ```

3. **Use with Python API**
   ```python
   from nv_ingest_client.client.interface import Ingestor

   # For markdown files (.md)
   ingestor = Ingestor()
   results = ingestor.files("./my_markdown_docs/") \
       .extract(document_type="text", extract_text=True) \
       .udf(
           udf_function="./examples/udfs/structural_split_udf.py:structural_split",
           target_stage="text_splitter", 
           run_before=True
       ) \
       .embed() \
       .ingest()
   ```

### Verified Working Examples

> **✅ Tested with**: Native markdown files (`.md`) containing markdown header syntax

**CLI Command (Tested):**
```bash
# Test with the sample markdown file
nv-ingest-cli \
  --doc data/multimodal_test.md \
  --task='extract:{"document_type":"text", "extract_text":true}' \
  --task='udf:{"udf_function": "./examples/udfs/structural_split_udf.py:structural_split", "target_stage": "text_splitter", "run_before": true}' \
  --output_directory=./test_output_structural_udf

# Results: 12 chunks created from 1 markdown file with proper hierarchical metadata
```

**Python API Usage:**
```python
from nv_ingest_client.client.interface import Ingestor

# Works with markdown files containing header syntax
ingestor = Ingestor()
results = ingestor.files("data/multimodal_test.md") \
    .extract(document_type="text", extract_text=True) \
    .udf(
        udf_function="./examples/udfs/structural_split_udf.py:structural_split",
        target_stage="text_splitter",
        run_before=True
    ) \
    .ingest()

# Python API returns nested structure: [[chunks...]]
chunks = results[0]  # Access the actual chunks
print(f"Created {len(chunks)} chunks from 1 markdown document")
```

**Expected Output:**
- Input: 1 markdown document  
- Output: 12 markdown sections split by headers (`#`, `##`, `###`, etc.)
- Enhanced metadata includes: `hierarchical_header`, `header_level`, `chunk_index`, `splitting_method`
- **Note**: Python API results are nested - access chunks with `results[0]`
- **Splitting Logic**: Each markdown header creates a new document chunk with that header's content

### What You Get

- **Input**: 1 document with markdown content  
- **Output**: N documents split at **markdown header boundaries**
- **Enhanced Metadata**: Each chunk includes markdown-specific hierarchical information like header level, chunk index, and splitting method

### Implementation Details

The UDF processes **text primitives** by:
1. Filtering for text primitives with `document_type: "text"` (regardless of original file type)
2. Extracting content from document metadata
3. **Parsing markdown syntax** and splitting at header boundaries (`#`, `##`, `###`, etc.)
4. Creating new document rows for each markdown section with enriched metadata
5. Returning the updated DataFrame with all markdown chunks


### Customization

Adapt the markdown splitter for your needs:
```python
# Use different markdown header levels
markdown_headers = ["#", "##"]  # Only split on major headers

# Add alternative markdown syntax
markdown_headers = ["#", "##", "===", "---"]  # Include setext headers

# Add custom metadata for markdown sections
metadata["custom_content"]["document_category"] = detect_category(content)
metadata["custom_content"]["markdown_variant"] = "github_flavored"
```



## LLM Content Summarizer (Production Example)

**Purpose**: Generate AI-powered summaries of your text content using NVIDIA's LLM APIs. This demonstrates how to structure UDFs for production use with robust error handling, configuration, and testing.

### Setup

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"
```

### Usage

**CLI Example:**
```bash
nv-ingest-cli \
  --doc 'my_documents/' \
  --task='extract:{"document_type":"pdf", "extract_text":true}' \
  --task='split' \
  --task='udf:{"udf_function": "./api/src/udfs/llm_summarizer_udf.py:content_summarizer", "target_stage": "text_splitter", "run_after": true}' \
  --output_directory=./output
```

**Python API Example:**
```python
from nv_ingest_client.client.interface import Ingestor

ingestor = Ingestor()
results = ingestor.files("my_documents/") \
    .extract(document_type="pdf", extract_text=True) \
    .split() \
    .udf(
        udf_function="./api/src/udfs/llm_summarizer_udf.py:content_summarizer",
        target_stage="text_splitter", 
        run_after=True
    ) \
    .ingest()

# Access chunks: results[0] for Python API
chunks = results[0]
```

### Viewing Outputs

After processing, summaries are stored in the output metadata files. Look for the `llm_summary` section:

**CLI Output Location:**
```bash
./output/text/your_document.pdf.metadata.json
```

**Finding Summaries in JSON:**
```json
{
  "metadata": {
    "custom_content": {
      "llm_summary": {
        "summary": "Your AI-generated summary appears here...",
        "model": "nvidia/llama-3.1-nemotron-70b-instruct"
      }
    }
  }
}
```

**Python API Access:**
```python
# Access summaries from results
chunks = results[0]  # Get document chunks
for chunk in chunks:
    summary_info = chunk["metadata"]["custom_content"]["llm_summary"]
    summary_text = summary_info["summary"]
    print(f"Summary: {summary_text}")
```

### Configuration

Environment variables to customize behavior:

```bash
export NVIDIA_API_KEY="your-nvidia-api-key"                              # Required API key
export LLM_SUMMARIZATION_MODEL="nvidia/llama-3.1-nemotron-70b-instruct"  # Model choice
export LLM_SUMMARIZATION_BASE_URL="https://integrate.api.nvidia.com/v1"  # API base URL
export LLM_SUMMARIZATION_TIMEOUT="60"                                    # API timeout seconds
export LLM_MIN_CONTENT_LENGTH="50"                                       # Skip content shorter than this  
export LLM_MAX_CONTENT_LENGTH="12000"                                    # Content limit per summary
```

### Testing

UDF tests are organized by their purpose:

**Production UDF Tests** (in `api/api_tests/udfs/`):
```bash
# Test production UDFs that are part of the API
pytest api/api_tests/udfs/test_llm_summarizer_udf.py -v
```

**Example UDF Tests** (in `examples/tests/`):
```bash
# Test example UDFs (may not be available in all installations)
pytest examples/tests/test_structural_split_udf.py -v
```

**Run All UDF Tests:**
```bash
# Test both production and example UDFs
pytest api/api_tests/udfs/ examples/tests/ -v
```

The test organization demonstrates proper separation: production UDF tests in `api/` ensure they work in all deployments, while example UDF tests in `examples/` acknowledge they may not be available everywhere.

## Resources

- **Comprehensive Guide**: [User-Defined Functions](../../docs/docs/extraction/user_defined_functions.md)
- **Pipeline Stages**: [User-Defined Stages](../../docs/docs/extraction/user-defined-stages.md)  
- **Metadata Schema**: [Content Metadata](../../docs/docs/extraction/content-metadata.md)

## UDF Development Guidelines

**For Learning & Experimentation** → `examples/udfs/`
- Simple, focused examples
- Self-contained functionality  
- Educational comments and documentation

**For Production Use** → `api/src/udfs/`
- Robust error handling and logging
- Configurable via environment variables
- Comprehensive unit tests in `api/api_tests/udfs/`
- Can be imported as part of the API library

## Common Issues

**UDF not running?**
- Verify `NVIDIA_API_KEY` is set
- Check UDF file path is accessible
- Ensure function signature: `def my_udf(control_message: IngestControlMessage) -> IngestControlMessage`

**No summaries in output?**
- Check logs for API errors
- Verify documents have text content > 50 characters  
- Use `document_type="pdf"` for PDF text extraction (creates text primitives for the UDF)
- For markdown/text files, use `document_type="text"`

**Python API access:**
- Results are nested: access chunks via `results[0]`
- CLI returns chunks directly, Python API wraps in document array
