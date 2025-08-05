# Structural Text Splitter UDF Guide

Your structural text splitter is now ready to use as a **User-Defined Function (UDF)** in the NV-Ingest CLI pipeline!

## 🚀 Quick Start

### Basic Usage (Text Files)
```bash
nv-ingest-cli \
  --batch_size=64 \
  --doc './data/multimodal_test.md' \
  --client_host=localhost \
  --output_directory=../datasets/processed/udf_testing \
  --task='extract:{"document_type": "text", "extract_text": "True"}' \
  --task='udf:{"udf_function": "./data/structural_split_udf.py:structural_split", "phase": "transform"}' \
  --client_port=7670
```

### Full Pipeline (Your Exact Pattern)
```bash
nv-ingest-cli \
  --batch_size=64 \
  --doc './data/multimodal_test.pdf' \
  --client_host=localhost \
  --output_directory=../datasets/processed/udf_testing \
  --task='extract:{"document_type": "pdf", "extract_tables": "True", "extract_charts": "True", "extract_images": "True", "extract_text": "True"}' \
  --task='dedup' \
  --task='filter' \
  --task='split' \
  --task='udf:{"udf_function": "./data/structural_split_udf.py:structural_split", "phase": "transform"}' \
  --task='embed' \
  --client_port=7670
```

## 📁 Files Created

1. **`data/structural_split_udf.py`** - The main UDF implementation
2. **`test_structural_splitter_cli.sh`** - Test script for text files  
3. **`test_structural_splitter_pdf_cli.sh`** - Test script for PDF files
4. **`data/multimodal_test.md`** - Sample test document

## 🔧 UDF Functions Available

### `structural_split`
**Default configuration** - splits on all header levels (#, ##, ###, ####, #####, ######)
```bash
--task='udf:{"udf_function": "./data/structural_split_udf.py:structural_split", "phase": "transform"}'
```

### `structural_split_custom_headers`  
**Coarser splitting** - only splits on major headers (#, ##) for larger chunks
```bash
--task='udf:{"udf_function": "./data/structural_split_udf.py:structural_split_custom_headers", "phase": "transform"}'
```

## ⚙️ Configuration Options

The UDF is pre-configured with sensible defaults, but you can modify `data/structural_split_udf.py` to customize:

```python
config = StructuralTextSplitterSchema(
    markdown_headers_to_split_on=["#", "##", "###", "####"],  # Which headers to split on
    max_chunk_size_tokens=1024,                               # Max tokens per chunk
    preserve_headers_in_chunks=True,                          # Include headers in content
    min_chunk_size_chars=50                                   # Minimum chunk size
)
```

## 🔄 Pipeline Phases

Your structural splitter UDF should typically run in the **"transform"** phase:

- **"extraction"** - After text extraction but before other processing
- **"transform"** - ✅ **Recommended** - After basic splitting but before embedding
- **"response"** - Final processing before output

## 🧪 Testing Commands

### Quick Test (Dry Run)
```bash
nv-ingest-cli --doc './data/multimodal_test.md' \
  --task='extract:{"document_type": "text", "extract_text": "True"}' \
  --task='udf:{"udf_function": "./data/structural_split_udf.py:structural_split", "phase": "transform"}' \
  --output_directory=../datasets/processed/udf_testing \
  --dry_run
```

### Run Test Scripts
```bash
# Test with text/markdown files
./test_structural_splitter_cli.sh

# Test with PDF files (requires multimodal_test.pdf in data/ directory)
./test_structural_splitter_pdf_cli.sh
```

## 📊 Expected Results

When your structural splitter UDF runs, you should see:

1. **Log messages** indicating structural splitting started and completed
2. **Increased document count** if markdown headers were found and split
3. **Rich metadata** in output JSON files with hierarchical information:
   - `hierarchical_header`: The header text (e.g., "## Introduction")
   - `header_level`: Numeric level (1, 2, 3, etc.)
   - `splitting_method`: "structural_markdown"
   - `chunk_index` and `total_chunks`: Position information

## 🔍 Debugging

If the UDF isn't working as expected:

1. **Check logs** for "Starting structural text splitting UDF" messages
2. **Verify phase timing** - transform phase runs after text extraction
3. **Check document types** - UDF only processes documents with `source_type: "text"`
4. **Look for markdown headers** - no headers = no splitting

## 🎯 Integration with Your Pipeline

Your structural splitter UDF fits perfectly into the existing pipeline pattern:

```bash
# Your pattern:
--task='udf:{"udf_function": "./data/random_udf.py:add_random_metadata", "phase": "response"}' \

# Structural splitter pattern:
--task='udf:{"udf_function": "./data/structural_split_udf.py:structural_split", "phase": "transform"}' \
```

This allows you to:
- ✅ Use it alongside other UDFs
- ✅ Control when it runs in the pipeline
- ✅ Customize the splitting behavior
- ✅ Integrate with existing workflows

## 🎉 Success!

Your structural text splitter is now fully integrated as a UDF and ready for production use! It follows the exact CLI pattern you specified and can be used alongside your other UDF functions. 