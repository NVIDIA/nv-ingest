#!/usr/bin/env python3
"""
Example Usage Scripts for LLM Summarizer UDF

This file demonstrates how to use the LLM content summarizer UDF with both
the Python client interface and provides CLI command examples.
"""

import os
from nv_ingest_client.client.interface import Ingestor


def example_python_api_usage():
    """
    Example: Using the LLM summarizer UDF with the Python client API
    
    This example shows how to:
    1. Ingest PDF documents
    2. Extract content and split into chunks
    3. Generate LLM summaries for text chunks
    4. Generate embeddings for the summarized content
    5. Store results
    """
    # Set up environment variables (you would do this in your environment)
    os.environ.setdefault("NVIDIA_API_KEY", "your-nvidia-api-key-here")
    os.environ.setdefault("LLM_SUMMARIZATION_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
    
    print("Setting up NV-Ingest client with LLM summarization...")
    
    # Create an Ingestor instance
    ingestor = Ingestor()
    
    # Example 1: Basic usage with LLM summarization after text splitting
    print("Example 1: LLM summarization after text splitting")
    results = ingestor.files("./sample_pdfs/") \
        .extract() \
        .split() \
        .udf(
            udf_function="./examples/udfs/llm_summarizer_udf.py:content_summarizer",
            target_stage="text_splitter",
            run_after=True
        ) \
        .embed() \
        .ingest()
    
    print(f"Processed {len(results)} documents with LLM summaries")
    
    # Example 2: Multiple UDFs - structural splitting + LLM summarization
    print("\nExample 2: Combining structural splitting with LLM summarization")
    results_advanced = ingestor.files("./markdown_docs/") \
        .extract() \
        .udf(
            udf_function="./examples/udfs/structural_split_udf.py:structural_split",
            target_stage="text_splitter",
            run_before=True
        ) \
        .split() \
        .udf(
            udf_function="./examples/udfs/llm_summarizer_udf.py:content_summarizer",
            target_stage="text_splitter", 
            run_after=True
        ) \
        .embed() \
        .ingest()
    
    print(f"Processed {len(results_advanced)} documents with structural splitting + LLM summaries")
    
    # Example 3: LLM summarization just before embedding (alternative approach)
    print("\nExample 3: LLM summarization before embedding generation")
    results_pre_embed = ingestor.files("./reports/") \
        .extract() \
        .split() \
        .udf(
            udf_function="./examples/udfs/llm_summarizer_udf.py:content_summarizer",
            target_stage="text_embedder",
            run_before=True
        ) \
        .embed() \
        .ingest()
    
    print(f"Processed {len(results_pre_embed)} documents with pre-embedding summarization")
    
    return results, results_advanced, results_pre_embed


def example_inspect_results(results):
    """
    Example: How to inspect the results and access LLM summaries
    """
    print("Inspecting LLM summarization results...")
    
    for doc in results:
        print(f"\nDocument: {doc.get('source_name', 'Unknown')}")
        
        # Access the metadata
        metadata = doc.get('metadata', {})
        custom_content = metadata.get('custom_content', {})
        llm_summary = custom_content.get('llm_summary', {})
        
        if llm_summary:
            print(f"  ✅ Has LLM Summary:")
            print(f"     Summary: {llm_summary.get('summary', '')[:100]}...")
            print(f"     Model: {llm_summary.get('model', 'N/A')}")
            print(f"     Content Length: {llm_summary.get('content_length', 0)} chars")
            print(f"     Summary Length: {llm_summary.get('summary_length', 0)} chars")
            print(f"     Generated: {llm_summary.get('timestamp', 'N/A')}")
        else:
            print("  ❌ No LLM summary found")
            
        # Show original content snippet
        original_content = metadata.get('content', '')
        if original_content:
            print(f"  Original content: {original_content[:150]}...")


def print_cli_examples():
    """
    Print CLI command examples for different use cases
    """
    
    print("\n" + "="*80)
    print("CLI COMMAND EXAMPLES")
    print("="*80)
    
    print("\n1. BASIC LLM SUMMARIZATION")
    print("-" * 50)
    print("""
# Basic PDF processing with LLM summarization after text splitting
export NVIDIA_API_KEY="your-api-key-here"
export LLM_SUMMARIZATION_MODEL="nvidia/llama-3.1-nemotron-70b-instruct"

nv-ingest-cli \\
  --doc ./sample_documents/ \\
  --task='extract:{"document_type": "pdf", "extract_text": "True"}' \\
  --task='split:{"chunk_size": 512, "chunk_overlap": 20}' \\
  --task='udf:{"udf_function": "./examples/udfs/llm_summarizer_udf.py:content_summarizer", \\
    "target_stage": "text_splitter", "run_after": true}' \\
  --task='embed' \\
  --output_directory=./output_with_summaries
""")
    
    print("\n2. COMBINED STRUCTURAL + LLM PROCESSING")
    print("-" * 50)
    print("""
# Combine structural splitting with LLM summarization for markdown docs
nv-ingest-cli \\
  --doc ./markdown_docs/ \\
  --task='extract' \\
  --task='udf:{"udf_function": "./examples/udfs/structural_split_udf.py:structural_split", \\
    "target_stage": "text_splitter", "run_before": true}' \\
  --task='split' \\
  --task='udf:{"udf_function": "./examples/udfs/llm_summarizer_udf.py:content_summarizer", \\
    "target_stage": "text_splitter", "run_after": true}' \\
  --task='embed' \\
  --output_directory=./output_advanced
""")
    
    print("\n3. PRE-EMBEDDING SUMMARIZATION")
    print("-" * 50)
    print("""
# Run LLM summarization just before embedding generation
nv-ingest-cli \\
  --doc ./research_papers/ \\
  --task='extract' \\
  --task='split:{"chunk_size": 1024}' \\
  --task='udf:{"udf_function": "./examples/udfs/llm_summarizer_udf.py:content_summarizer", \\
    "target_stage": "text_embedder", "run_before": true}' \\
  --task='embed' \\
  --output_directory=./output_pre_embed
""")
    
    print("\n4. CUSTOM CONFIGURATION")
    print("-" * 50)
    print("""
# Custom model and parameters for specialized use cases
export NVIDIA_API_KEY="your-api-key-here"
export LLM_SUMMARIZATION_MODEL="nvidia/llama-3.1-nemotron-70b-instruct"
export LLM_MIN_CONTENT_LENGTH="200"
export LLM_MAX_CONTENT_LENGTH="10000"
export LLM_SUMMARIZATION_TIMEOUT="90"

nv-ingest-cli \\
  --doc ./large_documents/ \\
  --task='extract' \\
  --task='split:{"chunk_size": 2048, "chunk_overlap": 100}' \\
  --task='udf:{"udf_function": "./examples/udfs/llm_summarizer_udf.py:content_summarizer", \\
    "target_stage": "text_splitter", "run_after": true}' \\
  --task='embed' \\
  --output_directory=./output_custom
""")

    print("\n5. BATCH PROCESSING (20 PDFs)")
    print("-" * 50)
    print("""
# Process multiple PDFs efficiently
export NVIDIA_API_KEY="your-api-key-here"

nv-ingest-cli \\
  --doc ./pdf_collection/*.pdf \\
  --task='extract:{"document_type": "pdf"}' \\
  --task='split:{"chunk_size": 800, "chunk_overlap": 50}' \\
  --task='udf:{"udf_function": "./examples/udfs/llm_summarizer_udf.py:content_summarizer", \\
    "target_stage": "text_splitter", "run_after": true}' \\
  --task='embed' \\
  --output_directory=./batch_output \\
  --log_level=INFO
""")

    print("\n" + "="*80)
    print("ENVIRONMENT VARIABLES REFERENCE")
    print("="*80)
    print("""
Required:
  NVIDIA_API_KEY                 - Your NVIDIA API key for LLM access

Optional:
  LLM_SUMMARIZATION_MODEL        - Model name (default: nvidia/llama-3.1-nemotron-70b-instruct)
  LLM_SUMMARIZATION_BASE_URL     - API base URL (default: https://integrate.api.nvidia.com/v1)
  LLM_SUMMARIZATION_TIMEOUT      - API timeout in seconds (default: 60)
  LLM_MIN_CONTENT_LENGTH         - Min chars to summarize (default: 100)
  LLM_MAX_CONTENT_LENGTH         - Max chars to send to API (default: 8000)
""")

    print("\n" + "="*80)
    print("EXPECTED OUTPUT STRUCTURE")
    print("="*80)
    print("""
Each text chunk will have enhanced metadata:

{
  "metadata": {
    "content": "Original text content...",
    "custom_content": {
      "llm_summary": {
        "summary": "AI-generated summary of the content...",
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "timestamp": "2024-01-01T12:00:00Z",
        "content_length": 1234,
        "summary_length": 156,
        "summarization_method": "llm_api"
      }
    },
    // ... other metadata fields
  }
}
""")


def performance_tips():
    """
    Print performance optimization tips for large-scale processing
    """
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("="*80)
    
    print("""
1. CONTENT FILTERING:
   - Adjust LLM_MIN_CONTENT_LENGTH to skip very short chunks
   - Set LLM_MAX_CONTENT_LENGTH to limit API payload size
   - Filter document types before processing

2. API OPTIMIZATION:
   - Use reasonable timeout values (60-90 seconds)
   - Monitor API rate limits and adjust batch sizes
   - Consider using batch processing UDF variant for very large jobs

3. PIPELINE OPTIMIZATION:
   - Run LLM UDF after text_splitter to process optimally-sized chunks
   - Consider running before text_embedder if summaries should influence embeddings
   - Use appropriate chunk sizes (512-1024 chars work well)

4. ERROR HANDLING:
   - Monitor logs for API failures
   - Failed summaries don't stop the entire pipeline
   - Set INGEST_DISABLE_UDF_PROCESSING=1 to temporarily disable if needed

5. RESOURCE MANAGEMENT:
   - Consider API costs for large document collections
   - Monitor memory usage during processing
   - Use appropriate replicas configuration for your workload
""")


if __name__ == "__main__":
    print("LLM Summarizer UDF - Usage Examples")
    print("=" * 50)
    
    # Show CLI examples
    print_cli_examples()
    
    # Show performance tips
    performance_tips()
    
    print("\n" + "="*80)
    print("To run Python API examples, uncomment the following lines:")
    print("="*80)
    print("# results, results_advanced, results_pre_embed = example_python_api_usage()")
    print("# example_inspect_results(results)")
    
    # Uncomment these to run the Python examples:
    # results, results_advanced, results_pre_embed = example_python_api_usage()
    # example_inspect_results(results)
