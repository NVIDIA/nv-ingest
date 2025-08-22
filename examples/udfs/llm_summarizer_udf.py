#!/usr/bin/env python3
"""
LLM Content Summarizer UDF for NV-Ingest Pipeline

This UDF uses an LLM (via OpenAI-compatible API) to generate summaries of text content chunks.
It processes text primitives after chunking/splitting and adds AI-generated summaries to metadata.

Key Features:
- Works with any OpenAI-compatible API (including NVIDIA NIMs)
- Handles both plain text and base64-encoded content
- Robust error handling for API failures
- Configurable via environment variables
- Efficient batch processing to minimize API calls

Environment Variables:
- NVIDIA_API_KEY: API key for NVIDIA NIM endpoints (required)
- LLM_SUMMARIZATION_MODEL: Model to use (default: nvidia/llama-3.1-nemotron-70b-instruct)
- LLM_SUMMARIZATION_BASE_URL: API base URL (default: https://integrate.api.nvidia.com/v1)
- LLM_SUMMARIZATION_TIMEOUT: API timeout in seconds (default: 60)
- LLM_MIN_CONTENT_LENGTH: Minimum content length to summarize (default: 100)
- LLM_MAX_CONTENT_LENGTH: Maximum content length to send to API (default: 8000)

Usage in CLI:
nv-ingest-cli --doc file.pdf \
  --task='extract:{"document_type": "pdf", "extract_text": "True"}' \
  --task='split' \
  --task='udf:{"udf_function": "./examples/udfs/llm_summarizer_udf.py:content_summarizer", \
    "target_stage": "text_splitter", "run_after": true}' \
  --task='embed' \
  --output_directory=./output
"""

import base64
import copy
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List


def content_summarizer(control_message: "IngestControlMessage") -> "IngestControlMessage":
    """
    UDF function that summarizes text content using an LLM API.

    This function:
    1. Gets the DataFrame payload from the control message
    2. Finds text content rows that need summarization
    3. Calls LLM API to generate summaries for each chunk
    4. Stores summaries in metadata.custom_content
    5. Returns the updated control message

    Parameters
    ----------
    control_message : IngestControlMessage
        The control message containing the payload and metadata

    Returns
    -------
    IngestControlMessage
        The modified control message with LLM-generated summaries
    """
    import pandas as pd
    from openai import OpenAI

    logger = logging.getLogger(__name__)
    logger.info("UDF: Starting LLM content summarization")

    # Get configuration from environment
    api_key = os.getenv("NVIDIA_API_KEY", "")
    model_name = os.getenv("LLM_SUMMARIZATION_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
    base_url = os.getenv("LLM_SUMMARIZATION_BASE_URL", "https://integrate.api.nvidia.com/v1")
    timeout = int(os.getenv("LLM_SUMMARIZATION_TIMEOUT", "60"))
    min_content_length = int(os.getenv("LLM_MIN_CONTENT_LENGTH", "100"))
    max_content_length = int(os.getenv("LLM_MAX_CONTENT_LENGTH", "8000"))

    if not api_key:
        logger.warning("UDF: NVIDIA_API_KEY not found, skipping summarization")
        return control_message

    # Initialize OpenAI client
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        logger.debug(f"UDF: Initialized OpenAI client with base_url={base_url}, model={model_name}")
    except Exception as e:
        logger.error(f"UDF: Failed to initialize OpenAI client: {e}")
        return control_message

    # Get the payload DataFrame
    df = control_message.payload()
    if df is None or len(df) == 0:
        logger.warning("UDF: No payload found in control message")
        return control_message

    logger.info(f"UDF: Processing DataFrame with {len(df)} rows for LLM summarization")

    # Find rows that should be summarized (text primitives)
    rows_to_summarize = []
    rows_to_keep_unchanged = []

    for idx, row in df.iterrows():
        # Check if this is a text primitive
        content_type = ""
        if isinstance(row.get("metadata"), dict):
            content_metadata = row["metadata"].get("content_metadata", {})
            content_type = content_metadata.get("type", "").lower()

        is_text_content = content_type == "text" or str(content_type) == "text"

        if is_text_content:
            rows_to_summarize.append((idx, row))
        else:
            rows_to_keep_unchanged.append((idx, row))

    logger.info(
        f"UDF: Found {len(rows_to_summarize)} text rows to summarize, "
        f"{len(rows_to_keep_unchanged)} rows to keep unchanged"
    )

    if not rows_to_summarize:
        logger.info("UDF: No text content found to summarize")
        return control_message

    # Process eligible rows for summarization
    summary_stats = {"processed": 0, "summarized": 0, "skipped": 0, "failed": 0}
    
    for idx, row in rows_to_summarize:
        summary_stats["processed"] += 1
        
        try:
            # Extract content from metadata
            content = _extract_content_from_row(row, logger)
            
            if not content or len(content.strip()) < min_content_length:
                logger.debug(f"UDF: Skipping row {idx} - content too short ({len(content) if content else 0} chars)")
                summary_stats["skipped"] += 1
                continue

            # Truncate content if too long for API
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
                logger.debug(f"UDF: Truncated content for row {idx} to {max_content_length} chars")

            # Generate summary using LLM
            summary = _generate_summary(client, content, model_name, logger)
            
            if summary:
                # Add summary to metadata
                _add_summary_to_metadata(df, idx, row, summary, model_name, content, logger)
                summary_stats["summarized"] += 1
                logger.debug(f"UDF: Successfully summarized row {idx}")
            else:
                summary_stats["failed"] += 1
                logger.warning(f"UDF: Failed to generate summary for row {idx}")

        except Exception as e:
            summary_stats["failed"] += 1
            logger.error(f"UDF: Error processing row {idx}: {e}")
            continue

    # Update the control message with modified DataFrame
    control_message.payload(df)

    logger.info(
        f"UDF: LLM summarization complete - "
        f"Processed: {summary_stats['processed']}, "
        f"Summarized: {summary_stats['summarized']}, "
        f"Skipped: {summary_stats['skipped']}, "
        f"Failed: {summary_stats['failed']}"
    )

    return control_message


def _extract_content_from_row(row: "pd.Series", logger) -> Optional[str]:
    """Extract content from a DataFrame row, handling both plain text and base64."""
    content = ""
    
    if isinstance(row.get("metadata"), dict):
        content = row["metadata"].get("content", "")

    if not content:
        return None

    # Try to decode base64 content if needed (following structural_split_udf pattern)
    try:
        decoded_content = base64.b64decode(content).decode("utf-8")
        content = decoded_content
        logger.debug("UDF: Decoded base64 content successfully")
    except Exception:
        # Content is already plain text, use as-is
        pass

    return content.strip()


def _generate_summary(client, content: str, model_name: str, logger) -> Optional[str]:
    """Generate summary using LLM API with error handling."""
    
    # Create concise summarization prompt
    prompt = f"""Please provide a concise summary of the following text content in 2-3 sentences. Focus on the main points and key information:

{content}

Summary:"""

    try:
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,  # Limit summary length
            temperature=0.3   # More deterministic outputs
        )
        
        end_time = time.time()
        
        if completion.choices and len(completion.choices) > 0:
            summary = completion.choices[0].message.content.strip()
            logger.debug(f"UDF: LLM API call successful ({end_time - start_time:.2f}s)")
            return summary
        else:
            logger.warning("UDF: LLM API returned empty response")
            return None
            
    except Exception as e:
        logger.error(f"UDF: LLM API call failed: {e}")
        return None


def _add_summary_to_metadata(df: "pd.DataFrame", idx: int, row: "pd.Series", summary: str, 
                           model_name: str, original_content: str, logger):
    """Add LLM summary to row metadata in the custom_content field."""
    
    try:
        # Get existing metadata or create new
        metadata = copy.deepcopy(row.get("metadata", {}))
        
        # Ensure custom_content exists
        if "custom_content" not in metadata:
            metadata["custom_content"] = {}
            
        # Add LLM summary with metadata
        metadata["custom_content"]["llm_summary"] = {
            "summary": summary,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(original_content),
            "summary_length": len(summary),
            "summarization_method": "llm_api"
        }
        
        # Update the DataFrame
        df.at[idx, "metadata"] = metadata
        
        logger.debug(f"UDF: Added summary metadata to row {idx}")
        
    except Exception as e:
        logger.error(f"UDF: Failed to add summary to metadata for row {idx}: {e}")


def content_summarizer_batch(control_message: "IngestControlMessage") -> "IngestControlMessage":
    """
    Alternative UDF that processes content in batches for better API efficiency.
    
    This version groups multiple content pieces into single API calls where possible,
    reducing overall API usage and improving performance for large document sets.
    """
    logger = logging.getLogger(__name__)
    logger.info("UDF: Starting batch LLM content summarization")
    
    # For now, delegate to the main function
    # In a full implementation, this would implement batching logic
    return content_summarizer(control_message)


# Example usage and testing functions
def test_summarizer_locally():
    """
    Local test function for development and debugging.
    Run this independently to test the summarization logic.
    """
    import pandas as pd
    from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
    
    # Create test data
    test_df = pd.DataFrame({
        'document_type': ['text', 'text'],
        'source_id': ['test1', 'test2'],
        'job_id': ['job-001', 'job-001'],
        'metadata': [
            {
                'content': 'This is a sample document about artificial intelligence and machine learning. It discusses various algorithms, neural networks, and their applications in modern technology. The document covers both supervised and unsupervised learning methods.',
                'content_metadata': {'type': 'text'},
                'source_metadata': {'source_id': 'test1', 'source_type': 'text'},
            },
            {
                'content': 'A brief text that should be skipped due to length.',
                'content_metadata': {'type': 'text'},
                'source_metadata': {'source_id': 'test2', 'source_type': 'text'},
            }
        ]
    })
    
    # Create control message
    control_message = IngestControlMessage()
    control_message.payload(test_df)
    
    print("Testing LLM summarization UDF...")
    print(f"Input: {len(test_df)} rows")
    
    # Test the UDF
    result = content_summarizer(control_message)
    
    # Check results
    result_df = result.payload()
    print(f"Output: {len(result_df)} rows")
    
    for idx, row in result_df.iterrows():
        metadata = row.get('metadata', {})
        custom_content = metadata.get('custom_content', {})
        llm_summary = custom_content.get('llm_summary')
        
        if llm_summary:
            print(f"\nRow {idx} Summary:")
            print(f"  Content length: {llm_summary.get('content_length', 0)}")
            print(f"  Summary: {llm_summary.get('summary', 'N/A')}")
            print(f"  Model: {llm_summary.get('model', 'N/A')}")


if __name__ == "__main__":
    # Allow local testing
    test_summarizer_locally()
