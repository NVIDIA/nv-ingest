#!/usr/bin/env python3
"""
LLM Content Summarizer UDF for NV-Ingest Pipeline

This UDF uses an LLM API to generate concise summaries
of text content chunks, adding AI-generated summaries to the metadata for
enhanced downstream processing and search capabilities.

Environment Variables:
- NVIDIA_API_KEY: API key for NVIDIA NIM endpoints (required)
- LLM_SUMMARIZATION_MODEL: Model to use (default: nvidia/llama-3.1-nemotron-70b-instruct)
- LLM_SUMMARIZATION_BASE_URL: API base URL (default: https://integrate.api.nvidia.com/v1)
- LLM_SUMMARIZATION_TIMEOUT: API timeout in seconds (default: 60)
- LLM_MIN_CONTENT_LENGTH: Minimum content length to summarize (default: 50)
- LLM_MAX_CONTENT_LENGTH: Maximum content length to send to API (default: 12000)
"""

import os
import logging
from typing import Optional


def content_summarizer(control_message: "IngestControlMessage") -> "IngestControlMessage":  # noqa: F821
    """
    UDF function that adds LLM-generated summaries to text content chunks.

    This function processes text primitives and generates concise summaries using
    an LLM API, storing the results in the metadata's custom_content field.

    Features:
    - Flexible content detection across multiple metadata locations
    - Robust error handling with graceful fallbacks
    - Comprehensive logging for monitoring and debugging
    - Configurable content length thresholds
    - Safe metadata manipulation preserving existing data

    Parameters
    ----------
    control_message : IngestControlMessage
        The control message containing the DataFrame payload with text content

    Returns
    -------
    IngestControlMessage
        The modified control message with LLM summaries added to metadata
    """
    from openai import OpenAI

    logger = logging.getLogger(__name__)
    logger.info("UDF: Starting LLM content summarization")

    # Get configuration from environment
    api_key = os.getenv("NVIDIA_API_KEY", "")
    model_name = os.getenv("LLM_SUMMARIZATION_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
    base_url = os.getenv("LLM_SUMMARIZATION_BASE_URL", "https://integrate.api.nvidia.com/v1")
    timeout = int(os.getenv("LLM_SUMMARIZATION_TIMEOUT", "60"))
    min_content_length = int(os.getenv("LLM_MIN_CONTENT_LENGTH", "50"))
    max_content_length = int(os.getenv("LLM_MAX_CONTENT_LENGTH", "12000"))

    if not api_key:
        logger.warning("NVIDIA_API_KEY not found, skipping summarization")
        return control_message

    # Get the DataFrame payload
    df = control_message.payload()
    if df is None or len(df) == 0:
        logger.warning("No payload found in control message")
        return control_message

    logger.info(f"Processing {len(df)} rows for LLM summarization")

    # Initialize OpenAI client with error handling
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return control_message

    # Stats for reporting
    stats = {"processed": 0, "summarized": 0, "skipped": 0, "failed": 0}

    # Process each row
    for idx, row in df.iterrows():
        stats["processed"] += 1

        try:
            # Extract content - be more flexible about where it comes from
            content = _extract_content(row, logger)

            if not content:
                stats["skipped"] += 1
                continue

            content = content.strip()
            if len(content) < min_content_length:
                stats["skipped"] += 1
                continue

            # Truncate if needed
            if len(content) > max_content_length:
                content = content[:max_content_length]

            # Generate summary
            summary = _generate_summary(client, content, model_name, logger)

            if summary:
                # Add to metadata
                _add_summary(df, idx, row, summary, model_name, logger)
                stats["summarized"] += 1
            else:
                stats["failed"] += 1

        except Exception as e:
            stats["failed"] += 1
            logger.error(f"Row {idx}: Error processing content: {e}")

    # Update the control message with modified DataFrame
    control_message.payload(df)

    logger.info(
        f"LLM summarization complete: {stats['summarized']}/{stats['processed']} documents summarized, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )

    return control_message


def _extract_content(row, logger) -> Optional[str]:
    """Extract text content from row, trying multiple locations."""
    content = ""

    # Try different locations for content
    if isinstance(row.get("metadata"), dict):
        metadata = row["metadata"]

        # Primary location: metadata.content
        content = metadata.get("content", "")

        # If no content, try other locations
        if not content:
            # Try in text_metadata
            text_metadata = metadata.get("text_metadata", {})
            content = text_metadata.get("text", "") or text_metadata.get("content", "")

    # Try top-level content field
    if not content:
        content = row.get("content", "")

    if not content:
        return None

    return content


def _generate_summary(client, content: str, model_name: str, logger) -> Optional[str]:
    """Generate summary with robust error handling."""
    prompt = f"""Please provide a comprehensive 3-4 sentence summary of the following document:

{content}

Focus on the main purpose, key topics, and important details.
This summary will be used for document search and understanding.

Summary:"""

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,  # Increased for more comprehensive summaries
            temperature=0.7,
        )

        if completion.choices and len(completion.choices) > 0:
            summary = completion.choices[0].message.content.strip()
            return summary
        else:
            return None

    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None


def _add_summary(df, idx: int, row, summary: str, model_name: str, logger):
    """Add summary to metadata with safe handling."""
    try:
        # Get current metadata or create new dict - handle None case properly
        existing_metadata = row.get("metadata")
        if existing_metadata is not None and isinstance(existing_metadata, dict):
            metadata = dict(existing_metadata)  # Create a copy
        else:
            metadata = {}

        # Ensure custom_content exists
        if "custom_content" not in metadata or metadata["custom_content"] is None:
            metadata["custom_content"] = {}

        # Add LLM summary
        metadata["custom_content"]["llm_summary"] = {"summary": summary, "model": model_name}

        # Update the DataFrame at the specific index
        try:
            df.at[idx, "metadata"] = metadata
        except Exception:
            # Alternative approach: update the original row reference
            df.iloc[idx]["metadata"] = metadata

    except Exception as e:
        logger.error(f"Failed to add summary to row {idx}: {e}")
