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

import logging
import os
import time


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

    Environment Variable Argument Handling:

    These variables can be set in the environment before running the pipeline. These can be treated as kwargs.

    - NVIDIA_API_KEY: (Required) The API key for authenticating with NVIDIA NIM endpoints.
    - LLM_SUMMARIZATION_MODEL: (Optional) The NIM model to use for summarization.
      default="nvidia/llama-3.1-nemotron-70b-instruct"
    - LLM_SUMMARIZATION_BASE_URL: (Optional) The base URL for the LLM API endpoint.
      default="https://integrate.api.nvidia.com/v1"
    - LLM_SUMMARIZATION_TIMEOUT: (Optional) Timeout in seconds for API requests.
      default=60 seconds
    - LLM_MIN_CONTENT_LENGTH: (Optional) Minimum number of characters required in a content
      chunk to trigger summarization. default=50
    - LLM_MAX_CONTENT_LENGTH: (Optional) Maximum number of characters to send to the API
      for summarization. default=12000
    - NUM_FIRST_LAST_PAGES: (Optional) Number of first and last pages to summarize. default=1


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

    # Remove me
    logger.info("")
    logger.info("=================SUMMARY OF THE DATAFRAME==================")
    logger.info(f"df: {df.head()}")
    logger.info("=================DIRECTORY CONTENTS==================")
    logger.info(f"Current absolute path: {os.path.abspath(os.curdir)}")
    logger.info(f"Directory contents: {os.listdir(os.curdir)}")
    df.to_csv("df.csv", index=False)
    # Remove me

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
    stats = {"processed": 0, "summarized": 0, "skipped": 0, "failed": 0, "tokens": 0}

    # Process each row (page) of the document
    # Probably don't need to loop here. Looped LLM calls is slow. We know # pages in doc and how many
    # pages to extract. This should be parallelized.
    for idx, row in df.iterrows():
        stats["processed"] += 1
        try:
            # Extract content - be more flexible about where it comes from
            content = _extract_content(row, logger)

            if content is not None:
                content = content.strip()
                if len(content) < min_content_length:
                    stats["skipped"] += 1
                    logger.info(f"Page {idx}: Content less than min={min_content_length}. Skipping...")
                    continue

                # Truncate if needed
                if len(content) > max_content_length:
                    logger.info(
                        "Warning: Content exceeds max length." f"Truncating content to {max_content_length} characters"
                    )
                    content = content[:max_content_length]
                # remove
                logger.info(f"Page {idx}: Content: {content}")
                # remove
                stats["tokens"] += _estimate_tokens(content)

                # Generate summary
                summary, duration = _generate_summary(client, content, model_name, logger)

                if summary is not None:
                    # Add to metadata
                    _add_summary(df, idx, row, summary, model_name, logger)
                    stats["summarized"] += 1
                else:
                    stats["failed"] += 1

            else:
                stats["skipped"] += 1

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


def _extract_content(row, logger) -> str | None:
    """Extract text content from row, trying multiple locations."""
    content = None

    # Try different locations for content
    if isinstance(row.get("metadata"), dict):
        metadata = row["metadata"]

        # Primary location: metadata.content
        content = metadata.get("content", None)

        # If no content, try other locations
        # TODO: This does not seem to adhere to the schema of the document!!
        if content is None:
            # Try in text_metadata
            text_metadata = metadata.get("text_metadata", {})
            content = text_metadata.get("text", "") or text_metadata.get("content", "")

    # Try top-level content field
    # TODO: Convert to GitHub Thread
    # This does not seem to adhere to the schema of a document!!
    # Why would we look in here? We only expect it to be under row.metadata
    if content is None:
        content = row.get("content", None)

    return content


def _generate_summary(client, content: str, model_name: str, logger) -> tuple[str | None, float]:
    """Generate summary with robust error handling."""
    prompt = f"""Please provide a comprehensive 3-4 sentence summary of the following document:

{content}

Focus on the main purpose, key topics, and important details.
This summary will be used for document search and understanding.

Summary:"""

    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,  # Increased for more comprehensive summaries
            temperature=0.7,
        )
        duration = time.time() - start_time

        if completion.choices:
            summary = completion.choices[0].message.content.strip()
            return summary, duration
        else:
            return None, duration

    except Exception as e:
        # TODO: GitHub Thread
        # Reviewers, tell me if this is a bad idea.
        # I think the convention is to return timestamp for time even if it fails
        logger.error(f"API call failed: {e}")
        return None, time.time() - start_time


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


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token for English."""
    return len(text) // 4
