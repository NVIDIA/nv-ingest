#!/usr/bin/env python3
"""
LLM Content Summarizer UDF for NV-Ingest Pipeline

This UDF uses an LLM to generate concise summaries of text content chunks. These summaries are added to the metadata
for enhanced downstream processing and search capabilities.

By default, this uses NVIDIA BUILD-hosted Nemotron-mini-4b-instruct as an example, but you can customize it to use any
OpenAI-compatible endpoint (other NVIDIA BUILD models, local LLMs via Ollama/vLLM, self-hosted NIM, etc.) by setting
LLM_SUMMARIZATION_BASE_URL and LLM_SUMMARIZATION_MODEL.

Environment variables (can be treated as kwargs):
- NVIDIA_API_KEY: API key for NVIDIA NIM endpoints (required for hosted endpoints)
- LLM_SUMMARIZATION_MODEL: Model to use (default: nvidia/nemotron-mini-4b-instruct)
- LLM_SUMMARIZATION_BASE_URL: Base URL for OpenAI-compatible API (default: https://integrate.api.nvidia.com/v1)
- LLM_SUMMARIZATION_TIMEOUT: API timeout in seconds (default: 60)
- LLM_MIN_CONTENT_LENGTH: Minimum content length to summarize (default: 50)
- LLM_MAX_CONTENT_LENGTH: Maximum content length to send to API (default: 12000)

More info can be found in `examples/udfs/README.md`
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

PROMPT = """
Based on the contents from the first and last page of a document below, provide a single sentence summary that \
captures the main purpose and key topics. Do not add special characters for formatting.

[CONTENT]
{content}
[END CONTENT]
"""


def content_summarizer(control_message: "IngestControlMessage") -> "IngestControlMessage":  # noqa: F821
    """
    UDF function that adds LLM-generated summaries to text content chunks.

    This function processes text primitives and generates concise summaries using
    an LLM API, storing the results in the metadata's custom_content field.

    Parameters
    ----------
    control_message : IngestControlMessage
        The control message containing the DataFrame payload with text content

    Returns
    -------
    IngestControlMessage
        The modified control message with LLM summaries added to metadata
    """
    udf_start_time = time.time()

    # Load configuration
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY")  # Using NGC_API_KEY if NVIDIA_API_KEY is not set
    model_name = os.getenv("LLM_SUMMARIZATION_MODEL", "nvidia/nemotron-mini-4b-instruct")
    base_url = os.getenv("LLM_SUMMARIZATION_BASE_URL", "https://integrate.api.nvidia.com/v1")
    min_content_length = int(os.getenv("LLM_MIN_CONTENT_LENGTH", 50))
    max_content_length = int(os.getenv("LLM_MAX_CONTENT_LENGTH", 12000))
    timeout = int(os.getenv("LLM_SUMMARIZATION_TIMEOUT", 60))

    stats = {
        "processed": 0,
        "summarized": 0,
        "skipped": 0,
        "failed": 0,
        "tokens": 0,
    }

    logger.info(f"Configuration: model={model_name}, base_url={base_url}")
    logger.info(
        f"Configuration: timeout={timeout}s, min_content={min_content_length}, max_content={max_content_length}"
    )

    if not api_key:
        logger.error("NVIDIA_API_KEY not set - skipping LLM summarization")
        return control_message

    df = control_message.payload()
    if df is None or df.empty:
        logger.warning("Empty payload - skipping LLM summarization")
        return control_message

    # Extract document name
    doc_name = _extract_document_name(df)
    logger.info(f"LLM summarization starting: {doc_name} ({len(df)} chunks, model={model_name})")

    # Save original dataframe to preserve all chunks
    original_df = df.copy()

    extraction_start = time.time()
    if len(df) > 1:
        # Select first and last chunk for summarization
        # TODO: add feature to select N first and last chunks
        # According to docs/docs/extraction/user_defined_functions.md#understanding-the-dataframe-payload
        # the rows are not necessarily pages. they are chunks of data extracted from the document. in order to select
        # pages, it must require parsing the payload to see which chunks correspond to which pages and then selecting
        # from there
        logger.info(f"Selecting first and last chunks (out of {len(df)} total) for summarization")
        selected_df = df.iloc[[0, -1]]
    else:
        logger.info("Document has only one chunk")
        selected_df = df

    # Combine all content into a single string
    logger.info("Extracting and combining content from selected chunks...")
    content = " ".join(
        selected_df.apply(
            _extract_content,
            axis=1,
            min_content_length=min_content_length,
            max_content_length=max_content_length,
            stats=stats,
        )
    )
    stats["tokens"] = _estimate_tokens(content)
    extraction_time = time.time() - extraction_start
    logger.info(
        f"Content extraction completed: {len(content)} characters, "
        f"~{stats['tokens']} tokens (took {extraction_time:.2f}s)"
    )

    logger.info(f"Calling LLM API ({model_name}) for summarization...")
    summary, llm_duration = _generate_llm_summary(content, model_name, base_url, api_key, timeout)

    if summary:
        tokens_per_sec = stats["tokens"] / llm_duration if llm_duration > 0 else 0
        logger.info(
            f"LLM API call completed: duration={llm_duration:.2f}s, "
            f"tokens={stats['tokens']}, throughput={tokens_per_sec:.1f} tokens/s"
        )
        logger.info(
            f"Generated summary ({len(summary)} chars): {summary[:100]}..."
            if len(summary) > 100
            else f"Generated summary: {summary}"
        )
    else:
        logger.error(f"LLM API call failed (took {llm_duration:.2f}s)")

    # Store summary in chunk 0 of the original dataframe (preserves all chunks)
    _store_summary(original_df, summary, model_name)

    # Calculate total UDF time
    udf_total_time = time.time() - udf_start_time

    # Log summary
    logger.info("=" * 80)
    logger.info(f"LLM Summarization Complete - Document: {doc_name}")
    logger.info(f"  Status: {'SUCCESS' if summary else 'FAILED'}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Content extraction time: {extraction_time:.2f}s")
    logger.info(f"  LLM API call time: {llm_duration:.2f}s")
    logger.info(f"  Total UDF time: {udf_total_time:.2f}s")
    logger.info(f"  Chunks preserved: {len(original_df)} (all chunks kept)")
    if summary and llm_duration > 0:
        logger.info(f"  Throughput: {stats['tokens']/llm_duration:.1f} tokens/s")
    logger.info("=" * 80)

    # Update the control message with modified DataFrame (all chunks preserved)
    control_message.payload(original_df)
    return control_message


def _extract_content(row, min_content_length: int, max_content_length: int, stats: dict) -> str:
    """Extract text content from row"""
    metadata = row.get("metadata")
    content = ""

    if isinstance(metadata, dict):
        content = metadata.get("content")
        if content is not None:
            content = content.strip()
            if len(content) < min_content_length:
                stats["skipped"] += 1
                return ""
            elif len(content) > max_content_length:
                logger.debug(f"Truncating content to {max_content_length} characters")
                content = content[:max_content_length]
        else:
            stats["skipped"] += 1

    return content


def _generate_llm_summary(
    content: str,
    model_name: str,
    base_url: str,
    api_key: str,
    timeout: int,
) -> tuple[str | None, float]:
    """
    Generate summary using LLM API.

    Returns
    -------
    tuple[str | None, float]
        Summary text (or None if failed) and duration in seconds
    """
    start_time = time.time()

    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT.format(content=content)}],
            max_tokens=400,
            temperature=0.7,
        )

        duration = time.time() - start_time

        if completion.choices:
            summary = completion.choices[0].message.content.strip()
            return summary, duration

        logger.warning("LLM returned no completion choices")
        return None, duration

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"LLM API call failed ({duration:.2f}s): {type(e).__name__}: {str(e)[:200]}")
        return None, duration


def _extract_document_name(df) -> str:
    """Extract source document name from dataframe metadata"""
    try:
        if len(df) > 0 and "metadata" in df.iloc[0]:
            metadata = df.iloc[0].get("metadata", {})
            if isinstance(metadata, dict):
                source_metadata = metadata.get("source_metadata", {})
                if isinstance(source_metadata, dict):
                    return source_metadata.get("source_name", "Unknown")
    except Exception as e:
        logger.debug(f"Could not extract document name: {e}")
    return "Unknown"


def _store_summary(df, summary: str, model_name: str):
    """Add summary to metadata and store in df"""
    row_0 = df.iloc[0]
    metadata = row_0.get("metadata")

    if metadata.get("custom_content") is None:
        metadata["custom_content"] = {}
    metadata["custom_content"]["llm_summarizer_udf"] = {"summary": summary, "model": model_name}


def _estimate_tokens(text: str) -> int:
    """Rough estimate (~4 characters per token)"""
    return len(text) // 4
