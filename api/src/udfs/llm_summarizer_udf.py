#!/usr/bin/env python3
"""
LLM Content Summarizer UDF for NV-Ingest Pipeline

Generates document summaries using NVIDIA-hosted LLMs. This production UDF demonstrates how to extract the pipeline
payload, run custom code (summarization), and inject results into the metadata for downstream usecases (such as
retrieval).

These variables can be set in the environment before running the pipeline. These can be treated as kwargs.
- NVIDIA_API_KEY: API key for NVIDIA NIM endpoints (required)
- LLM_SUMMARIZATION_MODEL: Model to use (default: nvidia/llama-3.1-nemotron-70b-instruct)
- LLM_BASE_URL: base URL (default: https://integrate.api.nvidia.com/v1)
- TIMEOUT: API timeout in seconds (default: 60)
- MIN_CONTENT_LENGTH: Minimum content length to summarize (default: 50)
- MAX_CONTENT_LENGTH: Maximum content length to send to API (default: 12000)
TODO: Implement this
- NUM_CHUNKS: (Optional) Number of first and last pages to summarize. default=1

More info can be found in `examples/udfs/README.md`
"""

import logging
import os
import time


logger = logging.getLogger(__name__)

PROMPT = """
Here are the contents from the first and last page of a document. Focus on the main purpose, key topics,
and important details. Just return the summary as a paragraph. Do not add special characters for formatting.
This summary will be used for document search and understanding.

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
    logger.info("UDF: Starting LLM content summarization")

    api_key = os.getenv("NVIDIA_API_KEY")
    model_name = os.getenv("LLM_SUMMARIZATION_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
    base_url = os.getenv("LLM_SUMMARIZATION_BASE_URL", "https://integrate.api.nvidia.com/v1")
    min_content_length = int(os.getenv("LLM_MIN_CONTENT_LENGTH", 50))
    max_content_length = int(os.getenv("LLM_MAX_CONTENT_LENGTH", 12000))
    timeout = int(os.getenv("LLM_SUMMARIZATION_TIMEOUT", 60))

    stats = {
        "skipped": False,
        "failed": False,
        "tokens": 0,
        "duration": 0.0,
    }

    if not api_key:
        logger.error("NVIDIA_API_KEY not set. Skipping...")
        return control_message

    df = control_message.payload()

    if df is None or df.empty:
        logger.warning("No payload found. Nothing to summarize.")
        return control_message

    # Select first and last chunk for summarization
    # According to docs/docs/extraction/user_defined_functions.md#understanding-the-dataframe-payload
    # the rows are not necessarily pages. they are chunks of data extracted from the document. in order to select
    # pages, it must require parsing the payload to see which chunks correspond to which pages
    original_df = df.copy()
    if len(df) > 1:
        # TODO: add feature to select N first and last chunks
        df = df.iloc[[0, -1]]
    else:
        logger.info("Document has only one chunk")

    # Combine all content into a single string
    content_list = df.apply(
        _extract_content,
        axis=1,
        min_content_length=min_content_length,
        max_content_length=max_content_length,
        stats=stats,
    )
    content = " ".join(content_list)

    # Nicely ask LLM to summarize content
    summary, stats["duration"] = _generate_llm_summary(content, model_name, base_url, api_key, timeout)

    stats["failed"] = summary is None
    if not stats["failed"]:
        stats["tokens"] = _estimate_tokens(content)
        logger.info("Summarized %d tokens in %f seconds using %s", stats["tokens"], stats["duration"], model_name)
        _store_summary(original_df, summary, model_name)

        # Update the control message with modified DataFrame
        control_message.payload(original_df)

    else:
        logger.warning("%s failed to summarize content", model_name)

    return control_message


def _extract_content(row, stats: dict, min_content_length: int = 50, max_content_length: int = 12000) -> str | None:
    """Extract text content from row"""
    metadata = row.get("metadata")

    if isinstance(metadata, dict):
        content = metadata.get("content")
        if content is not None:
            content = content.strip()
            if len(content) < min_content_length:
                stats["skipped"] = True
                logger.warning(f"Content less than min={min_content_length}. Skipping...")
                content = ""
            elif len(content) > max_content_length:
                logger.warning(f"Truncating content to {max_content_length} characters")
                content = content[:max_content_length]
        else:
            stats["skipped"] = True
            content = ""

    else:
        stats["skipped"] = True
        logger.warning("No metadata found. Skipping...")
        content = ""

    return content


def _generate_llm_summary(
    content: str,
    model_name: str,
    base_url: str,
    api_key: str,
    timeout: int,
) -> tuple[str | None, float]:
    """Ask an LLM to summarize content extracted from doc."""

    start_time = time.time()
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        start_time = time.time()
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT.format(content=content)}],
            max_tokens=400,  # Increased for more comprehensive summaries
            temperature=0.7,
        )
        duration = time.time() - start_time

        if completion.choices:
            summary = completion.choices[0].message.content.strip()
            return summary, duration
        return None, duration

    except Exception as e:
        logger.error(f"API call failed: {e}")
        # TODO: GitHub Thread
        # Reviewers, tell me if this is a bad idea.
        # I think the convention is to return timestamp for time even if it fails
        return None, time.time() - start_time


def _store_summary(df, summary: str, model_name: str):
    """Add summary to metadata and store in df"""
    # hardcoded heuristic to store everything on chunk 0's metadata
    row_0 = df.iloc[0]

    # this is a reference to a dictionary that is stored in the dataframe
    # and is modified in place
    metadata = row_0.get("metadata")

    if metadata.get("custom_content") is None:
        metadata["custom_content"] = {}
    metadata["custom_content"]["llm_summarizer_udf"] = {"summary": summary, "model": model_name}


def _estimate_tokens(text: str) -> int:
    """Rough estimate (~4 characters per token)"""
    return len(text) // 4


def _safe_model_name(name: str) -> str:
    return name.replace("/", "__").replace("-", "_")
