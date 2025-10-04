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

import yaml
from pathlib import Path

PROMPT = """
Here are the contents from the first and last page of a document. Focus on the main purpose, key topics,
and important details. Just return the summary as a paragraph. Do not add special characters for formatting.
This summary will be used for document search and understanding.

[CONTENT]
{content}
[END CONTENT]
"""

logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    """Log a message"""
    logger.info(msg)


def warn(msg: str) -> None:
    """Log warning msg"""
    logger.warning(msg)


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
    - LLM_MIN_CONTENT_LENGTH: (Optional) Minimum number of chars required in a content
      chunk to trigger summarization. default=50
    - LLM_MAX_CONTENT_LENGTH: (Optional) Maximum number of chars to send to the API
      for summarization. default=12000

      TODO: Implement this
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

    log("UDF: Starting LLM content summarization")

    # Get configuration from environment
    api_key = os.getenv("NVIDIA_API_KEY", "")
    # BUG: This doesn't work correctly. Env var set on client doesn't propagate to UDF on docker service
    model_name = os.getenv("LLM_SUMMARIZATION_MODEL", "nvdev/nvidia/llama-3.1-nemotron-70b-instruct")
    base_url = os.getenv("LLM_SUMMARIZATION_BASE_URL", "https://integrate.api.nvidia.com/v1")
    timeout = int(os.getenv("LLM_SUMMARIZATION_TIMEOUT", "60"))
    min_content_length = int(os.getenv("LLM_MIN_CONTENT_LENGTH", "50"))
    max_content_length = int(os.getenv("LLM_MAX_CONTENT_LENGTH", "12000"))

    if not api_key:
        warn("NVIDIA_API_KEY not found, skipping summarization")
        return control_message

    # Get the DataFrame payload
    df = control_message.payload()

    if df is None or len(df) == 0:
        warn("No payload found in control message")
        return control_message

    log(f"Processing {len(df)} pages for LLM summarization")
    # Select first and last page for summarization
    # TODO: add feature to select N first and last pages
    if len(df) > 1:
        df = df.iloc[[0, -1]]
    else:
        log("Document has only one page")

    # Remove me
    # df.to_csv("df.csv", index=False)
    # Remove me

    # Initialize OpenAI client with error handling
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return control_message

    # Stats for reporting
    stats = {"processed": 0, "summarized": 0, "skipped": 0, "failed": 0, "tokens": 0}

    content_for_summary = ""
    # Don't necessarily need to loop here. Should be able to get all content in one call for all pages.
    # TODO: Should profile this if it necessary
    for idx, row in df.iterrows():
        stats["processed"] += 1
        content = _extract_content(row)

        if content is not None:
            content = content.strip()
            if len(content) < min_content_length:
                stats["skipped"] += 1
                warn(f"Page {idx}: Content less than min={min_content_length}. Skipping...")
                continue
            elif len(content) > max_content_length:
                warn(
                    f"Page {idx}: Content exceeds max length." f"Truncating content to {max_content_length} characters"
                )
                content = content[:max_content_length]

            content_for_summary += content
        else:
            stats["skipped"] += 1

    # Generate summary from combined content from document
    stats["tokens"] = _estimate_tokens(content_for_summary)
    summary, duration = _generate_summary(client, content_for_summary, model_name)

    ## REMOVE STORING
    # Log the current directory
    log(f"Current working directory: {os.getcwd()}")

    # Create "prompt_dumps" directory if it doesn't exist
    doc_stats_dir = Path("doc_stats") / _safe_model_name(model_name)
    if not doc_stats_dir.exists():
        doc_stats_dir.mkdir(parents=True, exist_ok=True)
        log(f"Created directory: {doc_stats_dir.absolute()}")

    doc_name = Path(df.iloc[0]["metadata"]["source_metadata"]["source_name"].split("/")[-1])
    filename = doc_stats_dir / doc_name.with_suffix(".yaml")
    # Write the contents of for_randy as a YAML file named after the 'doc' field
    experiment_stats = {
        "doc": str(doc_name),
        "prompt": PROMPT.format(content=content_for_summary),
        "tokens": stats["tokens"],
        "duration": duration,
        "tokens/s": stats["tokens"] / duration,
        "summary": summary,
        "model": model_name,
    }
    with open(filename, "w") as f:
        log(f"Dumping prompt to {filename}")
        yaml.dump(experiment_stats, f, indent=2)
    ###### END

    if summary is not None:
        # _add_summary(df, summary, model_name)
        stats["summarized"] += 1
    else:
        stats["failed"] += 1

    log(
        f"LLM summarization complete:\n"
        f"\ttokens={stats['tokens']},\n"
        f"\tduration={duration:.2f}s,\n"
        f"\ttokens/s={(stats['tokens'] / duration):.2f}\n"
        f"\tmodel={model_name}\n"
    )

    # Update the control message with modified DataFrame
    # control_message.payload(df)
    return control_message


def _extract_content(row) -> str | None:
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


def _generate_summary(client, content: str, model_name: str) -> tuple[str | None, float]:
    """Ask an LLM to summarize content extracted from doc."""
    ### SEE if prompt should go in here. Also see if

    try:
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
        else:
            return None, duration

    except Exception as e:
        # TODO: GitHub Thread
        # Reviewers, tell me if this is a bad idea.
        # I think the convention is to return timestamp for time even if it fails
        logger.error(f"API call failed: {e}")
        return None, time.time() - start_time


def _add_summary(df, summary: str, model_name: str):
    """Add summary to metadata and store in df"""
    try:
        # TODO: INCOMPLETE
        log("Adding summary to df...")
        # metadata["custom_content"]["llm_summary"] = {"summary": summary, "model": model_name}

        # # Update the DataFrame at the specific index
        # try:
        #     df.at[idx, "metadata"] = metadata
        # except Exception:
        #     # Alternative approach: update the original row reference
        #     df.iloc[idx]["metadata"] = metadata

    except Exception as e:
        logger.error(e)


def _estimate_tokens(text: str) -> int:
    """Rough estimate (~4 characters per token)"""
    return len(text) // 4


def _safe_model_name(name: str) -> str:
    return name.replace("/", "__").replace("-", "_")
