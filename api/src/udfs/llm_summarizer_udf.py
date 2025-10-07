#!/usr/bin/env python3
"""
LLM Content Summarizer UDF for NV-Ingest Pipeline

This UDF uses an LLM to generate concise summaries of text content chunks. These summaries are added to the metadata
for enhanced downstream processing and search capabilities.

These variables can be set in the environment before running the pipeline. These can be treated as kwargs.
- NVIDIA_API_KEY: API key for NVIDIA NIM endpoints (required)
- LLM_SUMMARIZATION_MODEL: Model to use (default: nvidia/llama-3.1-nemotron-70b-instruct)
- LLM_BASE_URL: base URL (default: https://integrate.api.nvidia.com/v1)
- TIMEOUT: API timeout in seconds (default: 60)
- MIN_CONTENT_LENGTH: Minimum content length to summarize (default: 50)
- MAX_CONTENT_LENGTH: Maximum content length to send to API (default: 12000)
TODO: Implement this
- NUM_FIRST_LAST_PAGES: (Optional) Number of first and last pages to summarize. default=1
"""

import logging
import os
import time
import yaml
from pathlib import Path


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
    model_name = os.getenv("LLM_SUMMARIZATION_MODEL", "nvdev/nvidia/llama-3.1-nemotron-70b-instruct")
    base_url = os.getenv("LLM_SUMMARIZATION_BASE_URL", "https://integrate.api.nvidia.com/v1")
    timeout = int(os.getenv("LLM_SUMMARIZATION_TIMEOUT", "60"))
    min_content_length = int(os.getenv("LLM_MIN_CONTENT_LENGTH", "50"))
    max_content_length = int(os.getenv("LLM_MAX_CONTENT_LENGTH", "12000"))
    # Stats for reporting
    stats = {"processed": 0, "summarized": 0, "skipped": 0, "failed": 0, "tokens": 0}

    if api_key is None:
        logger.error("NVIDIA_API_KEY not set. Skipping...")
        return control_message

    # Get the DataFrame payload
    df = control_message.payload()

    if df is None or len(df) == 0:
        logger.warning("No payload found. Nothing to summarize.")
        return control_message

    logger.info(f"Processing {len(df)} chunks for LLM summarization")
    # Select first and last chunk for summarization
    # TODO: add feature to select N first and last chunks
    # According to docs/docs/extraction/user_defined_functions.md#understanding-the-dataframe-payload
    # the rows are not necessarily pages. they are chunks of data extracted from the document. in order to select pages
    # it must require parsing the payload to see which chunks correspond to which pages and then selecting from there
    if len(df) > 1:
        df = df.iloc[[0, -1]]
    else:
        logger.info("Document has only one chunk")

    content_for_summary = ""
    # Don't necessarily need to loop here. Should be able to get all content in one call for all pages.
    # TODO: Should profile this if it necessary
    for idx, row in df.iterrows():
        stats["processed"] += 1
        content = _extract_content(row)

    # Generate summary from combined content from document
    stats["tokens"] = _estimate_tokens(content_for_summary)
    summary, duration = _generate_llm_summary(content_for_summary, model_name, base_url, api_key, timeout)

    ## REMOVE STORING
    # Log the current directory
    logger.info(f"Current working directory: {os.getcwd()}")

    # Create "prompt_dumps" directory if it doesn't exist
    doc_stats_dir = Path("doc_stats") / _safe_model_name(model_name)
    if not doc_stats_dir.exists():
        doc_stats_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {doc_stats_dir.absolute()}")

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
        logger.info(f"Dumping prompt to {filename}")
        yaml.dump(experiment_stats, f, indent=2)
    ###### END

    logger.info(f"LLM summarization complete:\n" f"\tFailed={summary is None},\n" f"\tmodel={model_name}\n")

    # Update the control message with modified DataFrame
    # control_message.payload(df)
    return control_message


def _extract_content(row, min_content_length: int, max_content_length: int) -> str:
    """Extract text content from row"""
    logger.info(f"Extracting content from row: \n\n{row}\n\n")
    metadata = row.get("metadata")

    if metadata is not None and isinstance(metadata, dict):
        content = metadata.get("content")

        if content is not None:
            content = content.strip()
            if len(content) < min_content_length:
                stats["skipped"] += 1
                logger.warning(f"Content less than min={min_content_length}. Skipping...")
                continue
            elif len(content) > max_content_length:
                logger.warning(
                    f"Page {idx}: Content exceeds max length." f"Truncating content to {max_content_length} characters"
                )
                content = content[:max_content_length]

            content_for_summary += content
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
    """Ask an LLM to summarize content extracted from doc."""

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
        else:
            return None, duration

    except Exception as e:
        logger.error(f"API call failed: {e}")
        # TODO: GitHub Thread
        # Reviewers, tell me if this is a bad idea.
        # I think the convention is to return timestamp for time even if it fails
        return None, time.time() - start_time


def _add_summary(df, summary: str, model_name: str):
    """Add summary to metadata and store in df"""
    try:
        # TODO: INCOMPLETE
        logger.info("Adding summary to df...")
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
