# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import copy
import logging
import re
import uuid
from typing import Any, Optional, Dict, List

import pandas as pd
from openai import OpenAI
from transformers import AutoTokenizer

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.transform.transform_structural_text_splitter_schema import StructuralTextSplitterSchema
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler

logger = logging.getLogger(__name__)


def _build_structural_split_documents(
    row, chunks: List[Dict[str, str]], hierarchical_headers: List[str]
) -> List[Dict[str, Any]]:
    """Build documents from text chunks and their hierarchical headers."""
    documents: List[dict] = []

    for i, text in enumerate(chunks):
        if text is None or not text.strip():
            continue

        metadata = row.metadata if hasattr(row, "metadata") and isinstance(row.metadata, dict) else {}
        metadata = copy.deepcopy(metadata)

        # Add hierarchical header to a new custom_content field
        header = hierarchical_headers[i] if i < len(hierarchical_headers) else ""
        metadata["custom_content"] = {"hierarchical_header": header}

        if row.document_type == ContentTypeEnum.AUDIO:
            metadata["audio_metadata"]["audio_transcript"] = text
            documents.append(
                {"document_type": ContentTypeEnum.AUDIO.value, "metadata": metadata, "uuid": str(uuid.uuid4())}
            )
        else:
            metadata["content"] = text
            documents.append(
                {"document_type": ContentTypeEnum.TEXT.value, "metadata": metadata, "uuid": str(uuid.uuid4())}
            )

    return documents


def _get_llm_split_point(text: str, client: OpenAI, config: StructuralTextSplitterSchema) -> Optional[str]:
    """Call LLM to get a suggested split point for oversized text."""
    try:
        prompt = config.llm_prompt.format(text=text)
        completion = client.chat.completions.create(
            model=config.llm_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,  # A few words should be enough
        )
        split_marker = completion.choices[0].message.content.strip()

        # Validate that the response is a verbatim substring
        if split_marker and split_marker in text:
            logger.debug(f"LLM returned valid split point: '{split_marker}'")
            return split_marker
        else:
            logger.warning(
                f"LLM response '{split_marker}' is not a valid substring. "
                "Falling back to hard split for this chunk."
            )
            return None
    except Exception as e:
        logger.error(f"Error calling LLM for text splitting: {e}. Falling back to hard split.", exc_info=True)
        return None


def _split_by_markdown(
    text: str, tokenizer: AutoTokenizer, config: StructuralTextSplitterSchema, client: Optional[OpenAI]
) -> (List[str], List[str]):
    """Splits text by markdown headers and then sub-splits oversized chunks."""
    sorted_headers = sorted(config.markdown_headers_to_split_on, key=len, reverse=True)
    # Use capturing groups to separate the hash markers from the header text.
    header_pattern = f"^({'|'.join(re.escape(h) for h in sorted_headers)})[ \t]+(.+)"
    splits = re.split(header_pattern, text, flags=re.MULTILINE)

    # The first split is the content before the first header
    if splits:
        initial_content = splits.pop(0).strip()
    else:
        initial_content = text.strip()

    chunks = []
    hierarchical_headers = []
    header_stack = []

    if initial_content:
        chunks.append(initial_content)
        hierarchical_headers.append("")

    # Process splits in groups of 3: (header_prefix, header_text, content)
    for i in range(0, len(splits), 3):
        header_prefix = splits[i]
        header_text = splits[i + 1].strip()
        content = splits[i + 2].strip()

        header_level = len(header_prefix)

        # Update header stack
        while header_stack and header_stack[-1]["level"] >= header_level:
            header_stack.pop()
        header_stack.append({"level": header_level, "text": header_text})

        # Create hierarchical header string
        current_hierarchical_header = " > ".join([h["text"] for h in header_stack])

        if content:
            chunks.append(content)
            hierarchical_headers.append(current_hierarchical_header)

    final_chunks = []
    final_headers = []
    llm_split_count = 0

    for i, chunk in enumerate(chunks):
        header = hierarchical_headers[i]
        tokens = tokenizer.encode(chunk, add_special_tokens=False)

        if len(tokens) <= config.max_chunk_size_tokens:
            final_chunks.append(chunk)
            final_headers.append(header)
            continue

        # Logic for oversized chunks
        logger.debug(f"Chunk with header '{header}' is oversized ({len(tokens)} tokens). Attempting to sub-split.")
        use_llm = (
            config.enable_llm_enhancement
            and client
            and llm_split_count < config.max_llm_splits_per_document
        )

        sub_chunks = []
        if use_llm:
            llm_split_count += 1
            split_point = _get_llm_split_point(chunk, client, config)
            if split_point:
                # Simple split in two for now, can be made recursive later
                try:
                    parts = chunk.split(split_point, 1)
                    if len(parts) == 2:
                        sub_chunks.extend([parts[0], split_point + parts[1]])
                except Exception as e:
                    logger.warning(f"Failed to split on LLM marker: {e}")

        # Fallback to hard splitting
        if not sub_chunks:
            logger.debug("Using hard token-based splitting for oversized chunk.")
            start = 0
            while start < len(tokens):
                end = start + config.max_chunk_size_tokens
                sub_chunk_tokens = tokens[start:end]
                sub_chunks.append(tokenizer.decode(sub_chunk_tokens, skip_special_tokens=True))
                if end >= len(tokens):
                    break
                start += config.max_chunk_size_tokens

        final_chunks.extend(sub_chunks)
        final_headers.extend([header] * len(sub_chunks)) # Propagate header to all sub-chunks

    return final_chunks, final_headers


@unified_exception_handler
def transform_text_split_structural_internal(
    df_transform_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    transform_config: StructuralTextSplitterSchema,
    execution_trace_log: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Internal function to split text using markdown headers and optionally an LLM.
    """
    _ = execution_trace_log

    # Combine and override configs
    config_dict = transform_config.model_dump()
    config_dict.update(task_config)
    config = StructuralTextSplitterSchema(**config_dict)

    logger.debug(
        f"Splitting text with structural config: {config.model_dump_json(indent=2)}"
    )

    # Initialize LLM Client if needed
    client = None
    if config.enable_llm_enhancement and config.llm_endpoint:
        api_key = os.getenv(config.llm_api_key_env_var)
        if not api_key:
            logger.warning(
                f"LLM enhancement is enabled, but env var '{config.llm_api_key_env_var}' is not set. Disabling."
            )
            config.enable_llm_enhancement = False
        else:
            logger.info(f"Initializing OpenAI client for LLM enhancement at endpoint: {config.llm_endpoint}")
            logger.warning(
                "LLM enhancement for structural splitting is enabled. This feature can significantly "
                "increase processing time and cost for large documents."
            )
            client = OpenAI(base_url=config.llm_endpoint, api_key=api_key)

    # Filter to documents with text content
    text_type_condition = df_transform_ledger["document_type"].isin([ContentTypeEnum.TEXT, ContentTypeEnum.AUDIO])
    df_filtered: pd.DataFrame = df_transform_ledger.loc[text_type_condition]

    if df_filtered.empty:
        return df_transform_ledger

    # Re-using tokenizer loading logic from standard splitter
    model_predownload_path = os.environ.get("MODEL_PREDOWNLOAD_PATH", "")
    tokenizer_identifier = "meta-llama/Llama-3.2-1B" # A reasonable default
    if os.path.exists(os.path.join(model_predownload_path, "llama-3.2-1b/tokenizer/tokenizer.json")):
        tokenizer_identifier = os.path.join(model_predownload_path, "llama-3.2-1b/tokenizer/")

    tokenizer_model = AutoTokenizer.from_pretrained(tokenizer_identifier)

    split_docs: List[Dict[str, Any]] = []
    for _, row in df_filtered.iterrows():
        if row["document_type"] == ContentTypeEnum.AUDIO:
            content: str = row.get("metadata", {}).get("audio_metadata", {}).get("audio_transcript", "")
        else:
            content: str = row.get("metadata", {}).get("content", "")

        if not content:
            continue

        chunks, headers = _split_by_markdown(content, tokenizer_model, config, client)
        split_docs.extend(_build_structural_split_documents(row, chunks, headers))

    if not split_docs:
         return df_transform_ledger[~text_type_condition].reset_index(drop=True)

    split_docs_df: pd.DataFrame = pd.DataFrame(split_docs)

    # Merge split documents with unsplit documents.
    merged_df: pd.DataFrame = pd.concat([split_docs_df, df_transform_ledger[~text_type_condition]], axis=0).reset_index(
        drop=True
    )

    result, execution_trace_log = merged_df, {}

    return result 