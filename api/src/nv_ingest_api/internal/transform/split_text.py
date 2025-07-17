# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import copy
import logging
import uuid
from typing import Any, Optional, Dict
from typing import List

import pandas as pd
from transformers import AutoTokenizer

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler

logger = logging.getLogger(__name__)


def _build_split_documents(row, chunks: List[str]) -> List[dict[str, Any]]:
    """Build documents from text chunks"""
    documents: List[dict] = []

    for i, text in enumerate(chunks):
        if text is None or not text.strip():
            continue

        metadata = row.metadata if hasattr(row, "metadata") and isinstance(row.metadata, dict) else {}
        metadata = copy.deepcopy(metadata)

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


def _split_into_chunks(text, tokenizer, chunk_size=1024, chunk_overlap=20):
    # Tokenize the text into token IDs
    encoding = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)

    # Get the token IDs and offsets for splitting
    offsets = encoding["offset_mapping"]

    # Split the tokens into chunks of the desired size with the desired overlap
    chunks = [offsets[i : i + chunk_size] for i in range(0, len(offsets), chunk_size - chunk_overlap)]

    # Convert token chunks back to text while preserving original spacing and case
    text_chunks = []
    for chunk in chunks:
        text_chunk = text[chunk[0][0] : chunk[-1][0]]
        text_chunks.append(text_chunk)

    return text_chunks


@unified_exception_handler
def transform_text_split_and_tokenize_internal(
    df_transform_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    transform_config: TextSplitterSchema,
    execution_trace_log: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Internal function to split and tokenize text in a ledger DataFrame.

    This function extracts text from documents that match a filter criteria based on source types,
    splits the text into chunks using the specified tokenizer, and rebuilds document records with the
    split text. The resulting DataFrame contains both split and unsplit documents.

    Parameters
    ----------
    df_transform_ledger : pd.DataFrame
        DataFrame containing documents to be processed. Expected to have columns 'document_type' and
        'metadata', where 'metadata' includes a 'content' field and nested source information.
    task_config : dict
        Dictionary with task-specific configuration. Expected keys include:
            - "tokenizer": Tokenizer identifier or path.
            - "chunk_size": Maximum number of tokens per chunk.
            - "chunk_overlap": Number of tokens to overlap between chunks.
            - "params": A sub-dictionary that may contain:
                - "hf_access_token": Hugging Face access token.
                - "split_source_types": List of source types to filter for splitting.
    transform_config : TextSplitterSchema
        Configuration object providing default values for text splitting parameters.
    execution_trace_log : Optional[dict]
        Optional dictionary for logging execution trace information; may be None.

    Returns
    -------
    pd.DataFrame
        DataFrame with processed documents. Documents with text matching the filter are split into chunks,
        and then merged with those that do not match the filter.

    Raises
    ------
    ValueError
        If the text splitting or tokenization process fails.
    """
    _ = execution_trace_log  # Placeholder for potential execution trace logging.

    # Override parameters using task_config, with fallback to transform_config.
    tokenizer_identifier: Optional[str] = task_config.get("tokenizer", transform_config.tokenizer)
    chunk_size: int = task_config.get("chunk_size", transform_config.chunk_size)
    chunk_overlap: int = task_config.get("chunk_overlap", transform_config.chunk_overlap)
    params: Dict[str, Any] = task_config.get("params", {})

    hf_access_token: Optional[str] = params.get("hf_access_token", None)
    split_source_types: List[str] = params.get("split_source_types", ["text"])

    logger.debug(
        f"Splitting text with tokenizer: {tokenizer_identifier}, "
        f"chunk_size: {chunk_size} tokens, "
        f"chunk_overlap: {chunk_overlap}"
    )

    # Filter to documents with text content.
    text_type_condition = df_transform_ledger["document_type"].isin([ContentTypeEnum.TEXT, ContentTypeEnum.AUDIO])

    normalized_meta_df = pd.json_normalize(df_transform_ledger["metadata"], errors="ignore")
    if "source_metadata.source_type" in normalized_meta_df.columns:
        source_type_condition = normalized_meta_df["source_metadata.source_type"].isin(split_source_types)
    else:
        source_type_condition = False

    bool_index = text_type_condition & source_type_condition
    df_filtered: pd.DataFrame = df_transform_ledger.loc[bool_index]

    if df_filtered.empty:
        return df_transform_ledger

    model_predownload_path = os.environ.get("MODEL_PREDOWNLOAD_PATH")

    if model_predownload_path is not None:
        if os.path.exists(os.path.join(model_predownload_path, "llama-3.2-1b/tokenizer/tokenizer.json")) and (
            tokenizer_identifier is None or tokenizer_identifier == "meta-llama/Llama-3.2-1B"
        ):
            tokenizer_identifier = os.path.join(model_predownload_path, "llama-3.2-1b/tokenizer/")
        elif os.path.exists(
            os.path.join(model_predownload_path, "e5-large-unsupervised/tokenizer/tokenizer.json")
        ) and (tokenizer_identifier is None or tokenizer_identifier == "intfloat/e5-large-unsupervised"):
            tokenizer_identifier = os.path.join(model_predownload_path, "e5-large-unsupervised/tokenizer/")

    # Defaulto to intfloat/e5-large-unsupervised if no tokenizer predownloaded or specified
    if tokenizer_identifier is None:
        tokenizer_identifier = "intfloat/e5-large-unsupervised"

    tokenizer_model = AutoTokenizer.from_pretrained(tokenizer_identifier, token=hf_access_token)

    split_docs: List[Dict[str, Any]] = []
    for _, row in df_filtered.iterrows():
        if row["document_type"] == ContentTypeEnum.AUDIO:
            content: str = (
                row["metadata"]["audio_metadata"]["audio_transcript"]
                if row["metadata"]["audio_metadata"]["audio_transcript"] is not None
                else ""
            )
        else:
            content: str = row["metadata"]["content"] if row["metadata"]["content"] is not None else ""
        chunks: List[str] = _split_into_chunks(content, tokenizer_model, chunk_size, chunk_overlap)
        split_docs.extend(_build_split_documents(row, chunks))

    split_docs_df: pd.DataFrame = pd.DataFrame(split_docs)

    # Merge split documents with unsplit documents.
    merged_df: pd.DataFrame = pd.concat([split_docs_df, df_transform_ledger[~bool_index]], axis=0).reset_index(
        drop=True
    )

    result, execution_trace_log = merged_df, {}

    return result
