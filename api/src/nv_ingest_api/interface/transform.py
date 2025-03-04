# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any, List

import pandas as pd

from nv_ingest.schemas import ImageCaptionExtractionSchema, TextSplitterSchema
from nv_ingest.schemas.embed_extractions_schema import EmbedExtractionsSchema
from nv_ingest_api.internal.transform.caption_image import transform_image_create_vlm_caption_internal
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal
from nv_ingest_api.internal.transform.split_text import transform_text_split_and_tokenize_internal
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler


@unified_exception_handler
def transform_text_create_embeddings(
    *,
    df_ledger: pd.DataFrame,
    api_key: str,
    embedding_model: Optional[str] = None,
    embedding_nim_endpoint: Optional[str] = None,
    encoding_format: Optional[str] = None,
    input_type: Optional[str] = None,
    truncate: Optional[str] = None,
):
    """
    Creates text embeddings using the provided configuration.
    Parameters provided as None will use the default values from EmbedExtractionsSchema.
    """
    task_config = {}

    # Build configuration parameters only if provided; defaults come from EmbedExtractionsSchema.
    config_kwargs = {
        "embedding_model": embedding_model,
        "embedding_nim_endpoint": embedding_nim_endpoint,
        "encoding_format": encoding_format,
        "input_type": input_type,
        "truncate": truncate,
    }
    # Remove any keys with a None value.
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    config_kwargs["api_key"] = api_key

    transform_config = EmbedExtractionsSchema(**config_kwargs)

    result, _ = transform_create_text_embeddings_internal(
        df_transform_ledger=df_ledger,
        task_config=task_config,
        transform_config=transform_config,
        execution_trace_log=None,
    )

    return result


@unified_exception_handler
def transform_image_create_vlm_caption(
    *,
    df_ledger: pd.DataFrame,
    api_key: Optional[str] = None,
    prompt: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extracts captions for image content in the given DataFrame using the VLM model API.
    The function constructs the task configuration from the provided parameters,
    omitting any parameters that are None, and then invokes the internal caption extraction
    routine.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing the image content. Each row is expected to have a 'metadata' column
        with image data that will be captioned.
    api_key : Optional[str], default=None
        API key for authentication with the VLM endpoint. If None, this parameter is omitted.
    prompt : Optional[str], default=None
        The text prompt to guide caption generation. If None, this parameter is omitted.
    endpoint_url : Optional[str], default=None
        The URL of the VLM model HTTP endpoint. If None, this parameter is omitted.
    model_name : Optional[str], default=None
        The model name to be used for caption generation. If None, this parameter is omitted.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with generated captions inserted into the 'image_metadata.caption'
        field within the 'metadata' column.

    Raises
    ------
    Exception
        Propagates any exceptions encountered during the caption extraction process, wrapped
        with additional context.
    """

    # Build the task configuration and filter out any keys with None values.
    task_config: Dict[str, Optional[str]] = {
        "api_key": api_key,
        "prompt": prompt,
        "endpoint_url": endpoint_url,
        "model_name": model_name,
    }
    filtered_task_config: Dict[str, str] = {k: v for k, v in task_config.items() if v is not None}

    # Create the transformation configuration using the filtered task configuration.
    transform_config = ImageCaptionExtractionSchema(**filtered_task_config)

    return transform_image_create_vlm_caption_internal(
        df_transform_ledger=df_ledger,
        task_config=filtered_task_config,
        transform_config=transform_config,
        execution_trace_log=None,
    )


@unified_exception_handler
def transform_text_split_and_tokenize(
    *,
    df_ledger: pd.DataFrame,
    tokenizer: str,
    chunk_size: int,
    chunk_overlap: int,
    split_source_types: Optional[List[str]] = None,
    hugging_face_access_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Transform and tokenize text by splitting documents into smaller chunks.

    This function prepares the configuration parameters for text splitting and then delegates
    the transformation to an internal function that performs the actual split and tokenization.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing the ledger of documents to be processed. Each document should include
        columns such as 'document_type' and 'metadata' (with nested content information).
    tokenizer : str
        The tokenizer identifier or path to use for tokenization.
    chunk_size : int
        Maximum number of tokens per chunk.
    chunk_overlap : int
        Number of tokens to overlap between consecutive chunks.
    split_source_types : Optional[List[str]], optional
        List of source types to filter for text splitting. If None or empty, defaults to ["text"].
    hugging_face_access_token : Optional[str], optional
        Access token for Hugging Face authentication, if required.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the documents processed, where text documents have been split into smaller chunks.
    """
    if not split_source_types:
        split_source_types = ["text"]

    task_config: Dict[str, Any] = {
        "chunk_overlap": chunk_overlap,
        "chunk_size": chunk_size,
        "params": {
            "hf_access_token": hugging_face_access_token,
            "split_source_types": split_source_types,
        },
        "tokenizer": tokenizer,
    }

    transform_config: TextSplitterSchema = TextSplitterSchema(
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        tokenizer=tokenizer,
    )

    return transform_text_split_and_tokenize_internal(
        df_transform_ledger=df_ledger,
        task_config=task_config,
        transform_config=transform_config,
        execution_trace_log=None,
    )
