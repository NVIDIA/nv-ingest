# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Tuple, Optional, Iterable, List

import pandas as pd
from openai import OpenAI

from nv_ingest_api.internal.enums.common import ContentTypeEnum, StatusEnum, TaskTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import (
    InfoMessageMetadataSchema,
)
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Asynchronous Embedding Requests
# ------------------------------------------------------------------------------


def _make_async_request(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
) -> list:
    """
    Interacts directly with the NIM embedding service to calculate embeddings for a batch of prompts.

    Parameters
    ----------
    prompts : List[str]
        A list of prompt strings for which embeddings are to be calculated.
    api_key : str
        API key for authentication with the embedding service.
    embedding_nim_endpoint : str
        Base URL for the NIM embedding service.
    embedding_model : str
        The model to use for generating embeddings.
    encoding_format : str
        The desired encoding format.
    input_type : str
        The type of input data.
    truncate : str
        Truncation setting for the input data.
    filter_errors : bool
        Flag indicating whether to filter errors in the response.

    Returns
    -------
    list
        A dictionary with keys "embedding" (the embedding results) and "info_msg" (any error info).

    Raises
    ------
    RuntimeError
        If an error occurs during the embedding request, with an info message attached.
    """
    response = {}

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=embedding_nim_endpoint,
        )

        resp = client.embeddings.create(
            input=prompts,
            model=embedding_model,
            encoding_format=encoding_format,
            extra_body={"input_type": input_type, "truncate": truncate},
        )

        response["embedding"] = resp.data
        response["info_msg"] = None

    except Exception as err:
        info_msg = {
            "task": TaskTypeEnum.EMBED.value,
            "status": StatusEnum.ERROR.value,
            "message": f"Embedding error: {err}",
            "filter": filter_errors,
        }
        validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).model_dump()

        response["embedding"] = [None] * len(prompts)
        response["info_msg"] = validated_info_msg

        raise RuntimeError(f"Embedding error occurred. Info message: {validated_info_msg}") from err

    return response


def _async_request_handler(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
) -> List[dict]:
    """
    Gathers calculated embedding results from the NIM embedding service concurrently.

    Parameters
    ----------
    prompts : List[str]
        A list of prompt batches.
    api_key : str
        API key for authentication.
    embedding_nim_endpoint : str
        Base URL for the NIM embedding service.
    embedding_model : str
        The model to use for generating embeddings.
    encoding_format : str
        The desired encoding format.
    input_type : str
        The type of input data.
    truncate : str
        Truncation setting for the input data.
    filter_errors : bool
        Flag indicating whether to filter errors in the response.

    Returns
    -------
    List[dict]
        A list of response dictionaries from the embedding service.
    """
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _make_async_request,
                prompts=prompt_batch,
                api_key=api_key,
                embedding_nim_endpoint=embedding_nim_endpoint,
                embedding_model=embedding_model,
                encoding_format=encoding_format,
                input_type=input_type,
                truncate=truncate,
                filter_errors=filter_errors,
            )
            for prompt_batch in prompts
        ]
        results = [future.result() for future in futures]

    return results


def _async_runner(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
) -> dict:
    """
    Concurrently launches all NIM embedding requests and flattens the results.

    Parameters
    ----------
    prompts : List[str]
        A list of prompt batches.
    api_key : str
        API key for authentication.
    embedding_nim_endpoint : str
        Base URL for the NIM embedding service.
    embedding_model : str
        The model to use for generating embeddings.
    encoding_format : str
        The desired encoding format.
    input_type : str
        The type of input data.
    truncate : str
        Truncation setting for the input data.
    filter_errors : bool
        Flag indicating whether to filter errors in the response.

    Returns
    -------
    dict
        A dictionary with keys "embeddings" (flattened embedding results) and "info_msgs" (error messages).
    """
    results = _async_request_handler(
        prompts,
        api_key,
        embedding_nim_endpoint,
        embedding_model,
        encoding_format,
        input_type,
        truncate,
        filter_errors,
    )

    flat_results = {"embeddings": [], "info_msgs": []}
    for batch_dict in results:
        info_msg = batch_dict["info_msg"]
        for embedding in batch_dict["embedding"]:
            if not isinstance(embedding, list):
                if embedding is not None:
                    flat_results["embeddings"].append(embedding.embedding)
                else:
                    flat_results["embeddings"].append(embedding)
            else:
                flat_results["embeddings"].append(embedding)
            flat_results["info_msgs"].append(info_msg)

    return flat_results


# ------------------------------------------------------------------------------
# Pandas UDFs for Content Extraction
# ------------------------------------------------------------------------------


def _add_embeddings(row, embeddings, info_msgs):
    """
    Updates a DataFrame row with embedding data and associated error info.
    Ensures the 'embedding' field is always present, even if None.

    Parameters
    ----------
    row : pandas.Series
        A row of the DataFrame.
    embeddings : dict
        Dictionary mapping row indices to embeddings.
    info_msgs : dict
        Dictionary mapping row indices to info message dicts.

    Returns
    -------
    pandas.Series
        The updated row with 'embedding', 'info_message_metadata', and
        '_contains_embeddings' appropriately set.
    """
    embedding = embeddings.get(row.name, None)
    info_msg = info_msgs.get(row.name, None)

    # Always set embedding, even if None
    row["metadata"]["embedding"] = embedding

    if info_msg:
        row["metadata"]["info_message_metadata"] = info_msg
        row["document_type"] = ContentTypeEnum.INFO_MSG
        row["_contains_embeddings"] = False
    else:
        row["_contains_embeddings"] = embedding is not None

    return row


def _get_pandas_text_content(row):
    """
    Extracts text content from a DataFrame row.

    Parameters
    ----------
    row : pandas.Series
        A row containing the 'content' key.

    Returns
    -------
    str
        The text content from the row.
    """
    return row["content"]


def _get_pandas_table_content(row):
    """
    Extracts table/chart content from a DataFrame row.

    Parameters
    ----------
    row : pandas.Series
        A row containing 'table_metadata' with 'table_content'.

    Returns
    -------
    str
        The table/chart content from the row.
    """
    return row.get("table_metadata", {}).get("table_content")


def _get_pandas_image_content(row):
    """
    Extracts image caption content from a DataFrame row.

    Parameters
    ----------
    row : pandas.Series
        A row containing 'image_metadata' with 'caption'.

    Returns
    -------
    str
        The image caption from the row.
    """
    return row.get("image_metadata", {}).get("caption")


def _get_pandas_audio_content(row):
    """
    A pandas UDF used to select extracted audio transcription to be used to create embeddings.
    """
    return row.get("audio_metadata", {}).get("audio_transcript")


# ------------------------------------------------------------------------------
# Batch Processing Utilities
# ------------------------------------------------------------------------------


def _batch_generator(iterable: Iterable, batch_size: int = 10):
    """
    Yields batches of a specified size from an iterable.

    Parameters
    ----------
    iterable : Iterable
        The iterable to batch.
    batch_size : int, optional
        The size of each batch (default is 10).

    Yields
    ------
    list
        A batch of items from the iterable.
    """
    iter_len = len(iterable)
    for idx in range(0, iter_len, batch_size):
        yield iterable[idx : min(idx + batch_size, iter_len)]


def _generate_batches(prompts: List[str], batch_size: int = 100) -> List[str]:
    """
    Splits a list of prompts into batches.

    Parameters
    ----------
    prompts : List[str]
        The list of prompt strings.
    batch_size : int, optional
        The desired batch size (default is 100).

    Returns
    -------
    List[List[str]]
        A list of batches, each containing a subset of the prompts.
    """
    return [batch for batch in _batch_generator(prompts, batch_size)]


# ------------------------------------------------------------------------------
# DataFrame Concatenation Utility
# ------------------------------------------------------------------------------


def _concatenate_extractions_pandas(
    base_df: pd.DataFrame, dataframes: List[pd.DataFrame], masks: List[pd.Series]
) -> pd.DataFrame:
    """
    Concatenates processed DataFrame rows (with embeddings) with unprocessed rows from the base DataFrame.

    Parameters
    ----------
    base_df : pd.DataFrame
        The original DataFrame.
    dataframes : List[pd.DataFrame]
        List of DataFrames that have been enriched with embeddings.
    masks : List[pd.Series]
        List of boolean masks indicating the rows that were processed.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame with embeddings applied where available.
    """
    unified_mask = pd.Series(False, index=base_df.index)
    for mask in masks:
        unified_mask = unified_mask | mask

    df_no_text = base_df.loc[~unified_mask].copy()
    df_no_text["_contains_embeddings"] = False

    dataframes.append(df_no_text)
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True).reset_index(drop=True)
    return combined_df


# ------------------------------------------------------------------------------
# Embedding Extraction Pipeline
# ------------------------------------------------------------------------------


def transform_create_text_embeddings_internal(
    df_transform_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    transform_config: TextEmbeddingSchema = TextEmbeddingSchema(),
    execution_trace_log: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generates text embeddings for supported content types (TEXT, STRUCTURED, IMAGE, AUDIO)
    from a pandas DataFrame using asynchronous requests.

    This function ensures that even if the extracted content is empty or None,
    the embedding field is explicitly created and set to None.

    Parameters
    ----------
    df_transform_ledger : pd.DataFrame
        The DataFrame containing content for embedding extraction.
    task_config : Dict[str, Any]
        Dictionary containing task properties (e.g., filter error flag).
    transform_config : TextEmbeddingSchema, optional
        Validated configuration for text embedding extraction.
    execution_trace_log : Optional[Dict], optional
        Optional trace information for debugging or logging (default is None).

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing:
            - The updated DataFrame with embeddings applied.
            - A dictionary with trace information.
    """
    api_key = task_config.get("api_key") or transform_config.api_key
    endpoint_url = task_config.get("endpoint_url") or transform_config.embedding_nim_endpoint
    model_name = task_config.get("model_name") or transform_config.embedding_model

    if execution_trace_log is None:
        execution_trace_log = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df_transform_ledger.empty:
        return df_transform_ledger, {"trace_info": execution_trace_log}

    embedding_dataframes = []
    content_masks = []

    pandas_content_extractor = {
        ContentTypeEnum.TEXT: _get_pandas_text_content,
        ContentTypeEnum.STRUCTURED: _get_pandas_table_content,
        ContentTypeEnum.IMAGE: _get_pandas_image_content,
        ContentTypeEnum.AUDIO: _get_pandas_audio_content,
        ContentTypeEnum.VIDEO: lambda x: None,  # Not supported yet.
    }

    def _content_type_getter(row):
        return row["content_metadata"]["type"]

    for content_type, content_getter in pandas_content_extractor.items():
        if not content_getter:
            logger.debug(f"Skipping unsupported content type: {content_type}")
            continue

        # Get rows matching the content type
        content_mask = df_transform_ledger["metadata"].apply(_content_type_getter) == content_type.value
        if not content_mask.any():
            continue

        # Always include all content_mask rows and prepare them
        df_content = df_transform_ledger.loc[content_mask].copy().reset_index(drop=True)

        # Extract content and normalize empty or non-str to None
        extracted_content = (
            df_content["metadata"]
            .apply(content_getter)
            .apply(lambda x: x.strip() if isinstance(x, str) and x.strip() else None)
        )
        df_content["_content"] = extracted_content

        # Prepare batches for only valid (non-None) content
        valid_content_mask = df_content["_content"].notna()
        if valid_content_mask.any():
            filtered_content_batches = _generate_batches(
                df_content.loc[valid_content_mask, "_content"].tolist(), batch_size=transform_config.batch_size
            )
            content_embeddings = _async_runner(
                filtered_content_batches,
                api_key,
                endpoint_url,
                model_name,
                transform_config.encoding_format,
                transform_config.input_type,
                transform_config.truncate,
                False,
            )
            # Build a simple row index -> embedding map
            embeddings_dict = dict(
                zip(df_content.loc[valid_content_mask].index, content_embeddings.get("embeddings", []))
            )
            info_msgs_dict = dict(
                zip(df_content.loc[valid_content_mask].index, content_embeddings.get("info_msgs", []))
            )
        else:
            embeddings_dict = {}
            info_msgs_dict = {}

        # Apply embeddings or None to all rows
        df_content = df_content.apply(_add_embeddings, embeddings=embeddings_dict, info_msgs=info_msgs_dict, axis=1)

        embedding_dataframes.append(df_content)
        content_masks.append(content_mask)

    combined_df = _concatenate_extractions_pandas(df_transform_ledger, embedding_dataframes, content_masks)
    return combined_df, {"trace_info": execution_trace_log}
