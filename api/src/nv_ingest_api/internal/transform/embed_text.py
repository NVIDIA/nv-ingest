# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, Tuple, Optional, Iterable, List
from urllib.parse import urlparse

import glom
import pandas as pd

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.util.nim import infer_microservice


logger = logging.getLogger(__name__)

# Reduce SDK HTTP logging verbosity so request/response logs are not emitted
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


MULTI_MODAL_MODELS = ["llama-3.2-nemoretriever-1b-vlm-embed-v1"]


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
    modalities: Optional[List[str]] = None,
    dimensions: Optional[int] = None,
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
        # Normalize API key to avoid sending an empty bearer token via SDK internals
        _token = (api_key or "").strip()
        _api_key = _token if _token else "<no key provided>"

        resp = infer_microservice(
            prompts,
            embedding_model,
            embedding_endpoint=embedding_nim_endpoint,
            nvidia_api_key=_api_key,
            input_type=input_type,
            truncate=truncate,
            batch_size=8191,
            grpc="http" not in urlparse(embedding_nim_endpoint).scheme,
            input_names=["text"],
            output_names=["embeddings"],
            dtypes=["BYTES"],
        )

        response["embedding"] = resp
        response["info_msg"] = None

    except Exception as err:
        # Truncate error message to prevent memory blowup from large text content
        err_str = str(err)
        if len(err_str) > 500:
            truncated_err = err_str[:200] + "... [truncated to prevent memory blowup] ..." + err_str[-100:]
        else:
            truncated_err = err_str

        raise RuntimeError(f"Embedding error occurred: {truncated_err}") from err

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
    modalities: Optional[List[str]] = None,
    dimensions: Optional[int] = None,
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
    if modalities is None:
        modalities = [None] * len(prompts)

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
                modalities=modality_batch,
                dimensions=dimensions,
            )
            for prompt_batch, modality_batch in zip(prompts, modalities)
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
    modalities: Optional[List[str]] = None,
    dimensions: Optional[int] = None,
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
        modalities=modalities,
        dimensions=dimensions,
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


def _add_custom_embeddings(row, embeddings, result_target_field):
    """
    Updates a DataFrame row with embedding data and associated error info
    based on a user supplied custom content field.

    Parameters
    ----------
    row : pandas.Series
        A row of the DataFrame.
    embeddings : dict
        Dictionary mapping row indices to embeddings.
    result_target_field: str
        The field in custom_content to output the embeddings to

    Returns
    -------
    pandas.Series
        The updated row
    """
    embedding = embeddings.get(row.name, None)

    if embedding is not None:
        row["metadata"] = glom.assign(row["metadata"], "custom_content." + result_target_field, embedding, missing=dict)

    return row


def _format_image_input_string(image_b64: Optional[str]) -> str:
    if not image_b64:
        return
    return f"data:image/png;base64,{image_b64}"


def _format_text_image_pair_input_string(text: Optional[str], image_b64: Optional[str]) -> str:
    if (not text) or (not text.strip()) or (not image_b64):
        return
    return f"{text.strip()} {_format_image_input_string(image_b64)}"


def _get_pandas_text_content(row, modality="text"):
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


def _get_pandas_table_content(row, modality="text"):
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
    if modality == "text":
        content = row.get("table_metadata", {}).get("table_content")
    elif modality == "image":
        content = _format_image_input_string(row.get("content"))
    elif modality == "text_image":
        text = row.get("table_metadata", {}).get("table_content")
        image = row.get("content")
        content = _format_text_image_pair_input_string(text, image)

    return content


def _get_pandas_image_content(row, modality="text"):
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
    subtype = row.get("content_metadata", {}).get("subtype")
    if modality == "text":
        if subtype == "page_image":
            content = row.get("image_metadata", {}).get("text")
        else:
            content = row.get("image_metadata", {}).get("caption")
    elif modality == "image":
        content = _format_image_input_string(row.get("content"))
    elif modality == "text_image":
        if subtype == "page_image":
            text = row.get("image_metadata", {}).get("text")
        else:
            text = row.get("image_metadata", {}).get("caption")
        image = row.get("content")
        content = _format_text_image_pair_input_string(text, image)

    if subtype == "page_image":
        # A workaround to save memory for full page images.
        row["content"] = ""

    return content


def _get_pandas_audio_content(row, modality="text"):
    """
    A pandas UDF used to select extracted audio transcription to be used to create embeddings.
    """
    return row.get("audio_metadata", {}).get("audio_transcript")


def _get_pandas_custom_content(row, custom_content_field):
    custom_content = row.get("custom_content", {})
    content = glom.glom(custom_content, custom_content_field, default=None)
    if content is None:
        logger.warning(f"Custom content field: {custom_content_field} not found")
        return None

    try:
        return str(content)
    except (TypeError, ValueError):
        logger.warning(f"Cannot convert custom content field: {custom_content_field} to string")
        return None


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


def does_model_support_multimodal_embeddings(model: str) -> bool:
    """
    Checks if a given model supports multi-modal embeddings.

    Parameters
    ----------
    model : str
        The name of the model.

    Returns
    -------
    bool
        True if the model supports multi-modal embeddings, False otherwise.
    """
    return model in MULTI_MODAL_MODELS


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
    custom_content_field = task_config.get("custom_content_field") or transform_config.custom_content_field
    dimensions = task_config.get("dimensions") or transform_config.dimensions

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
    task_type_to_modality = {
        ContentTypeEnum.TEXT: task_config.get("text_elements_modality") or transform_config.text_elements_modality,
        ContentTypeEnum.STRUCTURED: (
            task_config.get("structured_elements_modality") or transform_config.structured_elements_modality
        ),
        ContentTypeEnum.IMAGE: task_config.get("image_elements_modality") or transform_config.image_elements_modality,
        ContentTypeEnum.AUDIO: task_config.get("audio_elements_modality") or transform_config.audio_elements_modality,
        ContentTypeEnum.VIDEO: lambda x: None,  # Not supported yet.
    }

    def _content_type_getter(row):
        return row["content_metadata"]["type"]

    for content_type, content_getter in pandas_content_extractor.items():
        if not content_getter:
            logger.warning(f"Skipping text_embedding generation for unsupported content type: {content_type}")
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
            .apply(partial(content_getter, modality=task_type_to_modality[content_type]))
            .apply(lambda x: x.strip() if isinstance(x, str) and x.strip() else None)
        )
        df_content["_content"] = extracted_content

        # Prepare batches for only valid (non-None) content
        valid_content_mask = df_content["_content"].notna()
        if valid_content_mask.any():
            filtered_content_list = df_content.loc[valid_content_mask, "_content"].tolist()
            filtered_content_batches = _generate_batches(filtered_content_list, batch_size=transform_config.batch_size)

            if model_name in MULTI_MODAL_MODELS:
                modality_list = [task_type_to_modality[content_type]] * len(filtered_content_list)
                modality_batches = _generate_batches(modality_list, batch_size=transform_config.batch_size)
            else:
                modality_batches = None

            content_embeddings = _async_runner(
                filtered_content_batches,
                api_key,
                endpoint_url,
                model_name,
                transform_config.encoding_format,
                transform_config.input_type,
                transform_config.truncate,
                False,
                modalities=modality_batches,
                dimensions=dimensions,
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

    # Embed custom content
    if custom_content_field is not None:
        result_target_field = task_config.get("result_target_field") or custom_content_field + "_embedding"

        extracted_custom_content = (
            combined_df["metadata"]
            .apply(partial(_get_pandas_custom_content, custom_content_field=custom_content_field))
            .apply(lambda x: x.strip() if isinstance(x, str) and x.strip() else None)
        )

        valid_custom_content_mask = extracted_custom_content.notna()
        if valid_custom_content_mask.any():
            custom_content_list = extracted_custom_content[valid_custom_content_mask].to_list()
            custom_content_batches = _generate_batches(custom_content_list, batch_size=transform_config.batch_size)

            custom_content_embeddings = _async_runner(
                custom_content_batches,
                api_key,
                endpoint_url,
                model_name,
                transform_config.encoding_format,
                transform_config.input_type,
                transform_config.truncate,
                False,
                dimensions=dimensions,
            )
            custom_embeddings_dict = dict(
                zip(
                    extracted_custom_content.loc[valid_custom_content_mask].index,
                    custom_content_embeddings.get("embeddings", []),
                )
            )
        else:
            custom_embeddings_dict = {}

        combined_df = combined_df.apply(
            _add_custom_embeddings, embeddings=custom_embeddings_dict, result_target_field=result_target_field, axis=1
        )

    return combined_df, {"trace_info": execution_trace_log}
