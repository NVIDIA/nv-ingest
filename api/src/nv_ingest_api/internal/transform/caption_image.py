# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel

from nv_ingest_api.internal.primitives.nim.model_interface.vlm import VLMModelInterface
from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler
from nv_ingest_api.util.image_processing import scale_image_to_encoding_size
from nv_ingest_api.util.nim import create_inference_client

logger = logging.getLogger(__name__)

_MAX_CONTEXT_TEXT_CHARS = 4096


def _gather_context_text_for_image(
    image_meta: Dict[str, Any],
    page_text_map: Dict[int, List[str]],
    max_chars: int,
) -> str:
    """
    Gather surrounding OCR text for an image to provide as VLM prompt context.

    Parameters
    ----------
    image_meta : dict
        The full metadata dict for the image row.
    page_text_map : dict
        Mapping of page number -> list of text strings, precomputed from the
        DataFrame's text rows.
    max_chars : int
        Maximum number of characters to return. Will be clamped to
        ``_MAX_CONTEXT_TEXT_CHARS``.

    Returns
    -------
    str
        Surrounding text (possibly truncated), or empty string if none found.
    """
    effective_max = min(max_chars, _MAX_CONTEXT_TEXT_CHARS)
    content_meta = image_meta.get("content_metadata", {})
    page_num = content_meta.get("page_number", -1)
    page_texts = page_text_map.get(page_num, [])
    if page_texts:
        combined = " ".join(page_texts)
        return combined[:effective_max]

    return ""


def _build_prompt_with_context(base_prompt: str, context_text: str) -> str:
    """
    Prepend surrounding-text context to the base VLM prompt.

    If *context_text* is empty the *base_prompt* is returned unchanged.
    """
    if not context_text:
        return base_prompt
    return f"Text near this image:\n---\n{context_text}\n---\n\n{base_prompt}"


def _build_page_text_map(df: pd.DataFrame) -> Dict[int, List[str]]:
    """
    Build a mapping of page number -> list of text content strings from text
    rows in the DataFrame.  Computed once per call to avoid O(images * rows).
    """
    page_text_map: Dict[int, List[str]] = {}
    for _, row in df.iterrows():
        meta = row.get("metadata")
        if meta is None:
            continue
        cm = meta.get("content_metadata", {})
        if cm.get("type") != "text":
            continue
        content = meta.get("content", "")
        if not content:
            continue
        page_num = cm.get("page_number", -1)
        page_text_map.setdefault(page_num, []).append(content)
    return page_text_map


def _prepare_dataframes_mod(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Prepares and returns three DataFrame-related objects from the input DataFrame.

    The function performs the following:
      1. Checks if the DataFrame is empty or if the "document_type" column is missing.
         In such a case, returns the original DataFrame, an empty DataFrame, and an empty boolean Series.
      2. Otherwise, it creates a boolean Series identifying rows where "document_type" equals IMAGE.
      3. Extracts a DataFrame containing only those rows.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame that should contain a "document_type" column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
        A tuple containing:
          - The original DataFrame.
          - A DataFrame filtered to rows where "document_type" is IMAGE.
          - A boolean Series indicating which rows in the original DataFrame are IMAGE rows.
    """
    try:
        if df.empty or "document_type" not in df.columns:
            return df, pd.DataFrame(), pd.Series(dtype=bool)

        bool_index: pd.Series = df["document_type"] == ContentTypeEnum.IMAGE
        df_matched: pd.DataFrame = df.loc[bool_index]

        return df, df_matched, bool_index

    except Exception as e:
        err_msg = f"_prepare_dataframes_mod: Error preparing dataframes. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def _generate_captions(
    base64_images: List[str],
    prompt: str,
    system_prompt: Optional[str],
    api_key: str,
    endpoint_url: str,
    model_name: str,
    temperature: float = 1.0,
) -> List[str]:
    """
    Generates captions for a list of base64-encoded PNG images using the VLM model API.

    This function performs the following steps:
      1. Scales each image to meet encoding size requirements using `scale_image_to_encoding_size`.
      2. Constructs the input payload containing the scaled images and the provided prompt.
      3. Creates an inference client using the VLMModelInterface.
      4. Calls the client's infer method to obtain a list of captions corresponding to the images.

    Parameters
    ----------
    base64_images : List[str]
        List of base64-encoded PNG image strings.
    prompt : str
        Text prompt to guide caption generation.
    api_key : str
        API key for authenticating with the VLM endpoint.
    endpoint_url : str
        URL of the VLM model HTTP endpoint.
    model_name : str
        The name of the model to use for inference.

    Returns
    -------
    List[str]
        A list of generated captions, each corresponding to an input image.

    Raises
    ------
    Exception
        Propagates any exception encountered during caption generation, with added context.
    """
    try:
        # Scale each image to ensure it meets encoding size requirements.
        scaled_images: List[str] = [scale_image_to_encoding_size(b64)[0] for b64 in base64_images]

        # Build the input payload for the VLM model.
        data: Dict[str, Any] = {
            "base64_images": scaled_images,
            "prompt": prompt,
        }
        if system_prompt:
            data["system_prompt"] = system_prompt

        # Create the inference client using the VLMModelInterface.
        nim_client = create_inference_client(
            model_interface=VLMModelInterface(),
            endpoints=(None, endpoint_url),
            auth_token=api_key,
            infer_protocol="http",
        )

        # Perform inference to generate captions.
        captions: List[str] = nim_client.infer(data, model_name=model_name, temperature=temperature)
        return captions

    except Exception as e:
        err_msg = f"_generate_captions: Error generating captions: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


@unified_exception_handler
def transform_image_create_vlm_caption_internal(
    df_transform_ledger: pd.DataFrame,
    task_config: Union[BaseModel, Dict[str, Any]],
    transform_config: Any,
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Extracts and adds captions for image content in a DataFrame using the VLM model API.

    This function updates the 'metadata' column for rows where the content type is "image".
    It uses configuration values from task_config (or falls back to transform_config defaults)
    to determine the API key, prompt, endpoint URL, and model name for caption generation.
    The generated captions are added under the 'image_metadata.caption' key in the metadata.

    Parameters
    ----------
    df_transform_ledger : pd.DataFrame
        The input DataFrame containing image data. Each row must have a 'metadata' column
        with at least the 'content' and 'content_metadata' keys.
    task_config : Union[BaseModel, Dict[str, Any]]
        Configuration parameters for caption extraction. If provided as a Pydantic model,
        it will be converted to a dictionary. Expected keys include "api_key", "prompt",
        "endpoint_url", and "model_name".
    transform_config : Any
        A configuration object providing default values for caption extraction. It should have
        attributes: api_key, prompt, endpoint_url, and model_name.
    execution_trace_log : Optional[Dict[str, Any]], default=None
        Optional trace information for debugging or logging purposes.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with generated captions added to the 'image_metadata.caption' field
        within the 'metadata' column for each image row.

    Raises
    ------
    Exception
        Propagates any exception encountered during the caption extraction process, with added context.
    """

    _ = execution_trace_log  # Unused variable; placeholder to prevent linter warnings.

    logger.debug("Attempting to caption image content")

    # Convert task_config to dictionary if it is a Pydantic model.
    if isinstance(task_config, BaseModel):
        task_config = task_config.model_dump()

    # Retrieve configuration values with fallback to transform_config defaults.
    api_key: str = task_config.get("api_key") or transform_config.api_key
    prompt: str = task_config.get("prompt") or transform_config.prompt
    system_prompt: str = task_config.get("system_prompt") or transform_config.system_prompt
    endpoint_url: str = task_config.get("endpoint_url") or transform_config.endpoint_url
    model_name: str = task_config.get("model_name") or transform_config.model_name

    # Context text: task config overrides pipeline default.
    context_text_max_chars: int = task_config.get("context_text_max_chars") or getattr(
        transform_config, "context_text_max_chars", 0
    )

    # Temperature: task config overrides pipeline default.
    temperature: float = task_config.get("temperature") or getattr(transform_config, "temperature", 1.0)

    # Create a mask for rows where the content type is "image".
    df_mask: pd.Series = df_transform_ledger["metadata"].apply(
        lambda meta: meta.get("content_metadata", {}).get("type") == "image"
    )

    # If no image rows exist, return the original DataFrame.
    if not df_mask.any():
        return df_transform_ledger

    if context_text_max_chars and context_text_max_chars > 0:
        page_text_map = _build_page_text_map(df_transform_ledger)

        for idx in df_transform_ledger.loc[df_mask].index:
            meta: Dict[str, Any] = df_transform_ledger.at[idx, "metadata"]
            base64_image: str = meta["content"]
            context_text = _gather_context_text_for_image(meta, page_text_map, context_text_max_chars)
            enriched_prompt = _build_prompt_with_context(prompt, context_text)

            captions: List[str] = _generate_captions(
                [base64_image],
                enriched_prompt,
                system_prompt,
                api_key,
                endpoint_url,
                model_name,
                temperature=temperature,
            )

            image_meta: Dict[str, Any] = meta.get("image_metadata", {})
            image_meta["caption"] = captions[0] if captions else ""
            meta["image_metadata"] = image_meta
            df_transform_ledger.at[idx, "metadata"] = meta
    else:
        base64_images: List[str] = (
            df_transform_ledger.loc[df_mask, "metadata"].apply(lambda meta: meta["content"]).tolist()
        )

        captions: List[str] = _generate_captions(
            base64_images,
            prompt,
            system_prompt,
            api_key,
            endpoint_url,
            model_name,
            temperature=temperature,
        )

        for idx, caption in zip(df_transform_ledger.loc[df_mask].index, captions):
            meta: Dict[str, Any] = df_transform_ledger.at[idx, "metadata"]
            image_meta: Dict[str, Any] = meta.get("image_metadata", {})
            image_meta["caption"] = caption
            meta["image_metadata"] = image_meta
            df_transform_ledger.at[idx, "metadata"] = meta

    logger.debug("Image content captioning complete")
    result, execution_trace_log = df_transform_ledger, {}
    _ = execution_trace_log  # Unused variable; placeholder to prevent linter warnings.

    return result
