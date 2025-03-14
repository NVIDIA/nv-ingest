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
    base64_images: List[str], prompt: str, api_key: str, endpoint_url: str, model_name: str
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

        # Create the inference client using the VLMModelInterface.
        nim_client = create_inference_client(
            model_interface=VLMModelInterface(),
            endpoints=(None, endpoint_url),
            auth_token=api_key,
            infer_protocol="http",
        )

        logger.debug(f"Calling VLM endpoint: {endpoint_url} with model: {model_name}")
        # Perform inference to generate captions.
        captions: List[str] = nim_client.infer(data, model_name=model_name)
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
    endpoint_url: str = task_config.get("endpoint_url") or transform_config.endpoint_url
    model_name: str = task_config.get("model_name") or transform_config.model_name

    # Create a mask for rows where the content type is "image".
    df_mask: pd.Series = df_transform_ledger["metadata"].apply(
        lambda meta: meta.get("content_metadata", {}).get("type") == "image"
    )

    # If no image rows exist, return the original DataFrame.
    if not df_mask.any():
        return df_transform_ledger

    # Collect base64-encoded images from the rows where the content type is "image".
    base64_images: List[str] = df_transform_ledger.loc[df_mask, "metadata"].apply(lambda meta: meta["content"]).tolist()

    # Generate captions for the collected images.
    captions: List[str] = _generate_captions(base64_images, prompt, api_key, endpoint_url, model_name)

    # Update the DataFrame: assign each generated caption to the corresponding row.
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
