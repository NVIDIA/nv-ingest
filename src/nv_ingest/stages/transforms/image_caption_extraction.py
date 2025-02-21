# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import partial
from typing import Any, List
from typing import Dict
from typing import Optional
from typing import Tuple

import pandas as pd
from pydantic import BaseModel
from morpheus.config import Config

from nv_ingest.schemas.image_caption_extraction_schema import ImageCaptionExtractionSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.image_processing.transforms import scale_image_to_encoding_size
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.nim.vlm import VLMModelInterface

logger = logging.getLogger(__name__)

MODULE_NAME = "image_caption_extraction"
MODULE_NAMESPACE = "nv_ingest"


def _prepare_dataframes_mod(df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Prepares and returns the full DataFrame, a DataFrame containing only image rows,
    and a boolean Series indicating image rows.
    """
    try:
        if df.empty or "document_type" not in df.columns:
            return df, pd.DataFrame(), pd.Series(dtype=bool)

        bool_index = df["document_type"] == ContentTypeEnum.IMAGE
        df_matched = df.loc[bool_index]

        return df, df_matched, bool_index

    except Exception as e:
        err_msg = f"_prepare_dataframes_mod: Error preparing dataframes. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def _generate_captions(
    base64_images: List[str], prompt: str, api_key: str, endpoint_url: str, model_name: str
) -> List[str]:
    """
    Sends a list of base64-encoded PNG images to the VLM model API using the NimClient,
    which is initialized with the VLMModelInterface, and retrieves the generated captions.

    Parameters
    ----------
    base64_images : List[str]
        List of base64-encoded PNG image strings.
    prompt : str
        Text prompt to guide caption generation.
    api_key : str
        API key for authentication with the VLM endpoint.
    endpoint_url : str
        URL of the VLM model HTTP endpoint.
    model_name : str
        The model name to use in the payload.

    Returns
    -------
    List[str]
        A list of generated captions corresponding to each image.
    """
    try:
        # Ensure each image is within acceptable encoding limits.
        scaled_images = []
        for b64 in base64_images:
            scaled_b64, _ = scale_image_to_encoding_size(b64)
            scaled_images.append(scaled_b64)

        # Build the input data for our VLM model interface.
        data = {
            "base64_images": scaled_images,
            "prompt": prompt,
        }

        # Instantiate the NimClient with our VLMModelInterface.
        nim_client = create_inference_client(
            model_interface=VLMModelInterface(),
            endpoints=(None, endpoint_url),
            auth_token=api_key,
            infer_protocol="http",
        )

        logger.debug(f"Calling: {endpoint_url} with model: {model_name}")
        # Call the infer method which handles batching and returns a list of captions.
        captions = nim_client.infer(data, model_name=model_name)
        return captions

    except Exception as e:
        err_msg = f"_generate_captions: Error generating captions: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def caption_extract_stage(
    df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Extracts captions for image content in the DataFrame using the VLM model API via VLMModelInterface.
    Updates the 'metadata' column by adding the generated captions under 'image_metadata.caption'.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing image data in 'metadata.content'.
    validated_config : Any
        A configuration schema object containing settings for caption extraction.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with generated captions in the 'metadata' column's 'image_metadata.caption' field.

    Raises
    ------
    Exception
        If there is an error during the caption extraction process.
    """
    try:
        logger.debug("Attempting to caption image content")

        if isinstance(task_props, BaseModel):
            task_props = task_props.model_dump()

        api_key = task_props.get("api_key") or validated_config.api_key
        prompt = task_props.get("prompt") or validated_config.prompt
        endpoint_url = task_props.get("endpoint_url") or validated_config.endpoint_url
        model_name = task_props.get("model_name") or validated_config.model_name

        # Create a mask for rows where the document type is IMAGE.
        df_mask = df["metadata"].apply(lambda meta: meta.get("content_metadata", {}).get("type") == "image")

        if not df_mask.any():
            return df

        # Collect all base64 images from the rows where the document type is IMAGE.
        base64_images = df.loc[df_mask, "metadata"].apply(lambda meta: meta["content"]).tolist()

        # Generate captions for all images using the new VLMModelInterface.
        captions = _generate_captions(base64_images, prompt, api_key, endpoint_url, model_name)

        # Update the DataFrame: for each image row, assign the corresponding caption.
        # (Assuming that the order of captions matches the order of images in base64_images.)
        for idx, caption in zip(df.loc[df_mask].index, captions):
            meta = df.at[idx, "metadata"]
            # Update or add the 'image_metadata' dict with the generated caption.
            image_meta = meta.get("image_metadata", {})
            image_meta["caption"] = caption
            meta["image_metadata"] = image_meta
            df.at[idx, "metadata"] = meta

        logger.debug("Image content captioning complete")
        return df

    except Exception as e:
        err_msg = f"caption_extract_stage: Error extracting captions. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def generate_caption_extraction_stage(
    c: Config,
    caption_config: Dict[str, Any],
    task: str = "caption",
    task_desc: str = "caption_extraction",
    pe_count: int = 8,
):
    """
    Generates a caption extraction stage with the specified configuration.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    caption_config : dict
        Configuration parameters for caption extraction.
    task : str, optional
        The task name to match for the stage worker function, by default "caption".
    task_desc : str, optional
        A descriptor to be used in latency tracing, by default "caption_extraction".
    pe_count : int, optional
        Number of processing elements to use, by default 8.

    Returns
    -------
    MultiProcessingBaseStage
        The generated caption extraction stage.

    Raises
    ------
    ValueError
        If an error occurs during stage generation.
    """
    try:
        validated_config = ImageCaptionExtractionSchema(**caption_config)
        _wrapped_caption_extract = partial(caption_extract_stage, validated_config=validated_config)

        logger.debug(f"Generating caption extraction stage with {pe_count} processing elements. Task: {task}")

        return MultiProcessingBaseStage(
            c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_caption_extract
        )

    except Exception as e:
        err_msg = f"generate_caption_extraction_stage: Error generating caption extraction stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
