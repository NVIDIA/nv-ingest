# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import logging
from functools import partial
from typing import Any, Optional
from typing import Dict
from typing import Tuple

import pandas as pd
import requests
from PIL import Image
from morpheus.config import Config

from nv_ingest.schemas.image_caption_extraction_schema import ImageCaptionExtractionSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.image_processing.transforms import scale_image_to_encoding_size
from nv_ingest.util.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)

MODULE_NAME = "image_caption_extraction"
MODULE_NAMESPACE = "nv_ingest"


def _prepare_dataframes_mod(df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if df.empty or "document_type" not in df.columns:
        return df, pd.DataFrame(), pd.Series(dtype=bool)

    bool_index = df["document_type"] == ContentTypeEnum.IMAGE
    df_matched = df.loc[bool_index]

    return df, df_matched, bool_index


def _generate_captions(base64_image: str, prompt: str, api_key: str, endpoint_url: str) -> str:
    """
    Sends a base64-encoded PNG image to the NVIDIA LLaMA model API and retrieves the generated caption.

    Parameters
    ----------
    base64_image : str
        Base64-encoded PNG image string.
    api_key : str
        API key for authentication with the NVIDIA model endpoint.

    Returns
    -------
    str
        Generated caption for the image or an error message.
    """
    stream = False  # Set to False for non-streaming response

    # Ensure the base64 image size is within acceptable limits
    base64_image = scale_image_to_encoding_size(base64_image)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    # Payload for the request
    payload = {
        "model": 'meta/llama-3.2-90b-vision-instruct',
        "messages": [
            {
                "role": "user",
                "content": f'{prompt} <img src="data:image/png;base64,{base64_image}" />'
            }
        ],
        "max_tokens": 512,
        "temperature": 1.00,
        "top_p": 1.00,
        "stream": stream
    }

    try:
        response = requests.post(endpoint_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        if stream:
            result = []
            for line in response.iter_lines():
                if line:
                    result.append(line.decode("utf-8"))
            return "\n".join(result)
        else:
            response_data = response.json()
            return response_data.get('choices', [{}])[0].get('message', {}).get('content', 'No caption returned')
    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating caption: {e}")
        raise


def caption_extract_stage(df: pd.DataFrame,
                          task_props: Dict[str, Any],
                          validated_config: Any,
                          trace_info: Optional[Dict[str, Any]] = None
                          ) -> pd.DataFrame:
    """
    Extracts captions for image content in the DataFrame using an external NVIDIA API.
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
    logger.debug("Attempting to caption image content")

    # Ensure the validated configuration is available for future use
    _ = trace_info

    api_key = task_props.get("api_key", validated_config.api_key)
    prompt = task_props.get("prompt", validated_config.prompt)
    endpoint_url = task_props.get("endpoint_url", validated_config.endpoint_url)

    # Create a mask for rows where the document type is IMAGE
    df_mask = df['metadata'].apply(lambda meta: meta.get('content_metadata', {}).get('type') == "image")

    if not df_mask.any():
        return df

    df.loc[df_mask, 'metadata'] = df.loc[df_mask, 'metadata'].apply(
        lambda meta: {
            **meta,
            'image_metadata': {
                **meta.get('image_metadata', {}),
                'caption': _generate_captions(meta['content'], prompt, api_key, endpoint_url)
            }
        }
    )

    logger.debug("Image content captioning complete")
    return df


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

    validated_config = ImageCaptionExtractionSchema(**caption_config)
    _wrapped_caption_extract = partial(caption_extract_stage, validated_config=validated_config)

    logger.debug(
        f"Generating caption extraction stage with {pe_count} processing elements. task: {task}, document_type: *"
    )
    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_caption_extract
    )
