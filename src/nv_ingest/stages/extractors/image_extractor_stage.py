# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import functools
import io
import logging
import traceback
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from pydantic import BaseModel
from morpheus.config import Config

import nv_ingest.extraction_workflows.image as image_helpers
from nv_ingest.schemas.image_extractor_schema import ImageExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage

logger = logging.getLogger(f"morpheus.{__name__}")


def decode_and_extract(
    base64_row: pd.Series,
    task_props: Dict[str, Any],
    validated_config: Any,
    default: str = "image",
    trace_info: Optional[List] = None,
) -> Any:
    """
    Decodes base64 content from a row and extracts data from it using the specified extraction method.

    Parameters
    ----------
    base64_row : pd.Series
        A series containing the base64-encoded content and other relevant data.
        The key "content" should contain the base64 string, and the key "source_id" is optional.
    task_props : dict
        A dictionary containing task properties. It should have the keys:
        - "method" (str): The extraction method to use (e.g., "image").
        - "params" (dict): Parameters to pass to the extraction function.
    validated_config : Any
        Configuration object that contains `image_extraction_config`. Used if the `image` method is selected.
    default : str, optional
        The default extraction method to use if the specified method in `task_props` is not available
        (default is "image").
    trace_info : Optional[List], optional
        An optional list for trace information to pass to the extraction function.

    Returns
    -------
    Any
        The extracted data from the decoded content. The exact return type depends on the extraction method used.

    Raises
    ------
    KeyError
        If the "content" key is missing from `base64_row`.
    Exception
        For any other unhandled exceptions during extraction.
    """
    # Retrieve document type and initialize source_id.
    document_type = base64_row["document_type"]
    source_id = None

    try:
        base64_content = base64_row["content"]
    except KeyError as e:
        err_msg = f"decode_and_extract: Missing 'content' key in row: {base64_row}"
        logger.error(err_msg, exc_info=True)
        raise KeyError(err_msg) from e

    try:
        # Prepare row data (excluding the "content" column) for extraction.
        bool_index = base64_row.index.isin(("content",))
        row_data = base64_row[~bool_index]
        task_props["params"]["row_data"] = row_data

        # Retrieve source_id if available.
        source_id = base64_row["source_id"] if "source_id" in base64_row.index else None

        # Decode the base64 image content.
        image_bytes = base64.b64decode(base64_content)
        image_stream = io.BytesIO(image_bytes)

        # Determine the extraction method and parameters.
        extract_method = task_props.get("method", "image")
        extract_params = task_props.get("params", {})

        logger.debug(
            f"decode_and_extract: Extracting image content using image_extraction_config: "
            f"{validated_config.image_extraction_config}"
        )
        if validated_config.image_extraction_config is not None:
            extract_params["image_extraction_config"] = validated_config.image_extraction_config

        if trace_info is not None:
            extract_params["trace_info"] = trace_info

        if not hasattr(image_helpers, extract_method):
            extract_method = default

        func = getattr(image_helpers, extract_method, default)
        logger.debug("decode_and_extract: Running extraction method: %s", extract_method)
        extracted_data = func(image_stream, document_type, **extract_params)
        return extracted_data

    except Exception as e:
        err_msg = f"decode_and_extract: Unhandled exception for source '{source_id}'. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def process_image(
    df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Processes a pandas DataFrame containing image files in base64 encoding.
    Each image's content is replaced with its extracted components.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with columns 'source_id' and 'content' (base64-encoded image data).
    task_props : dict
        Dictionary containing instructions and parameters for the image processing task.
    validated_config : Any
        Configuration object validated for processing images.
    trace_info : dict, optional
        Dictionary for tracing and logging additional information during processing.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        A tuple containing:
          - A pandas DataFrame with the processed image content, including columns 'document_type', 'metadata', and
          'uuid'.
          - A dictionary with trace information collected during processing.

    Raises
    ------
    Exception
        If an error occurs during the image processing stage.
    """
    logger.debug("process_image: Processing image content")
    if trace_info is None:
        trace_info = {}

    if isinstance(task_props, BaseModel):
        task_props = task_props.model_dump()

    try:
        # Apply the helper function to each row in the 'content' column.
        _decode_and_extract = functools.partial(
            decode_and_extract,
            task_props=task_props,
            validated_config=validated_config,
            trace_info=trace_info,
        )
        logger.debug(f"process_image: Processing with method: {task_props.get('method', None)}")
        sr_extraction = df.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        return extracted_df, {"trace_info": trace_info}

    except Exception as e:
        err_msg = f"process_image: Unhandled exception in image extractor stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        traceback.print_exc()

        raise type(e)(err_msg) from e


def generate_image_extractor_stage(
    c: Config,
    extractor_config: Dict[str, Any],
    task: str = "extract",
    task_desc: str = "image_content_extractor",
    pe_count: int = 1,
):
    """
    Helper function to generate a multiprocessing stage to perform image content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    extractor_config : dict
        Configuration parameters for image content extractor.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        The number of process engines to use for image content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with the applied worker function.
    """
    try:
        validated_config = ImageExtractorSchema(**extractor_config)
        _wrapped_process_fn = functools.partial(process_image, validated_config=validated_config)
        return MultiProcessingBaseStage(
            c=c,
            pe_count=pe_count,
            task=task,
            task_desc=task_desc,
            process_fn=_wrapped_process_fn,
            document_type="regex:^(png|jpeg|jpg|tiff|bmp)$",
        )
    except Exception as e:
        err_msg = f"generate_image_extractor_stage: Error generating image extractor stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
