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
import nv_ingest.extraction_workflows.image as image_helpers
from morpheus.config import Config
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
    base64_row : dict
        A dictionary containing the base64-encoded content and other relevant data.
        The key "content" should contain the base64 string, and the key "source_id" is optional.
    task_props : dict
        A dictionary containing task properties. It should have the keys:
        - "method" (str): The extraction method to use (e.g., "image").
        - "params" (dict): Parameters to pass to the extraction function.
    validated_config : Any
        Configuration object that contains `image_config`. Used if the `image` method is selected.
    default : str, optional
        The default extraction method to use if the specified method in `task_props` is not available (default is "image").

    Returns
    -------
    Any
        The extracted data from the decoded content. The exact return type depends on the extraction method used.

    Raises
    ------
    KeyError
        If the "content" key is missing from `base64_row`.
    Exception
        For any other unhandled exceptions during extraction, an error is logged, and the exception is re-raised.
    """

    document_type = base64_row["document_type"]
    source_id = None
    try:
        base64_content = base64_row["content"]
    except KeyError:
        log_error_message = f"Unhandled error processing row, no content was found:\n{base64_row}"
        logger.error(log_error_message)
        raise

    try:
        # Row data to include in extraction
        bool_index = base64_row.index.isin(("content",))
        row_data = base64_row[~bool_index]
        task_props["params"]["row_data"] = row_data

        # Get source_id
        source_id = base64_row["source_id"] if "source_id" in base64_row.index else None
        # Decode the base64 content
        image_bytes = base64.b64decode(base64_content)

        # Load the PDF
        image_stream = io.BytesIO(image_bytes)

        # Type of extraction method to use
        extract_method = task_props.get("method", "image")
        extract_params = task_props.get("params", {})

        logger.debug(
            f">>> Extracting image content, image_extraction_config: {validated_config.image_extraction_config}")
        if (validated_config.image_extraction_config is not None):
            extract_params["image_extraction_config"] = validated_config.image_extraction_config

        if (trace_info is not None):
            extract_params["trace_info"] = trace_info

        if (not hasattr(image_helpers, extract_method)):
            extract_method = default

        func = getattr(image_helpers, extract_method, default)
        logger.debug("Running extraction method: %s", extract_method)
        extracted_data = func(image_stream, document_type, **extract_params)

        return extracted_data

    except Exception as e:
        traceback.print_exc()
        err_msg = f"Unhandled exception in decode_and_extract for '{source_id}':\n{e}"
        logger.error(err_msg)

        raise

    # Propagate error back and tag message as failed.
    # exception_tag = create_exception_tag(error_message=log_error_message, source_id=source_id)


def process_image(
        df: pd.DataFrame,
        task_props: Dict[str, Any],
        validated_config: Any,
        trace_info: Optional[Dict[str, Any]] = None
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
        Dictionary for tracing and logging additional information during processing (default is None).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        A tuple containing:
        - A pandas DataFrame with the processed image content, including columns 'document_type', 'metadata', and 'uuid'.
        - A dictionary with trace information collected during processing.

    Raises
    ------
    Exception
        If an error occurs during the image processing stage.
    """
    logger.debug("Processing image content")
    if trace_info is None:
        trace_info = {}

    try:
        # Apply the helper function to each row in the 'content' column
        _decode_and_extract = functools.partial(
            decode_and_extract, task_props=task_props, validated_config=validated_config, trace_info=trace_info
        )
        logger.debug(f"Processing method: {task_props.get('method', None)}")
        sr_extraction = df.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        return extracted_df, {"trace_info": trace_info}

    except Exception as e:
        err_msg = f"Unhandled exception in image extractor stage's process_image: {e}"
        logger.error(err_msg)
        raise


def generate_image_extractor_stage(
        c: Config,
        extractor_config: Dict[str, Any],
        task: str = "extract",
        task_desc: str = "image_content_extractor",
        pe_count: int = 24,
):
    """
    Helper function to generate a multiprocessing stage to perform image content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object
    extractor_config : dict
        Configuration parameters for pdf content extractor.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        Integer for how many process engines to use for pdf content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with applied worker function.
    """
    validated_config = ImageExtractorSchema(**extractor_config)
    _wrapped_process_fn = functools.partial(process_image, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn,
        document_type="regex:^(png|svg|jpeg|jpg|tiff)$"
    )
