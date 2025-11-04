# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import functools
import io
import logging
from typing import Any, Union, Tuple
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from nv_ingest_api.internal.extract.image.image_helpers.common import unstructured_image_extractor
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageConfigSchema
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler

logger = logging.getLogger(__name__)


@unified_exception_handler
def _decode_and_extract_from_image(
    base64_row: pd.Series,
    task_config: Dict[str, Any],
    validated_extraction_config: ImageConfigSchema,
    execution_trace_log: Optional[List[Any]] = None,
) -> Any:
    """
    Decode base64-encoded image content from a DataFrame row and extract data using a specified extraction method.

    This function extracts the "content" (base64 string) from the row, prepares additional task parameters by
    inserting the remaining row data under "row_data", and decodes the base64 content into a BytesIO stream.
    It then determines which extraction method to use (defaulting to "image" if the specified method is not found)
    and calls the corresponding function from the image_helpers module.

    Parameters
    ----------
    base64_row : pd.Series
        A pandas Series representing a row containing base64-encoded content under the key "content"
        and optionally a "source_id" and "document_type".
    task_config : Dict[str, Any]
        A dictionary containing task properties. It should include:
            - "method" (str): The extraction method to use (e.g., "image").
            - "params" (dict): Additional parameters to pass to the extraction function.
    validated_extraction_config : Any
        A configuration object that contains an attribute `image_extraction_config` to be used when
        extracting image content.
    default : str, optional
        The default extraction method to use if the specified method is not available (default is "image").
    execution_trace_log : Optional[List[Any]], optional
        An optional list of trace information to pass to the extraction function (default is None).

    Returns
    -------
    Any
        The extracted data from the decoded image content. The exact return type depends on the extraction method used.

    Raises
    ------
    KeyError
        If the "content" key is missing from `base64_row`.
    Exception
        For any other unhandled exceptions during extraction.
    """

    # Retrieve document type and initialize source_id.
    document_type: Any = base64_row["document_type"]
    source_id: Optional[Any] = None

    try:
        base64_content: str = base64_row["content"]
    except KeyError as e:
        err_msg = f"decode_and_extract: Missing 'content' key in row: {base64_row}"
        logger.error(err_msg, exc_info=True)
        raise KeyError(err_msg) from e

    try:
        # Prepare additional row data (exclude "content") and inject into task parameters.
        row_data = base64_row.drop(labels=["content"], errors="ignore")
        task_config.setdefault("params", {})["row_data"] = row_data

        # Retrieve source_id if available.
        source_id = base64_row.get("source_id", None)

        # Decode the base64 image content.
        image_bytes: bytes = base64.b64decode(base64_content)
        image_stream: io.BytesIO = io.BytesIO(image_bytes)

        # Determine the extraction method and parameters.
        # extract_method: str = task_config.get("method", "image")
        extract_params: Dict[str, Any] = task_config.get("params", {})
        extract_params["document_type"] = document_type

        try:
            extract_text: bool = extract_params.pop("extract_text", False)
            extract_images: bool = extract_params.pop("extract_images", False)
            extract_tables: bool = extract_params.pop("extract_tables", False)
            extract_charts: bool = extract_params.pop("extract_charts", False)
            extract_infographics: bool = extract_params.pop("extract_infographics", False)
        except KeyError as e:
            raise ValueError(f"Missing required extraction flag: {e}")

        logger.debug(
            f"decode_and_extract: Extracting image content using image_extraction_config: "
            f"{validated_extraction_config}"
        )
        # Ensure we pass the correct nested config type (ImageConfigSchema) to helpers.
        # Some callers provide the full ImageExtractorSchema; extract its inner image_extraction_config.
        if validated_extraction_config is not None:
            inner_cfg = getattr(validated_extraction_config, "image_extraction_config", validated_extraction_config)
            if inner_cfg is not None:
                extract_params["image_extraction_config"] = inner_cfg

        if execution_trace_log is not None:
            extract_params["trace_info"] = execution_trace_log

        # func = getattr(image_helpers, extract_method, default)
        extracted_data: Any = unstructured_image_extractor(
            image_stream=image_stream,
            extract_text=extract_text,
            extract_images=extract_images,
            extract_infographics=extract_infographics,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extraction_config=extract_params,
            extraction_trace_log=execution_trace_log,
        )

        return extracted_data

    except Exception as e:
        err_msg = f"decode_and_extract: Unhandled exception for source '{source_id}'. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


@unified_exception_handler
def extract_primitives_from_image_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Union[Dict[str, Any], BaseModel],
    extraction_config: Any,
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process a DataFrame containing base64-encoded image files and extract primitives from each image.

    This function applies the `decode_and_extract_from_image` routine to every row of the input DataFrame.
    It then explodes any list results into separate rows, drops missing values, and compiles the extracted data
    into a new DataFrame with columns "document_type", "metadata", and "uuid". In addition, trace information is
    collected if provided.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        Input DataFrame containing image files in base64 encoding. Expected to include columns 'source_id'
        and 'content'.
    task_config : Union[Dict[str, Any], BaseModel]
        A dictionary or Pydantic model with instructions and parameters for the image processing task.
    extraction_config : Any
        A configuration object validated for processing images (e.g., containing `image_extraction_config`).
    execution_trace_log : Optional[Dict[str, Any]], default=None
        An optional dictionary for tracing and logging additional information during processing.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the extracted image primitives. Expected columns include "document_type", "metadata",
        and "uuid". Also returns a dictionary containing trace information under the key "trace_info".

    Raises
    ------
    Exception
        If an error occurs during the image processing stage, the exception is logged and re-raised.
    """
    logger.debug("process_image: Processing image content")
    if execution_trace_log is None:
        execution_trace_log = {}

    if isinstance(task_config, BaseModel):
        task_config = task_config.model_dump()

    try:
        # Create a partial function to decode and extract image data for each row.
        _decode_and_extract = functools.partial(
            _decode_and_extract_from_image,
            task_config=task_config,
            validated_extraction_config=extraction_config,
            execution_trace_log=execution_trace_log,
        )
        logger.debug("process_image: Processing with method: %s", task_config.get("method", None))
        sr_extraction = df_extraction_ledger.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        return extracted_df, {"trace_info": execution_trace_log}

    except Exception as e:
        err_msg = f"process_image: Unhandled exception in image extractor stage. Original error: {e}"
        logger.exception(err_msg)
        raise type(e)(err_msg) from e
