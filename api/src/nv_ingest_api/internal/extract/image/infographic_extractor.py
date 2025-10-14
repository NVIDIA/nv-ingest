# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd

from nv_ingest_api.internal.primitives.nim import NimClient
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import PaddleOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import NemoRetrieverOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import get_ocr_model_name
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema
from nv_ingest_api.util.image_processing.transforms import base64_to_numpy
from nv_ingest_api.util.nim import create_inference_client
from nv_ingest_api.util.image_processing.table_and_chart import reorder_boxes

logger = logging.getLogger(__name__)

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32


def _filter_infographic_images(
    base64_images: List[str],
) -> Tuple[List[str], List[int], List[Tuple[str, Optional[Any], Optional[Any]]]]:
    """
    Filters base64-encoded images based on minimum size requirements.

    Parameters
    ----------
    base64_images : List[str]
        List of base64-encoded image strings.

    Returns
    -------
    Tuple[List[str], List[int], List[Tuple[str, Optional[Any], Optional[Any]]]]
        - valid_images: List of images that meet the size requirements.
        - valid_indices: Original indices of valid images.
        - results: Initialized results list, with invalid images marked as (img, None, None).
    """
    results: List[Tuple[str, Optional[Any], Optional[Any]]] = [("", None, None)] * len(base64_images)
    valid_images: List[str] = []
    valid_indices: List[int] = []

    for i, img in enumerate(base64_images):
        array = base64_to_numpy(img)
        height, width = array.shape[0], array.shape[1]
        if width >= PADDLE_MIN_WIDTH and height >= PADDLE_MIN_HEIGHT:
            valid_images.append(img)
            valid_indices.append(i)
        else:
            # Mark image as skipped if it does not meet size requirements.
            results[i] = (img, None, None)
    return valid_images, valid_indices, results


def _update_infographic_metadata(
    base64_images: List[str],
    ocr_client: NimClient,
    ocr_model_name: str,
    worker_pool_size: int = 8,  # Not currently used
    trace_info: Optional[Dict] = None,
) -> List[Tuple[str, Optional[Any], Optional[Any]]]:
    """
    Filters base64-encoded images and uses OCR to extract infographic data.

    For each image that meets the minimum size, calls ocr_client.infer to obtain
    (text_predictions, bounding_boxes). Invalid images are marked as skipped.

    Parameters
    ----------
    base64_images : List[str]
        List of base64-encoded images.
    ocr_client : NimClient
        Client instance for OCR inference.
    worker_pool_size : int, optional
        Worker pool size (currently not used), by default 8.
    trace_info : Optional[Dict], optional
        Optional trace information for debugging.

    Returns
    -------
    List[Tuple[str, Optional[Any], Optional[Any]]]
        List of tuples in the same order as base64_images, where each tuple contains:
        (base64_image, text_predictions, bounding_boxes).
    """
    logger.debug(f"Running infographic extraction using protocol {ocr_client.protocol}")

    valid_images, valid_indices, results = _filter_infographic_images(base64_images)
    data_ocr = {"base64_images": valid_images}

    # worker_pool_size is not used in current implementation.
    _ = worker_pool_size

    infer_kwargs = dict(
        stage_name="infographic_extraction",
        trace_info=trace_info,
    )
    if ocr_model_name == "paddle":
        infer_kwargs.update(
            model_name="paddle",
            max_batch_size=1 if ocr_client.protocol == "grpc" else 2,
        )
    elif ocr_model_name in {"scene_text_ensemble", "scene_text_wrapper", "scene_text_python"}:
        infer_kwargs.update(
            model_name=ocr_model_name,
            input_names=["INPUT_IMAGE_URLS", "MERGE_LEVELS"],
            output_names=["OUTPUT"],
            dtypes=["BYTES", "BYTES"],
            merge_level="paragraph",
        )
    else:
        raise ValueError(f"Unknown OCR model name: {ocr_model_name}")

    try:
        ocr_results = ocr_client.infer(data_ocr, **infer_kwargs)
    except Exception as e:
        logger.error(f"Error calling ocr_client.infer: {e}", exc_info=True)
        raise

    if len(ocr_results) != len(valid_images):
        raise ValueError(f"Expected {len(valid_images)} ocr results, got {len(ocr_results)}")

    for idx, ocr_res in enumerate(ocr_results):
        original_index = valid_indices[idx]

        if ocr_model_name == "paddle":
            logger.debug(f"OCR results for image {base64_images[original_index]}: {ocr_res}")
        else:
            # Each ocr_res is expected to be a tuple (text_predictions, bounding_boxes, conf_scores).
            ocr_res = reorder_boxes(*ocr_res)

        results[original_index] = (
            base64_images[original_index],
            ocr_res[0],
            ocr_res[1],
        )

    return results


def _create_ocr_client(
    ocr_endpoints: Tuple[str, str],
    ocr_protocol: str,
    ocr_model_name: str,
    auth_token: str,
) -> NimClient:
    ocr_model_interface = (
        NemoRetrieverOCRModelInterface()
        if ocr_model_name in {"scene_text_ensemble", "scene_text_wrapper", "scene_text_python"}
        else PaddleOCRModelInterface()
    )

    ocr_client = create_inference_client(
        endpoints=ocr_endpoints,
        model_interface=ocr_model_interface,
        auth_token=auth_token,
        infer_protocol=ocr_protocol,
        enable_dynamic_batching=(
            True if ocr_model_name in {"scene_text_ensemble", "scene_text_wrapper", "scene_text_python"} else False
        ),
        dynamic_batch_memory_budget_mb=32,
    )

    return ocr_client


def _meets_infographic_criteria(row: pd.Series) -> bool:
    """
    Determines if a DataFrame row meets the criteria for infographic extraction.

    A row qualifies if:
      - It contains a 'metadata' dictionary.
      - The 'content_metadata' in metadata has type "structured" and subtype "infographic".
      - The 'table_metadata' is not None.
      - The 'content' is not None or an empty string.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame.

    Returns
    -------
    bool
        True if the row meets all criteria; False otherwise.
    """
    metadata = row.get("metadata", {})
    if not metadata:
        return False

    content_md = metadata.get("content_metadata", {})
    if (
        content_md.get("type") == "structured"
        and content_md.get("subtype") == "infographic"
        and metadata.get("table_metadata") is not None
        and metadata.get("content") not in [None, ""]
    ):
        return True

    return False


def extract_infographic_data_from_image_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    extraction_config: InfographicExtractorSchema,
    execution_trace_log: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts infographic data from a DataFrame in bulk, following the chart extraction pattern.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing the content from which infographic data is to be extracted.
    task_config : Dict[str, Any]
        Dictionary containing task properties and configurations.
    extraction_config : Any
        The validated configuration object for infographic extraction.
    execution_trace_log : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame and the trace information.
    """
    _ = task_config  # Unused

    if execution_trace_log is None:
        execution_trace_log = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df_extraction_ledger.empty:
        return df_extraction_ledger, execution_trace_log

    endpoint_config = extraction_config.endpoint_config

    # Get the grpc endpoint to determine the model if needed
    ocr_grpc_endpoint = endpoint_config.ocr_endpoints[0]
    ocr_model_name = get_ocr_model_name(ocr_grpc_endpoint)

    try:
        # Identify rows that meet the infographic criteria.
        mask = df_extraction_ledger.apply(_meets_infographic_criteria, axis=1)
        valid_indices = df_extraction_ledger[mask].index.tolist()

        # If no rows meet the criteria, return early.
        if not valid_indices:
            return df_extraction_ledger, {"trace_info": execution_trace_log}

        # Extract base64 images from valid rows.
        base64_images = [df_extraction_ledger.at[idx, "metadata"]["content"] for idx in valid_indices]

        # Call bulk update to extract infographic data.
        ocr_client = _create_ocr_client(
            endpoint_config.ocr_endpoints,
            endpoint_config.ocr_infer_protocol,
            ocr_model_name,
            endpoint_config.auth_token,
        )

        bulk_results = _update_infographic_metadata(
            base64_images=base64_images,
            ocr_client=ocr_client,
            ocr_model_name=ocr_model_name,
            worker_pool_size=endpoint_config.workers_per_progress_engine,
            trace_info=execution_trace_log,
        )

        # Write the extracted results back into the DataFrame.
        for result_idx, df_idx in enumerate(valid_indices):
            # Unpack result: (base64_image, ocr_bounding_boxes, ocr_text_predictions)
            _, _, text_predictions = bulk_results[result_idx]
            table_content = " ".join(text_predictions) if text_predictions else None
            df_extraction_ledger.at[df_idx, "metadata"]["table_metadata"]["table_content"] = table_content

        return df_extraction_ledger, {"trace_info": execution_trace_log}

    except Exception:
        err_msg = "Error occurred while extracting infographic data."
        logger.exception(err_msg)
        raise
