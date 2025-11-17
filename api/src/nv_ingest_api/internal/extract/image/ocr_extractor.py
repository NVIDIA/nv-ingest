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

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.primitives.nim import NimClient
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import PaddleOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import NemoRetrieverOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import get_ocr_model_name
from nv_ingest_api.internal.schemas.extract.extract_ocr_schema import OCRExtractorSchema
from nv_ingest_api.util.image_processing.transforms import base64_to_numpy
from nv_ingest_api.util.nim import create_inference_client

logger = logging.getLogger(__name__)

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32


def _filter_text_images(
    base64_images: List[str],
    min_width: int = PADDLE_MIN_WIDTH,
    min_height: int = PADDLE_MIN_HEIGHT,
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
    """
    valid_images: List[str] = []
    valid_indices: List[int] = []

    for i, img in enumerate(base64_images):
        array = base64_to_numpy(img)
        height, width = array.shape[0], array.shape[1]
        if width >= min_width and height >= min_height:
            valid_images.append(img)
            valid_indices.append(i)
    return valid_images, valid_indices


def _update_text_metadata(
    base64_images: List[str],
    ocr_client: NimClient,
    ocr_model_name: str,
    worker_pool_size: int = 8,  # Not currently used
    trace_info: Optional[Dict] = None,
) -> List[Tuple[str, Optional[Any], Optional[Any]]]:
    """
    Filters base64-encoded images and uses OCR to extract text data.

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
    logger.debug(f"Running text extraction using protocol {ocr_client.protocol}")

    if ocr_model_name == "paddle":
        valid_images, valid_indices = _filter_text_images(base64_images)
    else:
        valid_images, valid_indices = _filter_text_images(base64_images, min_width=1, min_height=1)
    data_ocr = {"base64_images": valid_images}

    # worker_pool_size is not used in current implementation.
    _ = worker_pool_size

    infer_kwargs = dict(
        stage_name="ocr_extraction",
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

    results = [(None, None, None)] * len(base64_images)
    for idx, ocr_res in enumerate(ocr_results):
        original_index = valid_indices[idx]
        results[original_index] = ocr_res

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


def _meets_page_elements_text_criteria(row: pd.Series) -> bool:
    """
    Determines if a DataFrame row meets the criteria for text extraction.

    A row qualifies if:
      - It contains a 'metadata' dictionary.
      - The 'content_metadata' in metadata has type "text" and one of subtype:
        "title", "paragraph", "header_footer".
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
    page_element_subtypes = {"paragraph", "title", "header_footer"}

    metadata = row.get("metadata", {})
    if not metadata:
        return False

    content_md = metadata.get("content_metadata", {})

    if (
        content_md.get("type") == ContentTypeEnum.TEXT
        and content_md.get("subtype") in page_element_subtypes
        and metadata.get("content") not in {None, ""}
    ):
        return True

    return False


def _meets_page_image_criteria(row: pd.Series) -> bool:
    """
    Determines if a DataFrame row meets the criteria for text extraction.

    A row qualifies if:
      - It contains a 'metadata' dictionary.
      - The 'content_metadata' in metadata has type "image" and subtype "page_image".
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
    page_image_subtypes = {ContentTypeEnum.PAGE_IMAGE}

    metadata = row.get("metadata", {})
    if not metadata:
        return False

    content_md = metadata.get("content_metadata", {})

    if (
        content_md.get("type") == ContentTypeEnum.IMAGE
        and content_md.get("subtype") in page_image_subtypes
        and metadata.get("content") not in {None, ""}
    ):
        return True

    return False


def _process_page_images(df_to_process: pd.DataFrame, ocr_results: List[Tuple]):
    valid_indices = df_to_process.index.tolist()

    for result_idx, df_idx in enumerate(valid_indices):
        # Unpack result: (bounding_boxes, text_predictions, confidence_scores)
        bboxes, texts, _ = ocr_results[result_idx]
        if not bboxes or not texts:
            df_to_process.loc[df_idx, "metadata"]["image_metadata"]["text"] = ""
            continue

        df_to_process.loc[df_idx, "metadata"]["image_metadata"]["text"] = " ".join([t for t in texts])

    return df_to_process


def _process_page_elements(df_to_process: pd.DataFrame, ocr_results: List[Tuple]):
    valid_indices = df_to_process.index.tolist()
    if not valid_indices:
        return df_to_process

    for result_idx, df_idx in enumerate(valid_indices):
        # Unpack result: (bounding_boxes, text_predictions, confidence_scores)
        bboxes, texts, _ = ocr_results[result_idx]
        if not bboxes or not texts:
            df_to_process.loc[df_idx, "_x0"] = None
            df_to_process.loc[df_idx, "_y0"] = None
            df_to_process.loc[df_idx, "metadata"]["content"] = ""
            continue

        combined_data = list(zip(bboxes, texts))
        try:
            # Sort by reading order (y_min, then x_min)
            combined_data.sort(key=lambda item: (min(p[1] for p in item[0]), min(p[0] for p in item[0])))
        except (ValueError, IndexError):
            logger.warning("Could not sort OCR results due to malformed bounding box.")
        df_to_process.loc[df_idx, "_x0"] = min(point[0] for item in combined_data for point in item[0])
        df_to_process.loc[df_idx, "_y0"] = min(point[1] for item in combined_data for point in item[0])
        df_to_process.loc[df_idx, "metadata"]["content"] = " ".join([item[1] for item in combined_data])

    df_to_process = df_to_process.drop(["_x0", "_y0"], axis=1)

    df_to_process.loc[:, "_page_number"] = df_to_process["metadata"].apply(
        lambda meta: meta["content_metadata"]["page_number"]
    )

    # Group by page number to aggregate all text blocks on each page
    grouped = df_to_process.groupby("_page_number")

    new_text = {}
    for page_num, group_df in grouped:
        if group_df.empty:
            continue
        # Sort text blocks by their original position for correct reading order
        group_df.loc[:, "_x0"] = group_df["metadata"].apply(lambda meta: meta["text_metadata"]["text_location"][0])
        group_df.loc[:, "_y0"] = group_df["metadata"].apply(lambda meta: meta["text_metadata"]["text_location"][1])

        loc_mask = group_df[["_y0", "_x0"]].notna().all(axis=1)
        sorted_group = group_df.loc[loc_mask].sort_values(by=["_y0", "_x0"], ascending=[True, True])
        page_text = " ".join(sorted_group["metadata"].apply(lambda meta: meta["content"]).tolist())

        if page_text.strip():
            new_text[page_num] = page_text

    df_text = df_to_process[df_to_process["document_type"] == "text"].drop_duplicates(
        subset=["_page_number"], keep="first"
    )

    for page_num, page_text in new_text.items():
        page_num_mask = df_text["_page_number"] == page_num
        df_text.loc[page_num_mask, "metadata"] = df_text.loc[page_num_mask, "metadata"].apply(
            lambda meta: {**meta, "content": page_text}
        )

    df_non_text = df_to_process[df_to_process["document_type"] != "text"]
    df_to_process = pd.concat([df_text, df_non_text])

    for col in {"_y0", "_x0", "_page_number"}:
        if col in df_to_process:
            df_to_process = df_to_process.drop(col, axis=1)

    return df_to_process


def extract_text_data_from_image_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    extraction_config: OCRExtractorSchema,
    execution_trace_log: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts text data from a DataFrame in bulk, following the chart extraction pattern.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing the content from which text data is to be extracted.
    task_config : Dict[str, Any]
        Dictionary containing task properties and configurations.
    extraction_config : Any
        The validated configuration object for text extraction.
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
        # Identify rows that meet the text criteria.
        page_images_mask = df_extraction_ledger.apply(_meets_page_image_criteria, axis=1)
        page_elements_mask = df_extraction_ledger.apply(_meets_page_elements_text_criteria, axis=1)

        df_to_process = df_extraction_ledger[page_images_mask | page_elements_mask].copy()
        df_unprocessed = df_extraction_ledger[~page_images_mask & ~page_elements_mask].copy()

        valid_indices = df_to_process.index.tolist()
        # If no rows meet the criteria, return early.
        if not valid_indices:
            return df_extraction_ledger, {"trace_info": execution_trace_log}

        # Extract base64 images from valid rows.
        base64_images = [row["metadata"]["content"] for _, row in df_to_process.iterrows()]

        # Call bulk update to extract text data.
        ocr_client = _create_ocr_client(
            endpoint_config.ocr_endpoints,
            endpoint_config.ocr_infer_protocol,
            ocr_model_name,
            endpoint_config.auth_token,
        )

        bulk_results = _update_text_metadata(
            base64_images=base64_images,
            ocr_client=ocr_client,
            ocr_model_name=ocr_model_name,
            worker_pool_size=endpoint_config.workers_per_progress_engine,
            trace_info=execution_trace_log,
        )

        df_page_images = df_to_process[df_to_process.apply(_meets_page_image_criteria, axis=1)]
        df_page_images = _process_page_images(df_page_images, bulk_results)

        df_page_elements = df_to_process[df_to_process.apply(_meets_page_elements_text_criteria, axis=1)]
        df_page_elements = _process_page_elements(df_page_elements, bulk_results)

        df_final = pd.concat([df_unprocessed, df_page_images, df_page_elements], ignore_index=True)

        return df_final, {"trace_info": execution_trace_log}

    except Exception:
        err_msg = "Error occurred while extracting text data."
        logger.exception(err_msg)
        raise
