# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
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


def _local_nemotron_ocr_text_predictions(
    base64_images: List[str],
    *,
    merge_level: str = "paragraph",
    trace_info: Optional[Dict] = None,
) -> List[Tuple[str, Optional[Any], Optional[Any]]]:
    """
    Local OCR fallback using the Nemotron OCR v1 pipeline via:
      `nemo_retriever.model.local.nemotron_ocr_v1.NemotronOCRV1`

    Returns list of tuples aligned with base64_images:
      (base64_image, bounding_boxes_or_none, text_predictions_or_none)

    This mirrors the shape consumed by the caller in this module.
    """
    # Keep the same "skip tiny images" behavior as the NIM path.
    valid_images, valid_indices, results = _filter_infographic_images(base64_images)

    model_dir = (
        os.getenv("RETRIEVER_NEMOTRON_OCR_MODEL_DIR", "").strip()
        or os.getenv("NEMOTRON_OCR_MODEL_DIR", "").strip()
        or os.getenv("NEMOTRON_OCR_V1_MODEL_DIR", "").strip()
    )

    # Import locally to avoid making `nv-ingest-api` hard-depend on nemo-retriever unless needed.
    try:
        from nemo_retriever.model.local.nemotron_ocr_v1 import NemotronOCRV1  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Local infographic OCR fallback requires the `nemo-retriever` package to be importable "
            "so we can use `nemo_retriever.model.local.nemotron_ocr_v1.NemotronOCRV1`."
        ) from e

    if trace_info is not None:
        trace_info.setdefault("ocr", {})
        trace_info["ocr"]["backend"] = "local_nemotron_ocr_v1"
        trace_info["ocr"]["model_dir"] = model_dir or None

    ocr = NemotronOCRV1(model_dir=model_dir) if model_dir else NemotronOCRV1()

    for idx, b64 in enumerate(valid_images):
        original_index = valid_indices[idx]
        try:
            # Pass base64 directly; NemotronOCR supports base64 bytes.
            preds = ocr.invoke(b64, merge_level=merge_level)

            # Best-effort extraction of text strings from Nemotron OCR outputs.
            texts: List[str] = []
            if isinstance(preds, dict):
                if isinstance(preds.get("text"), str):
                    texts.append(preds["text"])
                if isinstance(preds.get("texts"), list):
                    texts.extend([x for x in preds["texts"] if isinstance(x, str)])
                if isinstance(preds.get("words"), list):
                    for w in preds["words"]:
                        if isinstance(w, dict) and isinstance(w.get("text"), str):
                            texts.append(w["text"])
            elif isinstance(preds, list):
                for item in preds:
                    if isinstance(item, str):
                        if item.strip():
                            texts.append(item.strip())
                        continue
                    if isinstance(item, dict):
                        if isinstance(item.get("text"), str) and item["text"].strip() and item["text"].strip() != "nan":
                            texts.append(item["text"].strip())
                            continue
                        if isinstance(item.get("texts"), list):
                            texts.extend([x.strip() for x in item["texts"] if isinstance(x, str) and x.strip()])
                            continue
                        if isinstance(item.get("words"), list):
                            for w in item["words"]:
                                if isinstance(w, dict) and isinstance(w.get("text"), str) and w["text"].strip():
                                    texts.append(w["text"].strip())

            # Fallback: stringify unknown shapes.
            if not texts:
                try:
                    s = str(preds).strip()
                    if s and s.lower() not in {"none", "null"}:
                        texts = [s]
                except Exception:
                    texts = []

            # The rest of the pipeline expects `text_predictions` to be a list of strings.
            text_predictions = texts if texts else None
            results[original_index] = (base64_images[original_index], None, text_predictions)
        except Exception:
            logger.exception("Local Nemotron OCR failed for infographic image index=%s", original_index)
            results[original_index] = (base64_images[original_index], None, None)

    return results


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

    try:
        # Identify rows that meet the infographic criteria.
        mask = df_extraction_ledger.apply(_meets_infographic_criteria, axis=1)
        valid_indices = df_extraction_ledger[mask].index.tolist()

        # If no rows meet the criteria, return early.
        if not valid_indices:
            return df_extraction_ledger, {"trace_info": execution_trace_log}

        # Extract base64 images from valid rows.
        base64_images = [df_extraction_ledger.at[idx, "metadata"]["content"] for idx in valid_indices]

        # If endpoints are not configured, fall back to local Nemotron OCR.
        ocr_endpoints = (None, None)
        ocr_protocol = "local"
        auth_token = ""
        workers = 5
        if endpoint_config is not None:
            ocr_endpoints = getattr(endpoint_config, "ocr_endpoints", (None, None))
            ocr_protocol = getattr(endpoint_config, "ocr_infer_protocol", "") or "local"
            auth_token = getattr(endpoint_config, "auth_token", "") or ""
            workers = int(getattr(endpoint_config, "workers_per_progress_engine", 5) or 5)

        has_endpoint = bool((ocr_endpoints[0] or ocr_endpoints[1]))
        if not has_endpoint or str(ocr_protocol).lower() == "local":
            bulk_results = _local_nemotron_ocr_text_predictions(
                base64_images,
                merge_level="paragraph",
                trace_info=execution_trace_log,
            )
        else:
            # Get the grpc endpoint to determine the model if needed
            ocr_grpc_endpoint = ocr_endpoints[0]
            ocr_model_name = get_ocr_model_name(ocr_grpc_endpoint)

            # Call bulk update to extract infographic data via NIM endpoints.
            ocr_client = _create_ocr_client(
                ocr_endpoints,
                ocr_protocol,
                ocr_model_name,
                auth_token,
            )

            bulk_results = _update_infographic_metadata(
                base64_images=base64_images,
                ocr_client=ocr_client,
                ocr_model_name=ocr_model_name,
                worker_pool_size=workers,
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
