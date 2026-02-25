# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskChartExtraction
from nv_ingest_api.util.image_processing.table_and_chart import join_yolox_graphic_elements_and_ocr_output
from nv_ingest_api.util.image_processing.table_and_chart import process_yolox_graphic_elements
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import PaddleOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import NemoRetrieverOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import get_ocr_model_name  # noqa: F401
from nv_ingest_api.internal.primitives.nim import NimClient
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import YoloxGraphicElementsModelInterface
from nv_ingest_api.util.image_processing.transforms import base64_to_numpy
from nv_ingest_api.util.nim import create_inference_client

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32

logger = logging.getLogger(f"ray.{__name__}")


def _local_nemotron_ocr_boxes_texts(
    base64_images: List[str],
    *,
    merge_level: str = "paragraph",
    trace_info: Optional[Dict] = None,
) -> List[List[Any]]:
    """
    Local OCR fallback using the Nemotron OCR v1 pipeline via:
      `retriever.model.local.nemotron_ocr_v1.NemotronOCRV1`

    Returns list aligned with base64_images:
      [bounding_boxes, text_predictions, conf_scores]
    """
    model_dir = (
        os.getenv("NEMOTRON_OCR_MODEL_DIR", "").strip()
        or os.getenv("NEMOTRON_OCR_V1_MODEL_DIR", "").strip()
        or os.getenv("SLIMGEST_NEMOTRON_OCR_MODEL_DIR", "").strip()
    )
    if not model_dir:
        raise ValueError(
            "Local chart OCR requested but no model directory was configured. "
            "Set $NEMOTRON_OCR_MODEL_DIR (or $NEMOTRON_OCR_V1_MODEL_DIR) to the Nemotron OCR model directory."
        )

    # Import locally to avoid making `nv-ingest-api` hard-depend on retriever unless needed.
    try:
        from retriever.model.local.nemotron_ocr_v1 import NemotronOCRV1  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Local chart OCR fallback requires the `retriever` package to be importable "
            "so we can use `retriever.model.local.nemotron_ocr_v1.NemotronOCRV1`."
        ) from e

    if trace_info is not None:
        trace_info.setdefault("ocr", {})
        trace_info["ocr"]["backend"] = "local_nemotron_ocr_v1"
        trace_info["ocr"]["model_dir"] = model_dir

    ocr = NemotronOCRV1(model_dir=model_dir)

    results: List[List[Any]] = []
    for b64 in base64_images:
        try:
            arr = base64_to_numpy(b64)
            h, w = int(arr.shape[0]), int(arr.shape[1])
            preds = ocr.invoke(b64, merge_level=merge_level)

            boxes: List[List[List[float]]] = []
            texts: List[str] = []
            confs: List[float] = []

            # Common per-line dict form: left/right/upper/lower in [0,1] plus text.
            if isinstance(preds, list):
                for item in preds:
                    if not isinstance(item, dict):
                        continue
                    txt = item.get("text")
                    if not isinstance(txt, str) or not txt.strip() or txt.strip() == "nan":
                        continue

                    if all(k in item for k in ("left", "right", "upper", "lower")):
                        try:
                            x1 = float(item["left"]) * float(w)
                            x2 = float(item["right"]) * float(w)
                            # nemotron_ocr uses (lower, upper) y coords.
                            y1 = float(item["lower"]) * float(h)
                            y2 = float(item["upper"]) * float(h)
                            quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        except Exception:
                            quad = [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]
                    else:
                        quad = [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]

                    texts.append(txt.strip())
                    boxes.append(quad)
                    confs.append(
                        float(item.get("confidence")) if isinstance(item.get("confidence"), (int, float)) else 1.0
                    )

            # Fallback: stringify unknown output.
            if not texts:
                s = ""
                try:
                    s = str(preds).strip()
                except Exception:
                    s = ""
                if s and s.lower() not in {"none", "null"}:
                    texts = [s]
                    boxes = [[[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]]
                    confs = [1.0]

            results.append([boxes, texts, confs])
        except Exception:
            logger.exception("Local Nemotron OCR failed for chart image.")
            results.append([[], [], []])

    return results


class _LocalNemotronOCRClient:
    @property
    def protocol(self) -> str:
        return "local"

    def infer(self, data: Dict[str, Any], **kwargs: Any) -> Any:
        base64_images = data.get("base64_images") or []
        if not isinstance(base64_images, list):
            raise ValueError("Expected data['base64_images'] to be a list.")
        merge_level = kwargs.get("merge_level", "paragraph")
        trace_info = kwargs.get("trace_info")
        return _local_nemotron_ocr_boxes_texts(base64_images, merge_level=merge_level, trace_info=trace_info)


def _filter_valid_chart_images(
    base64_images: List[str],
) -> Tuple[List[str], List[np.ndarray], List[int], List[Tuple[str, Optional[Dict]]]]:
    """
    Filter base64-encoded images based on minimum dimensions for chart extraction.

    Returns:
      - valid_images: Base64 strings meeting size requirements.
      - valid_arrays: Corresponding numpy arrays.
      - valid_indices: Original indices of valid images.
      - results: Initial results list where invalid images are set to (img, None).
    """
    results: List[Tuple[str, Optional[Dict]]] = [("", None)] * len(base64_images)
    valid_images: List[str] = []
    valid_arrays: List[np.ndarray] = []
    valid_indices: List[int] = []

    for i, img in enumerate(base64_images):
        array = base64_to_numpy(img)
        height, width = array.shape[0], array.shape[1]
        if width >= PADDLE_MIN_WIDTH and height >= PADDLE_MIN_HEIGHT:
            valid_images.append(img)
            valid_arrays.append(array)
            valid_indices.append(i)
        else:
            # Image is too small; mark as skipped.
            results[i] = (img, None)
    return valid_images, valid_arrays, valid_indices, results


def _run_chart_inference(
    yolox_client: Any,
    yolox_model_name: str,
    ocr_client: Any,
    ocr_model_name: str,
    valid_arrays: List[np.ndarray],
    valid_images: List[str],
    trace_info: Dict,
) -> Tuple[List[Any], List[Any]]:
    """
    Run concurrent inference for chart extraction using YOLOX and Paddle.

    Returns a tuple of (yolox_results, ocr_results).
    """
    data_yolox = {"images": valid_arrays}
    data_ocr = {"base64_images": valid_images}

    future_yolox_kwargs = dict(
        data=data_yolox,
        model_name=yolox_model_name,
        stage_name="chart_extraction",
        input_names=["INPUT_IMAGES", "THRESHOLDS"],
        dtypes=["BYTES", "FP32"],
        output_names=["OUTPUT"],
        trace_info=trace_info,
        max_batch_size=8,
    )
    future_ocr_kwargs = dict(
        data=data_ocr,
        stage_name="chart_extraction",
        trace_info=trace_info,
    )
    if str(getattr(ocr_client, "protocol", "")).lower() == "local":
        # Local Nemotron OCR path (model_name/input_names/etc. are ignored by the local adapter).
        future_ocr_kwargs.update(merge_level="paragraph")
    else:
        if ocr_model_name == "paddle":
            future_ocr_kwargs.update(
                model_name="paddle",
                max_batch_size=1 if ocr_client.protocol == "grpc" else 2,
            )
        elif ocr_model_name in {"scene_text_ensemble", "scene_text_wrapper", "scene_text_python"}:
            future_ocr_kwargs.update(
                model_name=ocr_model_name,
                input_names=["INPUT_IMAGE_URLS", "MERGE_LEVELS"],
                output_names=["OUTPUT"],
                dtypes=["BYTES", "BYTES"],
                merge_level="paragraph",
            )
        else:
            raise ValueError(f"Unknown OCR model name: {ocr_model_name}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_yolox = executor.submit(yolox_client.infer, **future_yolox_kwargs)
        future_ocr = executor.submit(ocr_client.infer, **future_ocr_kwargs)

        try:
            yolox_results = future_yolox.result()
        except Exception as e:
            logger.error(f"Error calling yolox_client.infer: {e}", exc_info=True)
            raise

        try:
            ocr_results = future_ocr.result()
        except Exception as e:
            logger.error(f"Error calling ocr_client.infer: {e}", exc_info=True)
            raise

    return yolox_results, ocr_results


def _validate_chart_inference_results(
    yolox_results: Any,
    ocr_results: Any,
    valid_arrays: List[Any],
    valid_images: List[str],
) -> Tuple[List[Any], List[Any]]:
    """
    Ensure inference results are lists and have expected lengths.

    Raises:
      ValueError if results do not match expected types or lengths.
    """
    if not (isinstance(yolox_results, list) and isinstance(ocr_results, list)):
        raise ValueError("Expected list results from both yolox_client and ocr_client infer calls.")

    if len(yolox_results) != len(valid_arrays):
        raise ValueError(f"Expected {len(valid_arrays)} yolox results, got {len(yolox_results)}")
    if len(ocr_results) != len(valid_images):
        raise ValueError(f"Expected {len(valid_images)} ocr results, got {len(ocr_results)}")
    return yolox_results, ocr_results


def _merge_chart_results(
    base64_images: List[str],
    valid_indices: List[int],
    yolox_results: List[Any],
    ocr_results: List[Any],
    initial_results: List[Tuple[str, Optional[Dict]]],
) -> List[Tuple[str, Optional[Dict]]]:
    """
    Merge inference results into the initial results list using the original indices.

    For each valid image, processes the results from both inference calls and updates the
    corresponding entry in the results list.
    """
    for idx, (yolox_res, ocr_res) in enumerate(zip(yolox_results, ocr_results)):
        # Unpack ocr result into bounding boxes and text predictions.
        bounding_boxes, text_predictions, _ = ocr_res
        yolox_elements = join_yolox_graphic_elements_and_ocr_output(yolox_res, bounding_boxes, text_predictions)
        chart_content = process_yolox_graphic_elements(yolox_elements)
        original_index = valid_indices[idx]
        initial_results[original_index] = (base64_images[original_index], chart_content)
    return initial_results


def _update_chart_metadata(
    base64_images: List[str],
    yolox_client: Any,
    yolox_model_name: str,
    ocr_client: Any,
    ocr_model_name: str,
    trace_info: Dict,
    worker_pool_size: int = 8,  # Not currently used.
) -> List[Tuple[str, Optional[Dict]]]:
    """
    Given a list of base64-encoded chart images, concurrently call both YOLOX and Paddle
    inference services to extract chart data.

    For each base64-encoded image, returns:
      (original_image_str, joined_chart_content_dict)

    Images that do not meet minimum size requirements are marked as skipped.
    """
    logger.debug("Running chart extraction using updated concurrency handling.")

    # Initialize results with placeholders and filter valid images.
    valid_images, valid_arrays, valid_indices, results = _filter_valid_chart_images(base64_images)

    # Run concurrent inference only for valid images.
    yolox_results, ocr_results = _run_chart_inference(
        yolox_client=yolox_client,
        yolox_model_name=yolox_model_name,
        ocr_client=ocr_client,
        ocr_model_name=ocr_model_name,
        valid_arrays=valid_arrays,
        valid_images=valid_images,
        trace_info=trace_info,
    )

    # Validate that the returned inference results are lists of the expected length.
    yolox_results, ocr_results = _validate_chart_inference_results(
        yolox_results, ocr_results, valid_arrays, valid_images
    )

    # Merge the inference results into the results list.
    return _merge_chart_results(base64_images, valid_indices, yolox_results, ocr_results, results)


def _create_yolox_client(
    yolox_endpoints: Tuple[Optional[str], Optional[str]],
    yolox_protocol: str,
    auth_token: str,
) -> NimClient:
    yolox_model_interface = YoloxGraphicElementsModelInterface(endpoints=yolox_endpoints)

    yolox_client = create_inference_client(
        endpoints=yolox_endpoints,
        model_interface=yolox_model_interface,
        auth_token=auth_token,
        infer_protocol=yolox_protocol,
    )

    return yolox_client


def _create_ocr_client(
    ocr_endpoints: Tuple[Optional[str], Optional[str]],
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


def extract_chart_data_from_image_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Union[IngestTaskChartExtraction, Dict[str, Any]],
    extraction_config: ChartExtractorSchema,
    execution_trace_log: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts chart data from a DataFrame in a bulk fashion rather than row-by-row.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing the content from which chart data is to be extracted.
    task_config : Dict[str, Any]
        Dictionary containing task properties and configurations.
    extraction_config : Any
        The validated configuration object for chart extraction.
    execution_trace_log : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame and the trace information.

    Raises
    ------
    Exception
        If any error occurs during the chart data extraction process.
    """
    _ = task_config  # Unused variable

    if execution_trace_log is None:
        execution_trace_log = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df_extraction_ledger.empty:
        return df_extraction_ledger, execution_trace_log

    endpoint_config = extraction_config.endpoint_config

    try:
        # 1) Identify rows that meet criteria in a single pass
        #    - metadata exists
        #    - content_metadata.type == "structured"
        #    - content_metadata.subtype == "chart"
        #    - table_metadata not None
        #    - base64_image not None or ""
        def meets_criteria(row):
            m = row.get("metadata", {})
            if not m:
                return False

            content_md = m.get("content_metadata", {})
            if (
                content_md.get("type") == "structured"
                and content_md.get("subtype") == "chart"
                and m.get("table_metadata") is not None
                and m.get("content") not in [None, ""]
            ):
                return True

            return False

        mask = df_extraction_ledger.apply(meets_criteria, axis=1)
        valid_indices = df_extraction_ledger[mask].index.tolist()

        # If no rows meet the criteria, just return.
        if not valid_indices:
            return df_extraction_ledger, {"trace_info": execution_trace_log}

        # 2) Extract base64 images + keep track of row -> image mapping.
        base64_images = []
        for idx in valid_indices:
            meta = df_extraction_ledger.at[idx, "metadata"]
            base64_images.append(meta["content"])  # guaranteed by meets_criteria

        # 3) Call our bulk _update_metadata to get all results.
        # Extract endpoints (may be empty/None -> local fallback).
        yolox_endpoints: Tuple[Optional[str], Optional[str]] = (None, None)
        yolox_protocol = "local"
        ocr_endpoints: Tuple[Optional[str], Optional[str]] = (None, None)
        ocr_protocol = "local"
        auth_token = ""
        workers = 5
        if endpoint_config is not None:
            yolox_endpoints = getattr(endpoint_config, "yolox_endpoints", (None, None))
            yolox_protocol = getattr(endpoint_config, "yolox_infer_protocol", "") or "local"
            ocr_endpoints = getattr(endpoint_config, "ocr_endpoints", (None, None))
            ocr_protocol = getattr(endpoint_config, "ocr_infer_protocol", "") or "local"
            auth_token = getattr(endpoint_config, "auth_token", "") or ""
            workers = int(getattr(endpoint_config, "workers_per_progress_engine", 5) or 5)

        has_ocr_endpoint = bool((ocr_endpoints[0] or ocr_endpoints[1]))

        yolox_client = _create_yolox_client(yolox_endpoints, yolox_protocol, auth_token)

        # If OCR endpoints are not configured (or protocol is local), use local Nemotron OCR.
        if (not has_ocr_endpoint) or str(ocr_protocol).lower() == "local":
            ocr_client = _LocalNemotronOCRClient()
            ocr_model_name = "local"
        else:
            # Get the grpc endpoint to determine the model if needed
            ocr_grpc_endpoint = ocr_endpoints[0]  # noqa: F841
            # ocr_model_name = get_ocr_model_name(ocr_grpc_endpoint)
            ocr_model_name = "scene_text_ensemble"
            ocr_client = _create_ocr_client(ocr_endpoints, ocr_protocol, ocr_model_name, auth_token)

        bulk_results = _update_chart_metadata(
            base64_images=base64_images,
            yolox_client=yolox_client,
            yolox_model_name=yolox_client.model_interface.model_name,
            ocr_client=ocr_client,
            ocr_model_name=ocr_model_name,
            worker_pool_size=workers,
            trace_info=execution_trace_log,
        )

        # 4) Write the results back to each row’s table_metadata
        #    The order of base64_images in bulk_results should match their original
        #    indices if we process them in the same order.
        for row_id, idx in enumerate(valid_indices):
            _, chart_content = bulk_results[row_id]
            df_extraction_ledger.at[idx, "metadata"]["table_metadata"]["table_content"] = chart_content

        return df_extraction_ledger, {"trace_info": execution_trace_log}

    except Exception:
        logger.error("Error occurred while extracting chart data.", exc_info=True)

        raise
