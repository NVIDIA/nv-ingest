# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskTableExtraction
from nv_ingest_api.internal.enums.common import TableFormatEnum
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import PaddleOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import NemoRetrieverOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import get_ocr_model_name  # noqa: F401
from nv_ingest_api.internal.primitives.nim import NimClient
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.util.image_processing.table_and_chart import join_yolox_table_structure_and_ocr_output
from nv_ingest_api.util.image_processing.table_and_chart import convert_ocr_response_to_psuedo_markdown
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import YoloxTableStructureModelInterface
from nv_ingest_api.util.image_processing.transforms import base64_to_numpy
from nv_ingest_api.util.nim import create_inference_client

logger = logging.getLogger(__name__)

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32


def _maybe_dump_ocr_input_images(
    *,
    df_extraction_ledger: pd.DataFrame,
    df_indices: List[Any],
    base64_images: List[str],
    backend_tag: str,
) -> None:
    """
    Best-effort debug helper to write OCR input images to disk.

    Controlled via env vars set by higher-level CLIs:
      - NV_INGEST_DUMP_OCR_INPUT_IMAGES=1
      - NV_INGEST_DUMP_OCR_INPUT_IMAGES_DIR=/path/to/dir
      - NV_INGEST_DUMP_OCR_INPUT_IMAGES_PREFIX=<input_filename>

    Filenames:
      <prefix>.page_<PPPP>_<IIII>.<backend_tag>.png
    where IIII is 1-based within each page (pages can have multiple structured images).
    """
    if os.getenv("NV_INGEST_DUMP_OCR_INPUT_IMAGES", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    out_dir = os.getenv("NV_INGEST_DUMP_OCR_INPUT_IMAGES_DIR", "").strip()
    prefix = os.getenv("NV_INGEST_DUMP_OCR_INPUT_IMAGES_PREFIX", "").strip() or "input"
    if not out_dir:
        return

    try:
        from PIL import Image  # type: ignore
    except Exception:
        return

    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    per_page_counts: Dict[int, int] = {}

    for df_idx, b64 in zip(df_indices, base64_images):
        try:
            meta = df_extraction_ledger.at[df_idx, "metadata"]
            page = 0
            if isinstance(meta, dict):
                cm = meta.get("content_metadata") if isinstance(meta.get("content_metadata"), dict) else {}
                p = cm.get("page_number")
                try:
                    page = int(p) if p is not None else 0
                except Exception:
                    page = 0

            per_page_counts[page] = per_page_counts.get(page, 0) + 1
            within_page_i = per_page_counts[page]

            arr = base64_to_numpy(b64)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8) if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.uint8)

            img = Image.fromarray(arr, mode="RGB")
            out_path = Path(out_dir) / f"{prefix}.page_{page:04d}_{within_page_i:04d}.{backend_tag}.png"
            img.save(out_path, format="PNG")
        except Exception:
            continue


def _filter_valid_images(
    base64_images: List[str],
) -> Tuple[List[str], List[np.ndarray], List[int]]:
    """
    Filter base64-encoded images by their dimensions.

    Returns three lists:
      - valid_images: The base64 strings that meet minimum size requirements.
      - valid_arrays: The corresponding numpy arrays.
      - valid_indices: The original indices in the input list.
    """
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
            # Image is too small; skip it.
            continue

    return valid_images, valid_arrays, valid_indices


def _run_inference(
    enable_yolox: bool,
    yolox_client: Any,
    yolox_model_name: str,
    ocr_client: Any,
    ocr_model_name: str,
    valid_arrays: List[np.ndarray],
    valid_images: List[str],
    trace_info: Optional[Dict] = None,
) -> Tuple[List[Any], List[Any]]:
    """
    Run inference concurrently for YOLOX (if enabled) and Paddle.

    Returns a tuple of (yolox_results, ocr_results).
    """
    data_ocr = {"base64_images": valid_images}
    if enable_yolox:
        data_yolox = {"images": valid_arrays}
        future_yolox_kwargs = dict(
            data=data_yolox,
            model_name=yolox_model_name,
            stage_name="table_extraction",
            max_batch_size=8,
            input_names=["INPUT_IMAGES", "THRESHOLDS"],
            dtypes=["BYTES", "FP32"],
            output_names=["OUTPUT"],
            trace_info=trace_info,
        )

    future_ocr_kwargs = dict(
        data=data_ocr,
        stage_name="table_extraction",
        trace_info=trace_info,
    )
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
            merge_level="word",
        )
    else:
        raise ValueError(f"Unknown OCR model name: {ocr_model_name}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ocr = executor.submit(ocr_client.infer, **future_ocr_kwargs)
        future_yolox = None
        if enable_yolox:
            future_yolox = executor.submit(yolox_client.infer, **future_yolox_kwargs)
        if enable_yolox:
            try:
                yolox_results = future_yolox.result()
            except Exception as e:
                logger.error(f"Error calling yolox_client.infer: {e}", exc_info=True)
                raise
        else:
            yolox_results = [None] * len(valid_images)

        try:
            ocr_results = future_ocr.result()
        except Exception as e:
            logger.error(f"Error calling ocr_client.infer: {e}", exc_info=True)
            raise

    return yolox_results, ocr_results


def _validate_inference_results(
    yolox_results: Any,
    ocr_results: Any,
    valid_arrays: List[Any],
    valid_images: List[str],
) -> Tuple[List[Any], List[Any]]:
    """
    Validate that both inference results are lists and have the expected lengths.

    If not, default values are assigned. Raises a ValueError if the lengths do not match.
    """
    if not isinstance(yolox_results, list) or not isinstance(ocr_results, list):
        logger.warning(
            "Unexpected result types from inference clients: yolox_results=%s, ocr_results=%s. "
            "Proceeding with available results.",
            type(yolox_results).__name__,
            type(ocr_results).__name__,
        )
        if not isinstance(yolox_results, list):
            yolox_results = [None] * len(valid_arrays)
        if not isinstance(ocr_results, list):
            ocr_results = [(None, None)] * len(valid_images)

    if len(yolox_results) != len(valid_arrays):
        raise ValueError(f"Expected {len(valid_arrays)} yolox results, got {len(yolox_results)}")
    if len(ocr_results) != len(valid_images):
        raise ValueError(f"Expected {len(valid_images)} ocr results, got {len(ocr_results)}")

    return yolox_results, ocr_results


def _update_table_metadata(
    base64_images: List[str],
    yolox_client: Any,
    yolox_model_name: str,
    ocr_client: Any,
    ocr_model_name: str,
    worker_pool_size: int = 8,  # Not currently used
    enable_yolox: bool = False,
    trace_info: Optional[Dict] = None,
) -> List[Tuple[str, Any, Any, Any]]:
    """
    Given a list of base64-encoded images, this function filters out images that do not meet
    the minimum size requirements and then calls the OCR model via ocr_client.infer
    to extract table data.

    For each base64-encoded image, the result is a tuple:
        (base64_image, yolox_result, ocr_text_predictions, ocr_bounding_boxes)

    Images that do not meet the minimum size are skipped (resulting in placeholders).
    The ocr_client is expected to handle any necessary batching and concurrency.
    """
    logger.debug(f"Running table extraction using protocol {ocr_client.protocol}")

    # Initialize the results list with default placeholders.
    results: List[Tuple[str, Any, Any, Any]] = [("", None, None, None)] * len(base64_images)

    # Filter valid images based on size requirements.
    valid_images, valid_arrays, valid_indices = _filter_valid_images(base64_images)

    if not valid_images:
        return results

    # Run inference concurrently.
    yolox_results, ocr_results = _run_inference(
        enable_yolox=enable_yolox,
        yolox_client=yolox_client,
        yolox_model_name=yolox_model_name,
        ocr_client=ocr_client,
        ocr_model_name=ocr_model_name,
        valid_arrays=valid_arrays,
        valid_images=valid_images,
        trace_info=trace_info,
    )

    # Validate that the inference results have the expected structure.
    yolox_results, ocr_results = _validate_inference_results(yolox_results, ocr_results, valid_arrays, valid_images)

    # Combine results with the original order.
    for idx, (yolox_res, ocr_res) in enumerate(zip(yolox_results, ocr_results)):
        original_index = valid_indices[idx]
        results[original_index] = (
            base64_images[original_index],
            yolox_res,
            ocr_res[0],
            ocr_res[1],
        )

    return results


def _create_yolox_client(
    yolox_endpoints: Tuple[str, str],
    yolox_protocol: str,
    auth_token: str,
) -> NimClient:
    yolox_model_interface = YoloxTableStructureModelInterface(endpoints=yolox_endpoints)

    yolox_client = create_inference_client(
        endpoints=yolox_endpoints,
        model_interface=yolox_model_interface,
        auth_token=auth_token,
        infer_protocol=yolox_protocol,
    )

    return yolox_client


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


def _local_nemotron_ocr_boxes_texts(
    base64_images: List[str],
    *,
    merge_level: str = "word",
    trace_info: Optional[Dict] = None,
) -> List[Tuple[str, Any, Any, Any]]:
    """
    Local OCR fallback using the Nemotron OCR v1 pipeline via:
      `retriever.model.local.nemotron_ocr_v1.NemotronOCRV1`

    Returns a list aligned with base64_images:
      (base64_image, cell_predictions=None, bounding_boxes, text_predictions)

    Where:
      - bounding_boxes: List[List[List[float]]] (quadrilateral boxes [[x,y] * 4])
      - text_predictions: List[str]
    """
    # Preserve the same "skip tiny images" behavior as the NIM path.
    valid_images, valid_arrays, valid_indices = _filter_valid_images(base64_images)

    # Initialize defaults for all images (including those skipped).
    results: List[Tuple[str, Any, Any, Any]] = [(img, None, None, None) for img in base64_images]
    if not valid_images:
        return results

    model_dir = (
        os.getenv("RETRIEVER_NEMOTRON_OCR_MODEL_DIR", "").strip()
        or os.getenv("NEMOTRON_OCR_MODEL_DIR", "").strip()
        or os.getenv("NEMOTRON_OCR_V1_MODEL_DIR", "").strip()
    )

    # Lazy import to avoid hard dependency when running pure API package.
    try:
        from retriever.model.local.nemotron_ocr_v1 import NemotronOCRV1  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Local table OCR fallback requires the `retriever` package to be importable "
            "so we can use `retriever.model.local.nemotron_ocr_v1.NemotronOCRV1`."
        ) from e

    if trace_info is not None:
        trace_info.setdefault("ocr", {})
        trace_info["ocr"]["backend"] = "local_nemotron_ocr_v1"
        trace_info["ocr"]["model_dir"] = model_dir or None

    # Instantiate local OCR model once per call.
    ocr = NemotronOCRV1(model_dir=model_dir) if model_dir else NemotronOCRV1()

    def _xyxy_to_quad(xyxy: List[float]) -> List[List[float]]:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    for local_i, (b64, arr) in enumerate(zip(valid_images, valid_arrays)):
        original_index = valid_indices[local_i]
        try:
            h, w = int(arr.shape[0]), int(arr.shape[1])

            preds = ocr.invoke(b64, merge_level=merge_level)

            boxes: List[List[List[float]]] = []
            texts: List[str] = []

            # Common "packed" form: dict with boxes/texts.
            if isinstance(preds, dict):
                pb = preds.get("boxes") or preds.get("bboxes") or preds.get("bounding_boxes")
                pt = preds.get("texts") or preds.get("text_predictions") or preds.get("text")
                if isinstance(pb, list) and isinstance(pt, list):
                    for b, txt in zip(pb, pt):
                        if isinstance(txt, str) and txt.strip():
                            texts.append(txt.strip())
                            if isinstance(b, list):
                                # b may be [[x,y]*4] or flat len 8 or xyxy len 4
                                if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                                    boxes.append([[float(x), float(y)] for x, y in b])  # type: ignore[misc]
                                elif len(b) == 8 and all(isinstance(v, (int, float)) for v in b):
                                    pts = [float(v) for v in b]
                                    boxes.append(
                                        [[pts[0], pts[1]], [pts[2], pts[3]], [pts[4], pts[5]], [pts[6], pts[7]]]
                                    )
                                elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                                    boxes.append(_xyxy_to_quad([float(v) for v in b]))

            # Per-region list form: list[dict]
            if (not texts) and isinstance(preds, list):
                for item in preds:
                    if isinstance(item, str):
                        if item.strip():
                            texts.append(item.strip())
                            boxes.append([[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]])
                        continue
                    if not isinstance(item, dict):
                        continue

                    # Nemotron OCR pipeline returns normalized coords: left/right/lower/upper in [0,1]
                    if all(k in item for k in ("left", "right", "upper", "lower")) and isinstance(
                        item.get("text"), str
                    ):
                        txt0 = str(item.get("text") or "").strip()
                        if not txt0 or txt0 == "nan":
                            continue
                        try:
                            x1 = float(item["left"]) * float(w)
                            x2 = float(item["right"]) * float(w)
                            y1 = float(item["lower"]) * float(h)
                            y2 = float(item["upper"]) * float(h)
                            quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        except Exception:
                            quad = [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]
                        texts.append(txt0)
                        boxes.append(quad)
                        continue

                    txt = (
                        item.get("text")
                        or item.get("ocr_text")
                        or item.get("generated_text")
                        or item.get("output_text")
                    )
                    if not isinstance(txt, str) or not txt.strip():
                        continue
                    txt = txt.strip()

                    b = item.get("box") or item.get("bbox") or item.get("bounding_box") or item.get("bbox_points")
                    quad: Optional[List[List[float]]] = None
                    if isinstance(b, list):
                        if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                            quad = [[float(x), float(y)] for x, y in b]  # type: ignore[misc]
                        elif len(b) == 8 and all(isinstance(v, (int, float)) for v in b):
                            pts = [float(v) for v in b]
                            quad = [[pts[0], pts[1]], [pts[2], pts[3]], [pts[4], pts[5]], [pts[6], pts[7]]]
                        elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                            quad = _xyxy_to_quad([float(v) for v in b])
                    elif isinstance(b, dict):
                        pts = b.get("points")
                        if isinstance(pts, list) and len(pts) == 4:
                            try:
                                quad = [[float(p.get("x")), float(p.get("y"))] for p in pts]  # type: ignore[union-attr]
                            except Exception:
                                quad = None

                    if quad is None:
                        quad = [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]

                    texts.append(txt)
                    boxes.append(quad)

            # If we still didn't parse anything, best-effort stringify.
            if not texts:
                s = ""
                try:
                    s = str(preds).strip()
                except Exception:
                    s = ""
                if s and s.lower() not in {"none", "null"}:
                    texts = [s]
                    boxes = [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]  # type: ignore[assignment]  # noqa: E501

            if texts and not boxes:
                # Provide a dummy full-image box per text.
                boxes = [[[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]] for _ in texts]

            results[original_index] = (base64_images[original_index], None, boxes, texts)
        except Exception as ex:  # noqa: F841
            logger.exception("Local Nemotron OCR failed for table image index=%s", original_index)
            results[original_index] = (base64_images[original_index], None, None, None)

    return results


def _local_nemotron_table_structure_cell_predictions(
    base64_images: List[str],
    *,
    trace_info: Optional[Dict] = None,
) -> List[Optional[Dict[str, Any]]]:
    """
    Local table-structure fallback using:
      `retriever.model.local.nemotron_table_structure_v1.NemotronTableStructureV1`

    Returns a list aligned with base64_images where each element is either:
      - None (failed / skipped), or
      - dict with keys {"cell","row","column"} mapping to lists of [x1,y1,x2,y2,score] in **pixel** coords.

    This matches what `join_yolox_table_structure_and_ocr_output()` expects (it treats boxes as Nx4/Nx5).
    """
    valid_images, valid_arrays, valid_indices = _filter_valid_images(base64_images)
    out: List[Optional[Dict[str, Any]]] = [None for _ in base64_images]
    if not valid_images:
        return out

    # Lazy import to avoid hard dependency when running pure API package.
    try:
        from retriever.model.local.nemotron_table_structure_v1 import NemotronTableStructureV1  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Local table-structure fallback requires the `retriever` package to be importable "
            "so we can use `retriever.model.local.nemotron_table_structure_v1.NemotronTableStructureV1`."
        ) from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError("Local table-structure fallback requires PyTorch to be installed.") from e

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if trace_info is not None:
        trace_info.setdefault("yolox", {})
        trace_info["yolox"]["backend"] = "local_nemotron_table_structure_v1"
        trace_info["yolox"]["device"] = str(dev)

    model = NemotronTableStructureV1(endpoint=None)
    # Best-effort: move underlying module to device if present.
    try:
        if hasattr(model, "_model") and model._model is not None:
            model._model = model._model.to(dev)  # type: ignore[attr-defined]
    except Exception:
        pass

    def _to_list(v: Any) -> Any:
        try:
            import numpy as _np

            if isinstance(v, torch.Tensor):
                return v.detach().cpu().tolist()
            if isinstance(v, _np.ndarray):
                return v.tolist()
        except Exception:
            pass
        return v

    # Label mapping fallback if model emits integer ids.
    id_to_label = {0: "cell", 1: "row", 2: "column"}

    for local_i, arr in enumerate(valid_arrays):
        idx = valid_indices[local_i]
        try:
            # Ensure RGB uint8.
            if arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8) if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.uint8)

            h, w = int(arr.shape[0]), int(arr.shape[1])
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            t = t.to(device=dev, dtype=torch.uint8, non_blocking=(dev.type == "cuda"))

            x = model.preprocess(t, (h, w))
            with torch.inference_mode():
                with torch.autocast(device_type="cuda"):
                    preds = model.invoke(x, (h, w))

            # Normalize to (boxes, labels, scores)
            boxes = None
            labels = None
            scores = None

            if isinstance(preds, dict) and all(k in preds for k in ("boxes", "labels", "scores")):
                boxes = preds.get("boxes")
                labels = preds.get("labels")
                scores = preds.get("scores")
            else:
                try:
                    b, l, s = model.postprocess(preds)  # type: ignore[arg-type]
                    boxes, labels, scores = b, l, s
                except Exception:
                    boxes = None

            if boxes is None or labels is None or scores is None:
                out[idx] = None
                continue

            boxes_l = _to_list(boxes)
            labels_l = _to_list(labels)
            scores_l = _to_list(scores)

            # Flatten labels/scores shapes.
            if isinstance(labels_l, list) and labels_l and isinstance(labels_l[0], list):
                labels_l = [x[0] if isinstance(x, list) and x else x for x in labels_l]
            if isinstance(scores_l, list) and scores_l and isinstance(scores_l[0], list):
                scores_l = [x[0] if isinstance(x, list) and x else x for x in scores_l]

            # Detect normalized coordinates.
            mx = 0.0
            try:
                mx = float(max(max(b) for b in boxes_l)) if isinstance(boxes_l, list) and boxes_l else 0.0
            except Exception:
                mx = 0.0
            normalized = mx <= 1.5

            preds_by_label: Dict[str, List[List[float]]] = {"cell": [], "row": [], "column": []}
            for b, lab, sc in zip(boxes_l, labels_l, scores_l):
                if not (isinstance(b, list) and len(b) >= 4):
                    continue
                x1, y1, x2, y2 = [float(v) for v in b[:4]]
                if normalized:
                    x1, x2 = x1 * w, x2 * w
                    y1, y2 = y1 * h, y2 * h
                score = float(sc) if sc is not None else 0.0

                label_name = None
                if isinstance(lab, str):
                    label_name = lab
                else:
                    try:
                        label_name = id_to_label.get(int(lab))
                    except Exception:
                        label_name = None
                if label_name not in preds_by_label:
                    # Ignore unknown classes for table markdown joining.
                    continue
                preds_by_label[label_name].append([x1, y1, x2, y2, score])

            out[idx] = preds_by_label
        except Exception:
            logger.exception("Local Nemotron table-structure failed for index=%s", idx)
            out[idx] = None

    return out


def extract_table_data_from_image_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Union[IngestTaskTableExtraction, Dict[str, Any]],
    extraction_config: TableExtractorSchema,
    execution_trace_log: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts table data from a DataFrame in a bulk fashion rather than row-by-row,
    following the chart extraction pattern.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing the content from which table data is to be extracted.
    task_config : Dict[str, Any]
        Dictionary containing task properties and configurations.
    extraction_config : Any
        The validated configuration object for table extraction.
    execution_trace_log : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame and the trace information.
    """

    _ = task_config  # unused

    if execution_trace_log is None:
        execution_trace_log = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df_extraction_ledger.empty:
        return df_extraction_ledger, execution_trace_log

    endpoint_config = extraction_config.endpoint_config

    try:
        # 1) Identify rows that meet criteria (structured, subtype=table, table_metadata != None, content not empty)
        def meets_criteria(row):
            m = row.get("metadata", {})
            if not m:
                return False
            content_md = m.get("content_metadata", {})
            if (
                content_md.get("type") == "structured"
                and content_md.get("subtype") == "table"
                and m.get("table_metadata") is not None
                and m.get("content") not in [None, ""]
            ):
                return True
            return False

        mask = df_extraction_ledger.apply(meets_criteria, axis=1)
        valid_indices = df_extraction_ledger[mask].index.tolist()

        # If no rows meet the criteria, just return
        if not valid_indices:
            return df_extraction_ledger, {"trace_info": execution_trace_log}

        # 2) Extract base64 images in the same order
        base64_images = []
        for idx in valid_indices:
            meta = df_extraction_ledger.at[idx, "metadata"]
            base64_images.append(meta["content"])

        # 3) Call our bulk _update_metadata to get all results
        table_content_format = (
            df_extraction_ledger.at[valid_indices[0], "metadata"]["table_metadata"].get("table_content_format")
            or TableFormatEnum.PSEUDO_MARKDOWN
        )
        enable_yolox = True if table_content_format in (TableFormatEnum.MARKDOWN,) else False

        # Extract endpoints (may be empty/None -> local fallback).
        yolox_endpoints = (None, None)
        yolox_protocol = "local"
        ocr_endpoints = (None, None)
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

        has_yolox_endpoint = bool((yolox_endpoints[0] or yolox_endpoints[1]))
        has_ocr_endpoint = bool((ocr_endpoints[0] or ocr_endpoints[1]))

        # If markdown format requested but no YOLOX endpoints, use local table-structure model.
        local_yolox_preds: Optional[List[Optional[Dict[str, Any]]]] = None
        if enable_yolox and not has_yolox_endpoint:
            local_yolox_preds = _local_nemotron_table_structure_cell_predictions(
                base64_images,
                trace_info=execution_trace_log,
            )

        # If OCR endpoints are not configured (or protocol is local), use local Nemotron OCR.
        use_local_ocr = (not has_ocr_endpoint) or str(ocr_protocol).lower() == "local"
        if use_local_ocr:
            # Dump the exact images we will pass to OCR (after size filtering).
            try:
                valid_imgs, _valid_arrs, valid_pos = _filter_valid_images(base64_images)
                df_idxs_for_valid = [valid_indices[p] for p in valid_pos]
                _maybe_dump_ocr_input_images(
                    df_extraction_ledger=df_extraction_ledger,
                    df_indices=df_idxs_for_valid,
                    base64_images=valid_imgs,
                    backend_tag="hf",
                )
            except Exception:
                pass
            bulk_results = _local_nemotron_ocr_boxes_texts(
                base64_images,
                merge_level="paragraph",
                trace_info=execution_trace_log,
            )
        else:
            # Dump the exact images we will pass to OCR (after size filtering).
            try:
                valid_imgs, _valid_arrs, valid_pos = _filter_valid_images(base64_images)
                df_idxs_for_valid = [valid_indices[p] for p in valid_pos]
                _maybe_dump_ocr_input_images(
                    df_extraction_ledger=df_extraction_ledger,
                    df_indices=df_idxs_for_valid,
                    base64_images=valid_imgs,
                    backend_tag="nim",
                )
            except Exception:
                pass
            # Get the grpc endpoint to determine the model if needed
            ocr_grpc_endpoint = ocr_endpoints[0]  # noqa: F841
            # ocr_model_name = get_ocr_model_name(ocr_grpc_endpoint)
            ocr_model_name = "scene_text_ensemble"

            yolox_client = _create_yolox_client(
                yolox_endpoints,
                yolox_protocol,
                auth_token,
            )
            ocr_client = _create_ocr_client(
                ocr_endpoints,
                ocr_protocol,
                ocr_model_name,
                auth_token,
            )

            bulk_results = _update_table_metadata(
                base64_images=base64_images,
                yolox_client=yolox_client,
                yolox_model_name=yolox_client.model_interface.model_name,
                ocr_client=ocr_client,
                ocr_model_name=ocr_model_name,
                worker_pool_size=workers,
                enable_yolox=enable_yolox,
                trace_info=execution_trace_log,
            )

        # If we ran local table-structure, splice its predictions into bulk_results.
        if local_yolox_preds is not None:
            merged: List[Tuple[str, Any, Any, Any]] = []
            for i, rec in enumerate(bulk_results):
                try:
                    b64_img, _cell_preds, bbox, txts = rec
                except Exception:
                    merged.append(rec)
                    continue
                cell_preds = local_yolox_preds[i]
                merged.append((b64_img, cell_preds, bbox, txts))
            bulk_results = merged

        # 4) Write the results (bounding_boxes, text_predictions) back
        for row_id, idx in enumerate(valid_indices):
            # unpack (base64_image, (yolox_predictions, ocr_bounding boxes, ocr_text_predictions))
            _, cell_predictions, bounding_boxes, text_predictions = bulk_results[row_id]

            if table_content_format == TableFormatEnum.SIMPLE:
                table_content = " ".join(text_predictions) if text_predictions else ""
            elif table_content_format == TableFormatEnum.PSEUDO_MARKDOWN:
                table_content = convert_ocr_response_to_psuedo_markdown(bounding_boxes, text_predictions)
            elif table_content_format == TableFormatEnum.MARKDOWN:
                table_content = join_yolox_table_structure_and_ocr_output(
                    cell_predictions, bounding_boxes, text_predictions
                )
            else:
                raise ValueError(f"Unexpected table format: {table_content_format}")

            df_extraction_ledger.at[idx, "metadata"]["table_metadata"]["table_content"] = table_content
            df_extraction_ledger.at[idx, "metadata"]["table_metadata"]["table_content_format"] = table_content_format

        return df_extraction_ledger, {"trace_info": execution_trace_log}

    except Exception:
        logger.exception("Error occurred while extracting table data.", exc_info=True)
        raise
