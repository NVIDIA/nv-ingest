# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Crop-based OCR stage.

Runs Nemotron OCR on table / chart / infographic regions detected by
PageElements v3. Text extraction for the full page is handled upstream
by PDFium in the PDF extraction stage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import base64
import io
import logging
import time
import traceback

_logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from requests import HTTPError
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.models.nim.nim import NIMClient, invoke_image_inference_batches
from nemo_retriever.common.modality.table_and_chart import join_table_structure_and_ocr_output

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Page-element labels that carry running text (as opposed to structured
# content like tables/charts/infographics).  Used by the OCR stage to
# decide which detections contribute to the page's ``text`` column.
_TEXT_LABELS: frozenset[str] = frozenset({"text", "title", "header_footer"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "timing": None,
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _crop_b64_image_by_norm_bbox(
    page_image_b64: str,
    *,
    bbox_xyxy_norm: Sequence[float],
    image_format: str = "png",
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """
    Crop a base64-encoded RGB image by a normalized xyxy bbox.

    Returns
    -------
    cropped_image_b64 : str | None
        Base64-encoded cropped image (PNG), or *None* on failure.
    cropped_shape_hw : tuple[int, int] | None
        (H, W) of the crop, or *None* on failure.
    """
    if Image is None:  # pragma: no cover
        raise ImportError("Cropping requires pillow.")

    if not isinstance(page_image_b64, str) or not page_image_b64:
        return None, None
    try:
        x1n, y1n, x2n, y2n = [float(x) for x in bbox_xyxy_norm]
    except Exception:
        return None, None

    try:
        raw = base64.b64decode(page_image_b64)
        with Image.open(io.BytesIO(raw)) as im0:
            im = im0.convert("RGB")
            w, h = im.size
            if w <= 1 or h <= 1:
                return None, None

            def _clamp_int(v: float, lo: int, hi: int) -> int:
                if v != v:  # NaN
                    return lo
                return int(min(max(v, float(lo)), float(hi)))

            x1 = _clamp_int(x1n * w, 0, w)
            x2 = _clamp_int(x2n * w, 0, w)
            y1 = _clamp_int(y1n * h, 0, h)
            y2 = _clamp_int(y2n * h, 0, h)

            if x2 <= x1 or y2 <= y1:
                return None, None

            crop = im.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw <= 1 or ch <= 1:
                return None, None

            buf = io.BytesIO()
            fmt = str(image_format or "png").lower()
            if fmt not in {"png"}:
                fmt = "png"
            crop.save(buf, format=fmt.upper())
            return base64.b64encode(buf.getvalue()).decode("ascii"), (int(ch), int(cw))
    except Exception:
        return None, None


def _crop_all_from_page(
    page_image_b64: str,
    detections: List[Dict[str, Any]],
    wanted_labels: set,
    *,
    as_b64: bool = False,
) -> List[Tuple[str, List[float], Any]]:
    """
    Decode the page image **once** and crop all matching detections.

    Returns a list of ``(label_name, bbox_xyxy_norm, value)`` tuples for
    detections whose ``label_name`` is in *wanted_labels* and whose crop is
    valid.  Skips detections that fail to crop (bad bbox, tiny region, etc.).

    When *as_b64* is ``False`` (default), *value* is an HWC uint8 numpy array
    suitable for local model inference.  When ``True``, *value* is a base64-
    encoded PNG string — this avoids a wasteful numpy→PIL→PNG round-trip on
    the remote inference path.
    """
    if Image is None:  # pragma: no cover
        raise ImportError("Cropping requires pillow.")

    if not isinstance(page_image_b64, str) or not page_image_b64:
        return []

    try:
        raw = base64.b64decode(page_image_b64)
        im0 = Image.open(io.BytesIO(raw))
        im = im0.convert("RGB")
        im0.close()
    except Exception:
        return []

    w, h = im.size
    if w <= 1 or h <= 1:
        im.close()
        return []

    def _clamp_int(v: float, lo: int, hi: int) -> int:
        if v != v:  # NaN
            return lo
        return int(min(max(v, float(lo)), float(hi)))

    results: List[Tuple[str, List[float], Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label_name = str(det.get("label_name") or "").strip()
        if label_name not in wanted_labels:
            continue

        bbox = det.get("bbox_xyxy_norm")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            x1n, y1n, x2n, y2n = [float(x) for x in bbox]
        except Exception:
            continue

        x1 = _clamp_int(x1n * w, 0, w)
        x2 = _clamp_int(x2n * w, 0, w)
        y1 = _clamp_int(y1n * h, 0, h)
        y2 = _clamp_int(y2n * h, 0, h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = im.crop((x1, y1, x2, y2))
        cw, ch = crop.size
        if cw <= 1 or ch <= 1:
            crop.close()
            continue

        if as_b64:
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop.close()
            value = base64.b64encode(buf.getvalue()).decode("ascii")
        else:
            value = np.asarray(crop, dtype=np.uint8).copy()
            crop.close()
        results.append((label_name, [float(x) for x in bbox], value))

    im.close()
    return results


def _np_rgb_to_b64_png(crop_array: np.ndarray) -> str:
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required for image encoding.")
    img = Image.fromarray(crop_array.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_remote_ocr_item(response_item: Any) -> Any:
    if isinstance(response_item, dict):
        # NIM text_detections format: return full list (not v[0])
        td = response_item.get("text_detections")
        if isinstance(td, list) and td:
            return td
        for k in ("prediction", "predictions", "output", "outputs", "data"):
            v = response_item.get(k)
            if isinstance(v, list) and v:
                return v[0]
            if v is not None:
                return v
    return response_item


def _parse_ocr_result(preds: Any) -> List[Dict[str, Any]]:
    """
    Parse the output of ``NemotronOCRV1.invoke()`` into a flat list of
    ``{"text": str, "sort_y": float, "sort_x": float}`` blocks.

    The model may return:
    * ``dict`` with ``boxes`` / ``texts`` keys (packed form), or
    * ``list[dict]`` with ``left``/``right``/``upper``/``lower``/``text`` keys
      (Nemotron OCR normalized-coord form), or
    * ``list[dict]`` with generic ``text`` + ``box``/``bbox`` keys.
    """
    blocks: List[Dict[str, Any]] = []

    # ---- dict form: {"boxes": [...], "texts": [...]} ----
    if isinstance(preds, dict):
        pb = preds.get("boxes") or preds.get("bboxes") or preds.get("bounding_boxes")
        pt = preds.get("texts") or preds.get("text_predictions") or preds.get("text")
        if isinstance(pb, list) and isinstance(pt, list):
            for b, txt in zip(pb, pt):
                if not isinstance(txt, str) or not txt.strip():
                    continue
                sort_y, sort_x = 0.0, 0.0
                if isinstance(b, list):
                    if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                        # quadrilateral [[x1,y1], ...]
                        sort_y = float(b[0][1])
                        sort_x = float(b[0][0])
                    elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                        # xyxy [x1, y1, x2, y2]
                        sort_y = float(b[1])
                        sort_x = float(b[0])
                blocks.append({"text": txt.strip(), "sort_y": sort_y, "sort_x": sort_x})
        return blocks

    # ---- list form: list[dict] with various key conventions ----
    if isinstance(preds, list):
        for item in preds:
            if isinstance(item, str):
                if item.strip():
                    blocks.append({"text": item.strip(), "sort_y": 0.0, "sort_x": 0.0})
                continue
            if not isinstance(item, dict):
                continue

            # NIM text_detections format:
            # {"text_prediction": {"text": "...", "confidence": ...},
            #  "bounding_box": {"points": [{"x": ..., "y": ...}, ...]}}
            tp = item.get("text_prediction")
            if isinstance(tp, dict):
                txt0 = str(tp.get("text") or "").strip()
                if txt0 and txt0 != "nan":
                    sort_y, sort_x = 0.0, 0.0
                    bb = item.get("bounding_box")
                    if isinstance(bb, dict):
                        pts = bb.get("points")
                        if isinstance(pts, list) and pts:
                            try:
                                sort_x = float(pts[0].get("x", 0.0))
                                sort_y = float(pts[0].get("y", 0.0))
                            except Exception:
                                pass
                    blocks.append({"text": txt0, "sort_y": sort_y, "sort_x": sort_x})
                continue

            # Nemotron OCR normalized-coord form
            if all(k in item for k in ("left", "right", "upper", "lower")) and isinstance(item.get("text"), str):
                txt0 = str(item.get("text") or "").strip()
                if not txt0 or txt0 == "nan":
                    continue
                try:
                    sort_x = float(item["left"])
                    sort_y = float(item["lower"])
                except Exception:
                    sort_x, sort_y = 0.0, 0.0
                blocks.append({"text": txt0, "sort_y": sort_y, "sort_x": sort_x})
                continue

            # Generic text + box fallback
            txt = item.get("text") or item.get("ocr_text") or item.get("generated_text") or item.get("output_text")
            if not isinstance(txt, str) or not txt.strip():
                continue
            sort_y, sort_x = 0.0, 0.0
            b = item.get("box") or item.get("bbox") or item.get("bounding_box") or item.get("bbox_points")
            if isinstance(b, list):
                if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                    sort_y = float(b[0][1])
                    sort_x = float(b[0][0])
                elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                    sort_y = float(b[1])
                    sort_x = float(b[0])
            blocks.append({"text": txt.strip(), "sort_y": sort_y, "sort_x": sort_x})

    # ---- last-resort stringify ----
    if not blocks and preds is not None:
        s = ""
        try:
            s = str(preds).strip()
        except Exception:
            s = ""
        if s and s.lower() not in {"none", "null", "[]", "{}"}:
            # ``_fallback`` lets ``ocr_response_to_text`` suppress this row
            # without false-positives on legitimate OCR text that happens to
            # start with ``[`` or ``{``.
            blocks.append({"text": s, "sort_y": 0.0, "sort_x": 0.0, "_fallback": True})

    return blocks


def _blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """Sort text blocks by reading order (y then x) and join with whitespace."""
    blocks.sort(key=lambda b: (b.get("sort_y", 0.0), b.get("sort_x", 0.0)))
    return " ".join(b["text"] for b in blocks if b.get("text"))


def ocr_response_to_text(preds: Any) -> str:
    """Extract joined OCR text from a Nemotron OCR response, returning ``""``
    when no text is detected.

    Wraps :func:`_parse_ocr_result` + :func:`_blocks_to_text` but suppresses
    ``_parse_ocr_result``'s last-resort stringify fallback (which dumps the
    raw response repr when no shape matches) — that fallback produces noise
    rows for callers like the video frame OCR actor where many frames have
    no on-screen text and we need an empty-string sentinel to drop them.
    """
    blocks = _parse_ocr_result(preds)
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("_fallback"):
        return ""
    return _blocks_to_text(blocks)


def ocr_b64_to_text(
    image_b64_list: Sequence[str],
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    nim_client: Optional[NIMClient] = None,
    merge_level: str = "paragraph",
    batch_size: int = 8,
    timeout_s: float = 120.0,
    retry: Optional[RemoteRetryParams] = None,
) -> List[str]:
    """Run Nemotron OCR on a list of base64 PNG images; return one text per input.

    Routes to remote NIM when ``invoke_url`` is set (uses ``nim_client`` if
    provided, otherwise spins up a fresh batched call), or to a local
    Nemotron OCR model when ``model`` is provided. Empty/non-string
    inputs map to ``""``; per-image parse failures are logged and also map
    to ``""`` so the output preserves input order and length.
    """
    n = len(image_b64_list)
    if n == 0:
        return []

    use_remote = bool((invoke_url or "").strip())
    if not use_remote and model is None:
        raise ValueError("ocr_b64_to_text requires either invoke_url or a local model.")

    valid_idx = [i for i, b in enumerate(image_b64_list) if isinstance(b, str) and b]
    valid_b64 = [image_b64_list[i] for i in valid_idx]
    out = [""] * n
    if not valid_b64:
        return out

    retry_params = retry or RemoteRetryParams()

    if use_remote:
        try:
            invoke_kw = dict(
                invoke_url=invoke_url,
                image_b64_list=valid_b64,
                api_key=api_key,
                timeout_s=float(timeout_s),
                max_batch_size=int(batch_size),
                max_retries=int(retry_params.remote_max_retries),
                max_429_retries=int(retry_params.remote_max_429_retries),
            )
            if nim_client is not None:
                response_items = nim_client.invoke_image_inference_batches(**invoke_kw)
            else:
                response_items = invoke_image_inference_batches(
                    **invoke_kw,
                    max_pool_workers=int(retry_params.remote_max_pool_workers),
                )
        except Exception:
            _logger.exception("Remote OCR call failed")
            return out
        for resp, dst in zip(response_items, valid_idx):
            try:
                preds = _extract_remote_ocr_item(resp)
                out[dst] = ocr_response_to_text(preds)
            except Exception:
                _logger.warning("Failed to parse OCR response for index %d", dst)
                out[dst] = ""
        return out

    # Local model path.
    for b64, dst in zip(valid_b64, valid_idx):
        try:
            preds = model.invoke(b64.encode("utf-8"), merge_level=merge_level)
            out[dst] = ocr_response_to_text(preds)
        except Exception:
            _logger.exception("Local OCR failed on image at index %d", dst)
            out[dst] = ""
    return out


def split_ocrable_rows(
    batch_df: pd.DataFrame,
    ocrable_content_types: Sequence[str] = ("",),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partition rows into OCR-able rows and passthrough rows.

    Rows whose ``_content_type`` is in ``ocrable_content_types`` are treated
    as OCR-able; all others are passed through unchanged. Batches with no
    ``_content_type`` column at all (audio-free pipelines like PDF / image)
    are fully OCR-able. The default accepts only the empty discriminator;
    pipelines that mix OCR-able and non-OCR-able rows (e.g. video, which
    interleaves ``video_frame`` and ``audio``) pass their own sentinel set.
    """
    if "_content_type" not in batch_df.columns:
        return batch_df.copy(), pd.DataFrame()
    ct = batch_df["_content_type"].astype(str).fillna("")
    ocr_mask = ct.isin(list(ocrable_content_types))
    return (
        batch_df[ocr_mask].reset_index(drop=True),
        batch_df[~ocr_mask].reset_index(drop=True),
    )


def concat_with_passthrough(processed: pd.DataFrame, passthrough: pd.DataFrame) -> pd.DataFrame:
    """Concat the OCR-stage output with passthrough rows, harmonising columns."""
    if passthrough is None or passthrough.empty:
        return processed
    if processed is None or processed.empty:
        return passthrough
    for col in processed.columns:
        if col not in passthrough.columns:
            passthrough = passthrough.assign(**{col: None})
    for col in passthrough.columns:
        if col not in processed.columns:
            processed = processed.assign(**{col: None})
    return pd.concat([processed[passthrough.columns.tolist()], passthrough], ignore_index=True, sort=False)


def is_full_image_batch(batch_df: pd.DataFrame) -> bool:
    """True when the batch carries top-level ``image_b64`` and no usable
    ``page_elements_v3`` — i.e. the input came from frame extraction
    (or any other producer that hands raw images straight to OCR)."""
    if "image_b64" not in batch_df.columns:
        return False
    pe = batch_df.get("page_elements_v3")
    if pe is None:
        return True
    return not pe.notna().any()


def full_image_ocr_df(
    batch_df: pd.DataFrame,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    nim_client: Optional[NIMClient] = None,
    merge_level: str = "paragraph",
    batch_size: int = 8,
    timeout_s: float = 120.0,
    retry: Optional[RemoteRetryParams] = None,
) -> pd.DataFrame:
    """Run full-image OCR on a DataFrame whose rows carry top-level ``image_b64``.

    Writes ``text`` and drops rows whose OCR result is empty (so frames with
    no on-screen text don't pollute the embedder).
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return pd.DataFrame()
    out = batch_df.copy()
    b64s = [b if isinstance(b, str) else "" for b in out.get("image_b64", [])]
    out["text"] = ocr_b64_to_text(
        b64s,
        model=model,
        invoke_url=invoke_url,
        api_key=api_key,
        nim_client=nim_client,
        merge_level=merge_level,
        batch_size=batch_size,
        timeout_s=timeout_s,
        retry=retry,
    )
    return out[out["text"].astype(bool)].reset_index(drop=True)


def _blocks_to_pseudo_markdown(
    blocks: List[Dict[str, Any]],
    crop_hw: Tuple[int, int] = (0, 0),
) -> str:
    """Convert OCR text blocks into pseudo-markdown table format.

    Uses DBSCAN clustering on pixel y-coordinates to identify rows, then
    sorts within each row by x-coordinate and joins with pipe separators.

    Parameters
    ----------
    blocks : list of dict
        OCR text blocks with ``sort_y`` (normalised [0,1]) and ``sort_x``.
    crop_hw : (height, width)
        Pixel dimensions of the crop image.  When provided the normalised
        ``sort_y`` values are scaled to pixels and clustered with
        ``eps=10`` (matching `nemo_retriever.api` behaviour).  Falls back to the old
        normalised-space heuristic when the height is unavailable.
    """
    if not blocks:
        return ""

    valid = [b for b in blocks if b.get("text")]
    if not valid:
        return ""

    df = pd.DataFrame(valid)
    df = df.sort_values("sort_y")

    y_vals = df["sort_y"].values
    crop_h = crop_hw[0] if crop_hw else 0

    if crop_h > 0:
        y_pixels = (y_vals * crop_h).astype(int)
        eps = 10
    else:
        y_range = y_vals.max() - y_vals.min()
        if y_range > 0:
            y_pixels = (y_vals - y_vals.min()) / y_range
            eps = 0.03
        else:
            y_pixels = y_vals
            eps = 0.1

    try:
        from sklearn.cluster import DBSCAN

        dbscan = DBSCAN(eps=eps, min_samples=1)
        dbscan.fit(y_pixels.reshape(-1, 1))
        df["cluster"] = dbscan.labels_
    except ImportError:
        # Naive fallback: round y to a coarse grid to simulate row grouping.
        df["cluster"] = (y_pixels / (eps if eps > 0 else 1)).round().astype(int)

    df = df.sort_values(["cluster", "sort_x"])

    rows = []
    for _, grp in df.groupby("cluster", sort=True):
        rows.append("| " + " | ".join(grp["text"].tolist()) + " |")
    return "\n".join(rows)


def _bboxes_close(a: Sequence[float], b: Sequence[float], tol: float = 1e-4) -> bool:
    """Check if two normalized bboxes are approximately equal."""
    if len(a) != 4 or len(b) != 4:
        return False
    return all(abs(float(a[i]) - float(b[i])) < tol for i in range(4))


def _find_ts_detections_for_bbox(
    row: Any,
    table_bbox: Sequence[float],
) -> Optional[Tuple[List[Dict[str, Any]], Optional[Tuple[int, int]]]]:
    """Find table-structure detections + crop size for a table bbox.

    Reads the ``table_structure_v1`` column from *row* and returns the
    ``(detections, (H, W))`` tuple for the region whose ``bbox_xyxy_norm``
    matches *table_bbox*. Returns ``None`` if the column is missing, no
    region matches, or the matching region has no detections.
    """
    ts_col = getattr(row, "table_structure_v1", None)
    if not isinstance(ts_col, dict):
        return None
    regions = ts_col.get("regions")
    if not isinstance(regions, list):
        return None

    for region in regions:
        if not isinstance(region, dict):
            continue
        region_bbox = region.get("bbox_xyxy_norm")
        if not isinstance(region_bbox, (list, tuple)) or len(region_bbox) != 4:
            continue
        if not _bboxes_close(table_bbox, region_bbox):
            continue
        dets = region.get("detections")
        if not isinstance(dets, list) or not dets:
            return None
        hw = region.get("orig_shape_hw")
        hw_t: Optional[Tuple[int, int]] = None
        if isinstance(hw, (list, tuple)) and len(hw) == 2:
            try:
                hw_t = (int(hw[0]), int(hw[1]))
            except (TypeError, ValueError):
                hw_t = None
        return (dets, hw_t)
    return None


@dataclass
class _OCRRowResult:
    """Mutable OCR output accumulated for one source page row."""

    table_items: List[Dict[str, Any]] = field(default_factory=list)
    chart_items: List[Dict[str, Any]] = field(default_factory=list)
    infographic_items: List[Dict[str, Any]] = field(default_factory=list)
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    error: Any = None


@dataclass(frozen=True)
class _PreparedOCRRow:
    """Validated source row and the labels that should be cropped from it."""

    row_index: int
    row: Any
    page_image_b64: str
    detections: List[Dict[str, Any]]
    wanted_labels: set[str]


@dataclass(frozen=True)
class _OCRCropJob:
    """One local crop plus the address needed to stitch its result."""

    row_index: int
    row: Any
    label_name: str
    bbox: List[float]
    crop_array: np.ndarray


@dataclass(frozen=True)
class _RemoteOCRCropJob:
    """One remote crop plus the address needed to stitch its result."""

    row_index: int
    row: Any
    label_name: str
    bbox: List[float]
    crop_b64: str


def _record_ocr_error(row_result: _OCRRowResult, exc: BaseException) -> None:
    print(f"Warning: OCR failed: {type(exc).__name__}: {exc}")
    row_result.error = {
        "stage": "ocr_page_elements",
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _prepare_ocr_rows(
    batch_df: pd.DataFrame,
    *,
    wanted_labels: set[str],
    extract_text: bool,
) -> Tuple[List[_PreparedOCRRow], List[_OCRRowResult]]:
    """Validate page inputs and retain one result slot per source row."""

    prepared_rows: List[_PreparedOCRRow] = []
    row_results = [_OCRRowResult() for _ in range(len(batch_df.index))]

    for row_index, row in enumerate(batch_df.itertuples(index=False)):
        row_result = row_results[row_index]
        try:
            page_elements = getattr(row, "page_elements_v3", None)
            detections: List[Dict[str, Any]] = []
            if isinstance(page_elements, dict):
                detections = page_elements.get("detections") or []
            if not isinstance(detections, list):
                detections = []

            page_image = getattr(row, "page_image", None) or {}
            page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
            if not isinstance(page_image_b64, str) or not page_image_b64:
                metadata = getattr(row, "metadata", None) or {}
                upstream_error = metadata.get("error") if isinstance(metadata, dict) else None
                page_num = getattr(row, "page_number", "?")
                path = getattr(row, "path", "?")
                if upstream_error:
                    _logger.warning(
                        "OCR skipping page %s of %s — no page image (upstream error: %s)",
                        page_num,
                        path,
                        upstream_error,
                    )
                else:
                    _logger.debug(
                        "OCR skipping page %s of %s — no page image (text-only or raster not requested)",
                        page_num,
                        path,
                    )
                row_result.error = upstream_error
                continue

            row_wanted_labels = wanted_labels
            if extract_text:
                metadata = getattr(row, "metadata", None) or {}
                needs_ocr = metadata.get("needs_ocr_for_text", False) if isinstance(metadata, dict) else False
                if needs_ocr:
                    row_wanted_labels = wanted_labels | _TEXT_LABELS

            prepared_rows.append(
                _PreparedOCRRow(
                    row_index=row_index,
                    row=row,
                    page_image_b64=page_image_b64,
                    detections=detections,
                    wanted_labels=row_wanted_labels,
                )
            )
        except BaseException as exc:
            _record_ocr_error(row_result, exc)

    return prepared_rows, row_results


def _append_ocr_prediction(
    row_result: _OCRRowResult,
    *,
    row: Any,
    label_name: str,
    bbox: List[float],
    preds: Any,
    crop_hw: Tuple[int, int],
    use_table_structure: bool,
) -> None:
    """Parse one prediction and stitch it into its source-row result."""

    blocks = _parse_ocr_result(preds)
    if label_name == "table":
        text = ""
        if use_table_structure:
            ts_match = _find_ts_detections_for_bbox(row, bbox)
            if ts_match is not None:
                ts_dets, ts_hw = ts_match
                text = join_table_structure_and_ocr_output(ts_dets, preds, ts_hw or crop_hw)
        if not text:
            text = _blocks_to_pseudo_markdown(blocks, crop_hw=crop_hw)
        if not text:
            text = _blocks_to_text(blocks)
    else:
        text = _blocks_to_text(blocks)

    entry = {"bbox_xyxy_norm": bbox, "text": text}
    if label_name == "table":
        row_result.table_items.append(entry)
    elif label_name == "chart":
        row_result.chart_items.append(entry)
    elif label_name == "infographic":
        row_result.infographic_items.append(entry)
    elif label_name in _TEXT_LABELS or label_name == "full_page":
        row_result.text_blocks.extend(blocks)


def _remote_crop_shape(crop_b64: str) -> Tuple[int, int]:
    """Return ``(height, width)`` for a remote crop when it can be decoded."""

    try:
        raw = base64.b64decode(crop_b64)
        with Image.open(io.BytesIO(raw)) as crop_image:
            width, height = crop_image.size
            return (height, width)
    except Exception:
        return (0, 0)


def _run_remote_ocr(
    prepared_rows: List[_PreparedOCRRow],
    row_results: List[_OCRRowResult],
    *,
    invoke_url: str,
    api_key: Optional[str],
    request_timeout_s: float,
    max_batch_size: int,
    retry: RemoteRetryParams,
    nim_client: NIMClient | None,
    use_table_structure: bool,
) -> None:
    """Batch remote crops across pages, retaining per-page failure isolation."""

    for prepared in prepared_rows:
        row_result = row_results[prepared.row_index]
        try:
            crops = _crop_all_from_page(
                prepared.page_image_b64,
                prepared.detections,
                prepared.wanted_labels,
                as_b64=True,
            )
            meta = getattr(prepared.row, "metadata", None) or {}
            needs_text = meta.get("needs_ocr_for_text", False) if isinstance(meta, dict) else False
            if needs_text and not any(label in _TEXT_LABELS for label, _bbox, _crop_b64 in crops):
                crops.append(("full_page", [0.0, 0.0, 1.0, 1.0], prepared.page_image_b64))
            crop_b64s: List[str] = [crop_b64 for _label, _bbox, crop_b64 in crops]
            crop_metadata: List[Tuple[str, List[float]]] = [(label_name, bbox) for label_name, bbox, _crop_b64 in crops]
            if not crop_b64s:
                continue
    if max_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    # Bound queued requests and retained encoded crops as well as active HTTP
    # workers. Crop preparation retains at most this window plus one page.
    window_size = max_batch_size * max(1, int(retry.remote_max_pool_workers))

    def _invoke(jobs: List[_RemoteOCRCropJob]) -> List[Any]:
        invoke_kwargs = dict(
            invoke_url=invoke_url,
            image_b64_list=[job.crop_b64 for job in jobs],
            api_key=api_key,
            timeout_s=float(request_timeout_s),
            max_batch_size=max_batch_size,
            max_retries=int(retry.remote_max_retries),
            max_429_retries=int(retry.remote_max_429_retries),
        )
        if nim_client is not None:
            response_items = nim_client.invoke_image_inference_batches(**invoke_kwargs)
        else:
            response_items = invoke_image_inference_batches(
                **invoke_kwargs,
                max_pool_workers=int(retry.remote_max_pool_workers),
            )
        return response_items

    def _stitch(jobs: List[_RemoteOCRCropJob], response_items: List[Any]) -> None:
        for job, response_item in zip(jobs, response_items):
            if row_results[job.row_index].error is not None:
                continue
            try:
                preds = _extract_remote_ocr_item(response_item)
                crop_hw = _remote_crop_shape(job.crop_b64) if job.label_name == "table" else (0, 0)
                _append_ocr_prediction(
                    row_results[job.row_index],
                    row=job.row,
                    label_name=job.label_name,
                    bbox=job.bbox,
                    preds=preds,
                    crop_hw=crop_hw,
                    use_table_structure=use_table_structure,
                )
            except Exception as exc:
                _fail_page(job.row_index, exc)

    def _fail_page(row_index: int, exc: Exception) -> None:
        # A page can span windows. A failed crop must not publish
        # successful fragments from earlier windows as partial page output.
        row_results[row_index] = _OCRRowResult()
        _record_ocr_error(row_results[row_index], exc)

    def _flush(jobs: List[_RemoteOCRCropJob]) -> None:
        jobs = [job for job in jobs if row_results[job.row_index].error is None]
        if not jobs:
            return
        try:
            response_items = _invoke(jobs)
        except Exception as exc:
            jobs_by_page: Dict[int, List[_RemoteOCRCropJob]] = {}
            for job in jobs:
                jobs_by_page.setdefault(job.row_index, []).append(job)
            # Only input/payload rejections can benefit from splitting a
            # multi-page call. Transport retries belong to the NIM client.
            errors = exc.exceptions if isinstance(exc, ExceptionGroup) else (exc,)
            isolate_pages = len(jobs_by_page) > 1 and all(
                isinstance(error, HTTPError)
                and error.response is not None
                and error.response.status_code in (400, 413, 422)
                for error in errors
            )
            for row_index, page_jobs in jobs_by_page.items():
                if not isolate_pages:
                    _fail_page(row_index, exc)
                    continue
                try:
                    _stitch(page_jobs, _invoke(page_jobs))
                except Exception as page_exc:
                    _fail_page(row_index, page_exc)
            return
        _stitch(jobs, response_items)

    pending: List[_RemoteOCRCropJob] = []
    for prepared in prepared_rows:
        try:
            crops = _crop_all_from_page(
                prepared.page_image_b64,
                prepared.detections,
                prepared.wanted_labels,
                as_b64=True,
            )
            for label_name, bbox, crop_b64 in crops:
                if row_results[prepared.row_index].error is not None:
                    break
                pending.append(_RemoteOCRCropJob(prepared.row_index, prepared.row, label_name, bbox, crop_b64))
                if len(pending) == window_size:
                    _flush(pending)
                    pending = []
        except Exception as exc:
            _record_ocr_error(row_results[prepared.row_index], exc)
    _flush(pending)


def _collect_local_crop_jobs(
    prepared_rows: List[_PreparedOCRRow],
    row_results: List[_OCRRowResult],
) -> Dict[str, List[_OCRCropJob]]:
    """Collect compatible local crops across all prepared page rows."""

    jobs_by_merge_level: Dict[str, List[_OCRCropJob]] = {"word": [], "paragraph": []}
    for prepared in prepared_rows:
        try:
            crops = _crop_all_from_page(
                prepared.page_image_b64,
                prepared.detections,
                prepared.wanted_labels,
            )
            meta = getattr(prepared.row, "metadata", None) or {}
            needs_text = meta.get("needs_ocr_for_text", False) if isinstance(meta, dict) else False
            if needs_text and not any(label in _TEXT_LABELS for label, _bbox, _crop_arr in crops):
                try:
                    raw = base64.b64decode(prepared.page_image_b64)
                    with Image.open(io.BytesIO(raw)) as im0:
                        full_crop = np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
                    crops.append(("full_page", [0.0, 0.0, 1.0, 1.0], full_crop))
                except Exception:
                    if not crops:
                        crops = []
            for label_name, bbox, crop_array in crops:
                merge_level = "word" if label_name == "table" else "paragraph"
                jobs_by_merge_level[merge_level].append(
                    _OCRCropJob(
                        row_index=prepared.row_index,
                        row=prepared.row,
                        label_name=label_name,
                        bbox=bbox,
                        crop_array=crop_array,
                    )
                )
        except BaseException as exc:
            _record_ocr_error(row_results[prepared.row_index], exc)
    return jobs_by_merge_level


def _run_local_ocr_batches(
    model: Any,
    jobs_by_merge_level: Dict[str, List[_OCRCropJob]],
    row_results: List[_OCRRowResult],
    *,
    batch_size: int,
    use_table_structure: bool,
) -> None:
    """Run bounded local crop lists, falling back per crop when required."""

    for merge_level, jobs in jobs_by_merge_level.items():
        for start in range(0, len(jobs), batch_size):
            batch_jobs = jobs[start : start + batch_size]
            batch_crops = [job.crop_array for job in batch_jobs]

            try:
                batch_preds = model.invoke(batch_crops, merge_level=merge_level)
            except Exception:
                batch_preds = None

            if not isinstance(batch_preds, list) or len(batch_preds) != len(batch_jobs):
                for job in batch_jobs:
                    try:
                        preds = model.invoke(job.crop_array, merge_level=merge_level)
                        _append_ocr_prediction(
                            row_results[job.row_index],
                            row=job.row,
                            label_name=job.label_name,
                            bbox=job.bbox,
                            preds=preds,
                            crop_hw=(job.crop_array.shape[0], job.crop_array.shape[1]),
                            use_table_structure=use_table_structure,
                        )
                    except BaseException as exc:
                        _record_ocr_error(row_results[job.row_index], exc)
                continue

            for job, preds in zip(batch_jobs, batch_preds):
                try:
                    _append_ocr_prediction(
                        row_results[job.row_index],
                        row=job.row,
                        label_name=job.label_name,
                        bbox=job.bbox,
                        preds=preds,
                        crop_hw=(job.crop_array.shape[0], job.crop_array.shape[1]),
                        use_table_structure=use_table_structure,
                    )
                except BaseException as exc:
                    _record_ocr_error(row_results[job.row_index], exc)


def _build_ocr_page_elements_output(
    batch_df: pd.DataFrame,
    row_results: List[_OCRRowResult],
    *,
    extract_text: bool,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
    elapsed_s: float,
) -> pd.DataFrame:
    """Finalize per-row metadata and add OCR output columns to the batch."""

    all_table = [result.table_items for result in row_results]
    all_chart = [result.chart_items for result in row_results]
    all_infographic = [result.infographic_items for result in row_results]
    all_text: List[Optional[str]] = []
    all_ocr_meta: List[Dict[str, Any]] = []

    for result in row_results:
        all_text.append(_blocks_to_text(result.text_blocks) if extract_text and result.text_blocks else None)

        counts_by_label: Dict[str, int] = {}
        if result.table_items:
            counts_by_label["table"] = len(result.table_items)
        if result.chart_items:
            counts_by_label["chart"] = len(result.chart_items)
        if result.infographic_items:
            counts_by_label["infographic"] = len(result.infographic_items)
        if result.text_blocks:
            counts_by_label["text"] = len(result.text_blocks)

        all_ocr_meta.append(
            {
                "timing": {"seconds": float(elapsed_s)},
                "error": result.error,
                "num_detections": sum(counts_by_label.values()),
                "counts_by_label": counts_by_label,
            }
        )

    out = batch_df.copy()
    if extract_tables or "table" not in out.columns:
        out["table"] = all_table
    if extract_charts or "chart" not in out.columns:
        out["chart"] = all_chart
    if extract_infographics or "infographic" not in out.columns:
        out["infographic"] = all_infographic
    if extract_text and "text" in out.columns:
        for index, ocr_text in enumerate(all_text):
            if ocr_text is not None:
                out.iat[index, out.columns.get_loc("text")] = ocr_text
    elif extract_text:
        out["text"] = [text if text is not None else "" for text in all_text]
    out["ocr"] = all_ocr_meta
    out["ocr_v1_num_detections"] = [metadata["num_detections"] for metadata in all_ocr_meta]
    out["ocr_v1_counts_by_label"] = [metadata["counts_by_label"] for metadata in all_ocr_meta]
    return out


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def ocr_page_elements(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    use_table_structure: bool = False,
    inference_batch_size: int = 8,
    remote_retry: RemoteRetryParams | None = None,
    nim_client: NIMClient | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run Nemotron OCR on cropped regions detected by PageElements v3.

    For each row (page) in ``batch_df``:
    1. Read ``page_elements_v3`` detections and ``page_image["image_b64"]``.
    2. For each detection whose ``label_name`` is a requested type, crop the
       page image, invoke OCR, parse the result, and collect text.
    3. Write per-type content lists and timing metadata to output columns.

    Parameters
    ----------
    batch_df : pandas.DataFrame
        Ray Data batch with ``page_elements_v3`` and ``page_image`` columns.
    model
        Initialised OCR model.
    extract_tables, extract_charts, extract_infographics : bool
        Which element types to OCR.

    Returns
    -------
    pandas.DataFrame
        Original columns plus ``table``, ``chart``,
        ``infographic``, and ``ocr``.
    """
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("ocr_page_elements currently only supports pandas.DataFrame input.")

    invoke_url = str(invoke_url or "").strip() or str(kwargs.get("ocr_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    # Determine which labels we need to process.
    # Text/title labels are added per-row based on needs_ocr_for_text metadata.
    wanted_labels: set[str] = set()
    if extract_tables:
        wanted_labels.add("table")
    if extract_charts:
        wanted_labels.add("chart")
    if extract_infographics:
        wanted_labels.add("infographic")

    # This bounds the outer crop list passed to the persistent model wrapper.
    # Nemotron's internal detector_max_batch_size is a separate control.
    local_invoke_batch_size = 0
    if not use_remote:
        if inference_batch_size is None or inference_batch_size < 1:
            raise ValueError(f"inference_batch_size must be set and greater than 0. Value: {inference_batch_size}")
        local_invoke_batch_size = int(inference_batch_size)

    t0_total = time.perf_counter()
    prepared_rows, row_results = _prepare_ocr_rows(
        batch_df,
        wanted_labels=wanted_labels,
        extract_text=extract_text,
    )

    if use_remote:
        _run_remote_ocr(
            prepared_rows,
            row_results,
            invoke_url=invoke_url,
            api_key=api_key,
            request_timeout_s=request_timeout_s,
            max_batch_size=int(inference_batch_size),
            retry=retry,
            nim_client=nim_client,
            use_table_structure=use_table_structure,
        )
    else:
        jobs_by_merge_level = _collect_local_crop_jobs(prepared_rows, row_results)
        _run_local_ocr_batches(
            model,
            jobs_by_merge_level,
            row_results,
            batch_size=local_invoke_batch_size,
            use_table_structure=use_table_structure,
        )

    return _build_ocr_page_elements_output(
        batch_df,
        row_results,
        extract_text=extract_text,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extract_infographics=extract_infographics,
        elapsed_s=time.perf_counter() - t0_total,
    )


# ---------------------------------------------------------------------------
# Nemotron Parse v1.2
# ---------------------------------------------------------------------------


def _extract_parse_text(response_item: Any) -> str:
    if response_item is None:
        return ""
    if isinstance(response_item, str):
        return response_item.strip()
    if isinstance(response_item, dict):
        for key in ("generated_text", "text", "output_text", "prediction", "output", "data"):
            value = response_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, dict):
                    inner = _extract_parse_text(first)
                    if inner:
                        return inner
    if isinstance(response_item, list):
        for item in response_item:
            text = _extract_parse_text(item)
            if text:
                return text
    try:
        return str(response_item).strip()
    except Exception:
        return ""


def nemotron_parse_page_elements(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
    remote_retry: RemoteRetryParams | None = None,
    nim_client: NIMClient | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run Nemotron Parse v1.2 on cropped page elements.

    Emits OCR-compatible content columns (``table``, ``chart``, ``infographic``)
    so this stage can replace the page-elements + OCR pair in pipeline wiring.
    """
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("nemotron_parse_page_elements currently only supports pandas.DataFrame input.")

    invoke_url = str(invoke_url or "").strip() or str(kwargs.get("nemotron_parse_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    wanted_labels: set[str] = set()
    if extract_tables:
        wanted_labels.add("table")
    if extract_charts:
        wanted_labels.add("chart")
    if extract_infographics:
        wanted_labels.add("infographic")

    all_table: List[List[Dict[str, Any]]] = []
    all_chart: List[List[Dict[str, Any]]] = []
    all_infographic: List[List[Dict[str, Any]]] = []
    all_text: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
        chart_items: List[Dict[str, Any]] = []
        infographic_items: List[Dict[str, Any]] = []
        row_text: Optional[str] = None
        row_error: Any = None

        try:
            pe = getattr(row, "page_elements_v3", None)
            dets: List[Dict[str, Any]] = []
            if isinstance(pe, dict):
                dets = pe.get("detections") or []
            if not isinstance(dets, list):
                dets = []

            page_image = getattr(row, "page_image", None) or {}
            page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
            if not isinstance(page_image_b64, str) or not page_image_b64:
                all_table.append(table_items)
                all_chart.append(chart_items)
                all_infographic.append(infographic_items)
                all_text.append(None)
                all_meta.append({"timing": None, "error": None})
                continue

            if use_remote:
                crops = _crop_all_from_page(page_image_b64, dets, wanted_labels, as_b64=True)
                # Parse-only mode may skip page-elements detection entirely. In that
                # case, parse the full page once and fan out the text to enabled
                # content channels.  The image is already base64 — pass it through.
                if not crops and wanted_labels:
                    crops = [("full_page", [0.0, 0.0, 1.0, 1.0], page_image_b64)]

                crop_b64s: List[str] = [b64 for _label, _bbox, b64 in crops]
                crop_meta: List[Tuple[str, List[float]]] = [(label, bbox) for label, bbox, _b64 in crops]

                if crop_b64s:
                    _invoke_kw = dict(
                        invoke_url=invoke_url,
                        image_b64_list=crop_b64s,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        max_batch_size=int(kwargs.get("inference_batch_size", 8)),
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                    if nim_client is not None:
                        response_items = nim_client.invoke_image_inference_batches(**_invoke_kw)
                    else:
                        response_items = invoke_image_inference_batches(
                            **_invoke_kw,
                            max_pool_workers=int(retry.remote_max_pool_workers),
                        )
                    if len(response_items) != len(crop_meta):
                        raise RuntimeError(f"Expected {len(crop_meta)} Parse responses, got {len(response_items)}")

                    for i, (label_name, bbox) in enumerate(crop_meta):
                        text = _extract_parse_text(response_items[i])
                        entry = {"bbox_xyxy_norm": bbox, "text": text}
                        if label_name == "table":
                            table_items.append(entry)
                        elif label_name == "chart":
                            chart_items.append(entry)
                        elif label_name == "infographic":
                            infographic_items.append(entry)
                        elif label_name == "full_page":
                            if extract_tables:
                                table_items.append(dict(entry))
                            if extract_charts:
                                chart_items.append(dict(entry))
                            if extract_infographics:
                                infographic_items.append(dict(entry))
            else:
                crops = _crop_all_from_page(page_image_b64, dets, wanted_labels)
                if not crops and wanted_labels:
                    try:
                        raw = base64.b64decode(page_image_b64)
                        with Image.open(io.BytesIO(raw)) as im0:
                            full_crop = np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
                        crops = [("full_page", [0.0, 0.0, 1.0, 1.0], full_crop)]
                    except Exception:
                        crops = []
                for label_name, bbox, crop_array in crops:
                    text = str(model.invoke(crop_array, task_prompt=task_prompt) or "").strip()
                    entry = {"bbox_xyxy_norm": bbox, "text": text}
                    if label_name == "table":
                        table_items.append(entry)
                    elif label_name == "chart":
                        chart_items.append(entry)
                    elif label_name == "infographic":
                        infographic_items.append(entry)
                    elif label_name == "full_page":
                        if extract_tables:
                            table_items.append(dict(entry))
                        if extract_charts:
                            chart_items.append(dict(entry))
                        if extract_infographics:
                            infographic_items.append(dict(entry))

            # When extract_text is requested, parse the full page for text
            # (only for pages that need OCR-based text extraction).
            meta = getattr(row, "metadata", None) or {}
            needs_ocr = meta.get("needs_ocr_for_text", False) if isinstance(meta, dict) else False
            if extract_text and needs_ocr:
                try:
                    if use_remote:
                        _text_kw = dict(
                            invoke_url=invoke_url,
                            image_b64_list=[page_image_b64],
                            api_key=api_key,
                            timeout_s=float(request_timeout_s),
                            max_batch_size=1,
                            max_retries=int(retry.remote_max_retries),
                            max_429_retries=int(retry.remote_max_429_retries),
                        )
                        if nim_client is not None:
                            resp = nim_client.invoke_image_inference_batches(**_text_kw)
                        else:
                            resp = invoke_image_inference_batches(
                                **_text_kw,
                                max_pool_workers=int(retry.remote_max_pool_workers),
                            )
                        row_text = _extract_parse_text(resp[0]) if resp else ""
                    else:
                        raw = base64.b64decode(page_image_b64)
                        with Image.open(io.BytesIO(raw)) as im0:
                            full_crop = np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
                        row_text = str(model.invoke(full_crop, task_prompt=task_prompt) or "").strip()
                except Exception:
                    row_text = ""

        except BaseException as e:
            print(f"Warning: Nemotron Parse failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "nemotron_parse_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_text.append(row_text)
        all_table.append(table_items)
        all_chart.append(chart_items)
        all_infographic.append(infographic_items)
        all_meta.append({"timing": None, "error": row_error})

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    if extract_text and "text" in out.columns:
        # Only overwrite rows where parse produced text; preserve native text otherwise.
        for i, parse_text in enumerate(all_text):
            if parse_text is not None:
                out.iat[i, out.columns.get_loc("text")] = parse_text
    elif extract_text:
        out["text"] = [t if t is not None else "" for t in all_text]
    out["table"] = all_table
    out["chart"] = all_chart
    out["infographic"] = all_infographic
    # Aliases retained for experiments that read parse-specific columns.
    out["table_parse"] = all_table
    out["chart_parse"] = all_chart
    out["infographic_parse"] = all_infographic
    out["nemotron_parse_v1_2"] = all_meta
    return out
