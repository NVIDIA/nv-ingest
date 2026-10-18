# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-friendly content row transforms used by example pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from nemo_retriever.common.io.image_store import inline_image_b64
from nemo_retriever.operators.extract.ocr.ocr import _crop_b64_image_by_norm_bbox
from nemo_retriever.common.params.models import IMAGE_MODALITIES
from nemo_retriever.common.stage_errors import iter_stage_errors_from_value

_CONTENT_COLUMNS = ("table", "chart", "infographic")
_EXTRACTION_PAYLOAD_COLUMNS = (
    "bytes",
    "images",
    "tables",
    "charts",
    "infographics",
    "page_elements_v3",
    "table_structure_v1",
    "table_structure_ocr_v1",
    "ocr",
    "table_parse",
    "chart_parse",
    "infographic_parse",
    "nemotron_parse_v1_2",
) + _CONTENT_COLUMNS


def _compact_embedding_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove resolved extraction payloads, retaining their counts and errors."""
    dropped = [
        column
        for column in _EXTRACTION_PAYLOAD_COLUMNS
        if column in frame and not any(any(iter_stage_errors_from_value(value)) for value in frame[column])
    ]
    result = frame.drop(columns=dropped)

    if any(kind in frame for kind in _CONTENT_COLUMNS):
        metadata = []
        for _, row in frame.iterrows():
            row_metadata = dict(row["metadata"]) if isinstance(row.get("metadata"), dict) else {}
            content_metadata = row_metadata.get("content_metadata")
            content_metadata = dict(content_metadata) if isinstance(content_metadata, dict) else {}
            for kind in _CONTENT_COLUMNS:
                detections = row.get(kind)
                if isinstance(detections, list):
                    content_metadata.setdefault(f"ocr_{kind}_detections", len(detections))
            row_metadata["content_metadata"] = content_metadata
            metadata.append(row_metadata)
        result["metadata"] = pd.Series(metadata, index=frame.index, dtype=object)

    if "page_image" in frame and not any(any(iter_stage_errors_from_value(v)) for v in frame["page_image"]):
        # Keep page URI provenance so VDB conversion can distinguish uploaded crops.
        result["page_image"] = pd.Series(
            [
                {"stored_image_uri": v.get("stored_image_uri")} if isinstance(v, dict) else None
                for v in frame["page_image"]
            ],
            index=frame.index,
            dtype=object,
        )
    return result


def _is_content_collection(value: Any) -> bool:
    """Return whether a value is a supported extracted-content collection."""
    return isinstance(value, list) or (isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == object)


def _combine_text_with_content(row: Any, text_column: str, content_columns: Sequence[str]) -> str:
    """Combine page text with OCR content text for embedding."""
    parts = []
    base = row.get(text_column)
    if isinstance(base, str) and base.strip():
        parts.append(base.strip())
    for col in content_columns:
        content_list = row.get(col)
        if _is_content_collection(content_list):
            for item in content_list:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                    caption = item.get("caption", "")
                    if isinstance(caption, str) and caption.strip():
                        parts.append(caption.strip())
    return "\n\n".join(parts) if parts else ""


def _deep_copy_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Copy row dicts without sharing nested mutable values across exploded rows."""
    import copy

    out: Dict[str, Any] = {}
    for key, value in row_dict.items():
        if isinstance(value, np.ndarray) and _is_content_collection(value):
            out[key] = copy.deepcopy(value.tolist())
        elif isinstance(value, (dict, list)):
            out[key] = copy.deepcopy(value)
        else:
            out[key] = value
    return out


def _normalize_bbox(value: Any) -> Any:
    """Return bbox coordinates in a stable non-tensor container."""
    if hasattr(value, "to_numpy") and callable(value.to_numpy):
        value = value.to_numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _normalize_bbox_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Force every batch to expose bbox values as ordinary Python objects."""
    values = frame.get("_bbox_xyxy_norm", pd.Series([None] * len(frame.index), index=frame.index))
    frame["_bbox_xyxy_norm"] = pd.Series(
        [_normalize_bbox(value) for value in values],
        index=frame.index,
        dtype=object,
    )
    return frame


def explode_content_to_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
    text_elements_modality: Optional[str] = None,
    structured_elements_modality: Optional[str] = None,
    compact: bool = False,
) -> Any:
    """Expand each page row into multiple rows for per-element embedding."""
    text_mod = text_elements_modality or modality
    struct_mod = structured_elements_modality or modality

    if not isinstance(batch_df, pd.DataFrame):
        return batch_df
    if batch_df.empty:
        result = _normalize_bbox_column(batch_df)
        return _compact_embedding_rows(result) if compact else result

    any_images = text_mod in IMAGE_MODALITIES or struct_mod in IMAGE_MODALITIES

    if not any(column in batch_df.columns for column in content_columns):
        batch_df = batch_df.copy()
        if text_mod in IMAGE_MODALITIES and "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: inline_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        if "page_image" in batch_df.columns:
            batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
                lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
            )
        batch_df["_embed_modality"] = text_mod
        result = _normalize_bbox_column(batch_df)
        return _compact_embedding_rows(result) if compact else result

    new_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        row_dict = row.to_dict()
        exploded_any = False

        page_image = row_dict.get("page_image")
        page_image_b64: Optional[str] = None
        page_stored_uri: Optional[str] = None
        if isinstance(page_image, dict):
            page_stored_uri = page_image.get("stored_image_uri")
            if any_images:
                page_image_b64 = inline_image_b64(page_image)

        page_text = row_dict.get(text_column)
        if isinstance(page_text, str) and page_text.strip():
            page_row = _deep_copy_row(row_dict)
            page_row["_embed_modality"] = text_mod
            page_row["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                page_row["_image_b64"] = page_image_b64
            page_row["_stored_image_uri"] = page_stored_uri
            page_row["_bbox_xyxy_norm"] = None
            new_rows.append(page_row)
            exploded_any = True

        for column in content_columns:
            content_list = row_dict.get(column)
            if not _is_content_collection(content_list):
                continue
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                item_b64 = inline_image_b64(item) if struct_mod in IMAGE_MODALITIES else None
                # Emit rows for text and (optionally) caption fields.
                for field, content_type in [("text", column), ("caption", f"{column}_caption")]:
                    value = item.get(field, "")
                    if not isinstance(value, str) or not value.strip():
                        continue
                    content_row = _deep_copy_row(row_dict)
                    content_row[text_column] = value.strip()
                    content_row["_embed_modality"] = struct_mod
                    content_row["_content_type"] = content_type
                    if struct_mod in IMAGE_MODALITIES:
                        if item_b64:
                            content_row["_image_b64"] = item_b64
                        elif page_image_b64:
                            bbox = _normalize_bbox(item.get("bbox_xyxy_norm"))
                            if bbox is not None and len(bbox) == 4:
                                cropped_b64, _ = _crop_b64_image_by_norm_bbox(page_image_b64, bbox_xyxy_norm=bbox)
                                content_row["_image_b64"] = cropped_b64
                            else:
                                content_row["_image_b64"] = page_image_b64
                        else:
                            content_row["_image_b64"] = None
                    content_row["_stored_image_uri"] = item.get("stored_image_uri") or page_stored_uri
                    content_row["_bbox_xyxy_norm"] = _normalize_bbox(item.get("bbox_xyxy_norm"))
                    new_rows.append(content_row)
                    exploded_any = True

        if not exploded_any:
            preserved = _deep_copy_row(row_dict)
            preserved["_embed_modality"] = text_mod
            preserved["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                preserved["_image_b64"] = page_image_b64
            preserved["_stored_image_uri"] = page_stored_uri
            preserved["_bbox_xyxy_norm"] = None
            new_rows.append(preserved)

    result = _normalize_bbox_column(pd.DataFrame(new_rows).reset_index(drop=True))
    return _compact_embedding_rows(result) if compact else result


def collapse_content_to_page_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
    compact: bool = False,
) -> Any:
    """Collapse each page into a single row for page-level embedding."""
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    has_page_image = "page_image" in batch_df.columns
    new_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        row_dict = _deep_copy_row(row.to_dict())
        row_dict[text_column] = _combine_text_with_content(row_dict, text_column, content_columns)
        page_image = row_dict.get("page_image") if has_page_image else None
        if modality in IMAGE_MODALITIES:
            row_dict["_image_b64"] = inline_image_b64(page_image) if isinstance(page_image, dict) else None
        if has_page_image:
            row_dict["_stored_image_uri"] = page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
        row_dict["_embed_modality"] = modality
        new_rows.append(row_dict)

    result = _normalize_bbox_column(pd.DataFrame(new_rows).reset_index(drop=True))
    return _compact_embedding_rows(result) if compact else result
