# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import hashlib
from collections import defaultdict
from typing import Any, Dict, Optional, List, Tuple, Set

import pandas as pd

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema

logger = logging.getLogger(__name__)


def calculate_iou(bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Boxes are in format (x1, y1, x2, y2) where (x1, y1) is the top-left corner
    and (x2, y2) is the bottom-right corner.

    Parameters
    ----------
    bbox1 : tuple
        First bounding box as (x1, y1, x2, y2).
    bbox2 : tuple
        Second bounding box as (x1, y1, x2, y2).

    Returns
    -------
    float
        IoU value between 0.0 and 1.0.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
    x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Check for no intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def _normalize_bbox(bbox: Tuple[float, ...], max_dimensions: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box coordinates by max dimensions.

    Parameters
    ----------
    bbox : tuple
        Bounding box as (x1, y1, x2, y2).
    max_dimensions : tuple
        Max dimensions as (max_width, max_height).

    Returns
    -------
    tuple
        Normalized bounding box as (x1, y1, x2, y2) with values in [0, 1].
    """
    x1, y1, x2, y2 = bbox[:4]
    max_width, max_height = max_dimensions

    if max_width <= 0 or max_height <= 0:
        # Cannot normalize, return original bbox
        return (x1, y1, x2, y2)

    return (x1 / max_width, y1 / max_height, x2 / max_width, y2 / max_height)


def _get_image_bbox_info(row: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Extract normalized bounding box info from an IMAGE row.

    Parameters
    ----------
    row : pd.Series
        DataFrame row with metadata.

    Returns
    -------
    dict or None
        Dictionary with 'page', 'bbox' (normalized), and 'index', or None if bbox not available.
    """
    try:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            return None

        content_metadata = metadata.get("content_metadata", {})
        image_metadata = metadata.get("image_metadata", {})

        page = content_metadata.get("page_number", -1)
        bbox = image_metadata.get("image_location")
        max_dims = image_metadata.get("image_location_max_dimensions", (0, 0))

        if bbox is None or not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
            return None

        # Normalize bbox by max dimensions
        if max_dims and len(max_dims) >= 2 and max_dims[0] > 0 and max_dims[1] > 0:
            normalized_bbox = _normalize_bbox(tuple(bbox[:4]), tuple(max_dims[:2]))
        else:
            normalized_bbox = tuple(bbox[:4])

        return {"page": page, "bbox": normalized_bbox, "index": row.name}
    except Exception:
        return None


def _get_structured_bbox_info(row: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Extract normalized bounding box info from a STRUCTURED row (table/chart/infographic).

    Parameters
    ----------
    row : pd.Series
        DataFrame row with metadata.

    Returns
    -------
    dict or None
        Dictionary with 'page', 'bbox' (normalized), 'index', and 'subtype', or None if bbox not available.
    """
    try:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            return None

        content_metadata = metadata.get("content_metadata", {})
        table_metadata = metadata.get("table_metadata", {})

        page = content_metadata.get("page_number", -1)
        subtype = content_metadata.get("subtype", "")
        bbox = table_metadata.get("table_location")
        max_dims = table_metadata.get("table_location_max_dimensions", (0, 0))

        if bbox is None or not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
            return None

        # Normalize bbox by max dimensions
        if max_dims and len(max_dims) >= 2 and max_dims[0] > 0 and max_dims[1] > 0:
            normalized_bbox = _normalize_bbox(tuple(bbox[:4]), tuple(max_dims[:2]))
        else:
            normalized_bbox = tuple(bbox[:4])

        return {"page": page, "bbox": normalized_bbox, "index": row.name, "subtype": subtype}
    except Exception:
        return None


def deduplicate_by_bbox_internal(
    df_ledger: pd.DataFrame,
    iou_threshold: float = 0.45,
    prefer_structured: bool = True,
) -> pd.DataFrame:
    """
    Remove duplicate visual elements based on bounding box overlap.

    When an IMAGE element's bounding box substantially overlaps with a STRUCTURED
    element (table/chart/infographic) on the same page, one is removed based on
    the prefer_structured flag.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame with document_type, metadata columns.
    iou_threshold : float
        Minimum IoU to consider elements as duplicates (default 0.4).
    prefer_structured : bool
        If True, keep structured elements and drop images when duplicates found.
        If False, keep images and drop structured elements.

    Returns
    -------
    pd.DataFrame
        DataFrame with bbox-based duplicates removed.
    """
    # Identify rows by type
    image_mask = df_ledger["document_type"] == ContentTypeEnum.IMAGE
    structured_mask = df_ledger["document_type"] == ContentTypeEnum.STRUCTURED

    if not image_mask.any() or not structured_mask.any():
        return df_ledger  # Nothing to deduplicate

    # Extract bounding box info for each type
    image_infos = []
    for idx in df_ledger[image_mask].index:
        info = _get_image_bbox_info(df_ledger.loc[idx])
        if info is not None:
            image_infos.append(info)

    structured_infos = []
    for idx in df_ledger[structured_mask].index:
        info = _get_structured_bbox_info(df_ledger.loc[idx])
        if info is not None:
            structured_infos.append(info)

    if not image_infos or not structured_infos:
        return df_ledger

    # Group by page for efficient comparison
    images_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for info in image_infos:
        images_by_page[info["page"]].append(info)

    structured_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for info in structured_infos:
        structured_by_page[info["page"]].append(info)

    # Find duplicates by comparing bounding boxes on the same page
    indices_to_drop: Set[Any] = set()

    for page in images_by_page:
        if page not in structured_by_page:
            continue

        for image_info in images_by_page[page]:
            if image_info["index"] in indices_to_drop:
                continue  # Already marked for removal

            for struct_info in structured_by_page[page]:
                if struct_info["index"] in indices_to_drop:
                    continue  # Already marked for removal

                iou = calculate_iou(image_info["bbox"], struct_info["bbox"])

                if iou >= iou_threshold:
                    # Found a duplicate pair
                    if prefer_structured:
                        indices_to_drop.add(image_info["index"])
                    else:
                        indices_to_drop.add(struct_info["index"])
                    break  # One match is enough to mark as duplicate

    if not indices_to_drop:
        return df_ledger

    logger.info(f"Bbox dedup: Removed {len(indices_to_drop)} duplicate elements")
    result = df_ledger.drop(index=list(indices_to_drop))

    return result


def _hash_content(x: Any, algorithm: str = "md5") -> bytes:
    """
    Compute a hash of the content using the specified algorithm.

    Parameters
    ----------
    x : dict
        A dictionary containing the content under the key "content".
    algorithm : str, optional
        Hashing algorithm to use (default "md5").

    Returns
    -------
    bytes
        The computed hash.
    """
    try:
        return hashlib.new(algorithm, x["content"].encode()).digest()
    except Exception as e:
        msg = f"hash_content: Error computing hash: {e}"
        logger.error(msg, exc_info=True)
        raise type(e)(msg) from e


def deduplicate_images_internal(
    df_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    mutate_config: ImageDedupSchema = ImageDedupSchema(),
    execution_trace_log: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    Deduplicate images in a DataFrame based on content hashes and/or bounding box overlap.

    The function processes rows where the 'document_type' is IMAGE, computes a content hash for each,
    and then either removes duplicates or marks them based on the 'filter' flag in task_config.
    A 'hash_algorithm' flag in task_config determines the algorithm used for hashing.

    Additionally, if 'enable_bbox_dedup' is True, removes images that substantially overlap
    with structured elements (tables/charts) based on IoU threshold.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing at least 'document_type' and 'metadata' columns.
    task_config : dict
        Configuration parameters, including:
            - "filter": bool, if True duplicate rows are removed; if False, duplicates are marked.
            - "hash_algorithm": str, the algorithm to use for hashing (default "md5").
            - "enable_bbox_dedup": bool, if True also deduplicate by bounding box overlap.
            - "iou_threshold": float, IoU threshold for bbox dedup (default 0.45).
            - "bbox_dedup_prefer_structured": bool, if True keep structured elements (default True).
    mutate_config : ImageDedupSchema, optional
    execution_trace_log : Optional[List[Any]], optional

    Returns
    -------
    pd.DataFrame
        The DataFrame with duplicate images either removed or marked.

    Raises
    ------
    ValueError
        If the required columns are missing.
    Exception
        For any other errors encountered during deduplication.
    """

    _ = mutate_config  # Unused variable
    _ = execution_trace_log  # TODO(Devin): Implement trace logging

    try:
        # Verify required columns exist.
        for col in ("document_type", "metadata"):
            if col not in df_ledger.columns:
                raise ValueError(f"Missing required column '{col}'.")

        result = df_ledger

        # Parameters are nested inside "params" key from task_properties
        params = task_config.get("params", {})

        # Content-hash based deduplication for images
        image_mask = result["document_type"] == ContentTypeEnum.IMAGE
        if image_mask.any():
            df_images = result.loc[image_mask].copy()
            hash_algorithm = params.get("hash_algorithm", "md5")

            # Compute content hash for each image.
            df_images["_image_content_hash"] = df_images["metadata"].apply(_hash_content, args=(hash_algorithm,))
            df_images_deduped = df_images.drop_duplicates(subset="_image_content_hash")
            deduped_indices = df_images_deduped.index

            non_image_rows = result.loc[~image_mask]
            deduped_images = df_images.loc[deduped_indices][result.columns.difference(["_image_content_hash"])]

            result = pd.concat([deduped_images, non_image_rows], axis=0)

        # Bounding box based deduplication (enabled by default)
        enable_bbox_dedup = params.get("enable_bbox_dedup", True)

        if enable_bbox_dedup:
            iou_threshold = params.get("iou_threshold", 0.45)

            prefer_structured = params.get("bbox_dedup_prefer_structured", True)

            result = deduplicate_by_bbox_internal(
                df_ledger=result,
                iou_threshold=iou_threshold,
                prefer_structured=prefer_structured,
            )

        return result
    except Exception as e:
        msg = f"deduplicate_images_internal: Error applying deduplication filter: {e}"
        logger.error(msg, exc_info=True)
        raise type(e)(msg) from e
