# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for saving images from ingestion results to disk as actual image files.

This module provides comprehensive utilities for extracting and saving base64-encoded
images from nv-ingest results to local filesystem. Features include:
- Configurable filtering by image type (charts, tables, infographics, etc.)
- Descriptive filename generation with source and page information
- Organized directory structure by image type
- Detailed image counting and statistics

Typical use cases:
- Debugging and visual inspection of extracted content
- Quality assessment of image extraction pipeline
"""

import base64
import logging
import os
from typing import Any, Dict, List

import cv2
import numpy as np

from nv_ingest_client.client.util.processing import get_valid_filename

logger = logging.getLogger(__name__)


def _is_base64_image_format(base64_string: str, target_format: str) -> bool:
    """
    Checks if base64 image is already in the target format using simple magic bytes.

    Parameters
    ----------
    base64_string : str
        Base64-encoded image string.
    target_format : str
        Target format to check ("PNG", "JPEG").

    Returns
    -------
    bool
        True if image is already in target format, False otherwise.
    """
    try:
        image_bytes = base64.b64decode(base64_string)

        # Simple magic byte checks for common formats
        if target_format.upper() == "JPEG" and image_bytes.startswith(b"\xff\xd8\xff"):
            return True
        elif target_format.upper() == "PNG" and image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return True

        return False
    except Exception:
        return False


def _base64_to_disk_direct(base64_string: str, output_path: str) -> bool:
    """
    Write base64 image data directly to disk without conversion.

    Parameters
    ----------
    base64_string : str
        Base64-encoded image data.
    output_path : str
        Path where the image should be saved.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Strip data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return True
    except Exception as e:
        logger.error(f"Failed to write image directly to disk: {e}")
        return False


def _base64_to_disk_with_conversion(
    base64_string: str, output_path: str, target_format: str, quality: int = 95
) -> bool:
    """
    Convert base64 image to target format and save to disk using OpenCV.

    Parameters
    ----------
    base64_string : str
        Base64-encoded image data.
    output_path : str
        Path where the converted image should be saved.
    target_format : str
        Target format ("PNG", "JPEG").
    quality : int, optional
        JPEG quality (1-100). Ignored for PNG. Default is 95.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Strip data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # Decode using OpenCV
        image_bytes = base64.b64decode(base64_string)
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("OpenCV failed to decode image")

        # Simple alpha channel handling for JPEG
        if target_format.upper() == "JPEG" and img.ndim == 3 and img.shape[2] == 4:
            # Convert RGBA to RGB with white background
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Save with format-specific parameters
        if target_format.upper() == "JPEG":
            success = cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif target_format.upper() == "PNG":
            success = cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            success = cv2.imwrite(output_path, img)

        if not success:
            raise ValueError(f"OpenCV failed to save image to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to convert and save image: {e}")
        return False


def _save_image_efficiently(base64_content: str, output_path: str, target_format: str, quality: int = 95) -> bool:
    """
    Save base64 image to disk using the most efficient method available.

    This function automatically chooses between direct write (when no conversion
    is needed) and conversion (when format change is required).

    Parameters
    ----------
    base64_content : str
        Base64-encoded image data.
    output_path : str
        Path where the image should be saved.
    target_format : str
        Target format ("PNG", "JPEG").
    quality : int, optional
        JPEG quality (1-100). Ignored for PNG. Default is 95.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    target_format_upper = target_format.upper()
    if target_format_upper == "JPG":
        target_format_upper = "JPEG"

    # Try direct write first if format matches (most efficient)
    if _is_base64_image_format(base64_content, target_format_upper):
        logger.debug(f"OpenCV Direct write: {target_format_upper}")
        return _base64_to_disk_direct(base64_content, output_path)
    else:
        # Format conversion required
        logger.debug(f"OpenCV Format conversion to {target_format_upper}")
        return _base64_to_disk_with_conversion(base64_content, output_path, target_format_upper, quality)


def save_images_to_disk(
    response_data: List[Dict[str, Any]],
    output_directory: str,
    save_charts: bool = True,
    save_tables: bool = True,
    save_infographics: bool = True,
    save_page_images: bool = False,
    save_raw_images: bool = False,
    count_images: bool = True,
    organize_by_type: bool = True,
    output_format: str = "jpeg",
) -> Dict[str, int]:
    """
    Save base64-encoded images from ingestion results to disk as actual image files.

    This utility extracts images from ingestion response data and saves them to disk
    with descriptive filenames that include the image subtype and page information.
    It provides granular control over which types of images to save.

    Parameters
    ----------
    response_data : List[Dict[str, Any]]
        List of document results from ingestion, each containing metadata with base64 images.
    output_directory : str
        Base directory where images will be saved.
    save_charts : bool, optional
        Whether to save chart images. Default is True.
    save_tables : bool, optional
        Whether to save table images. Default is True.
    save_infographics : bool, optional
        Whether to save infographic images. Default is True.
    save_page_images : bool, optional
        Whether to save page-as-image files. Default is False.
    save_raw_images : bool, optional
        Whether to save raw/natural images. Default is False.
    count_images : bool, optional
        Whether to count and log image statistics. Default is True.
    organize_by_type : bool, optional
        Whether to organize images into subdirectories by type. Default is True.
    output_format : str, optional
        Output image format for saved files. Supports "png" and "jpeg". Default is "jpeg".
        JPEG provides faster compression performance than PNG. Use "png" for lossless quality.

    Returns
    -------
    Dict[str, int]
        Dictionary with counts of images saved by type.

    Examples
    --------
    >>> from nv_ingest_client.util.image_disk_utils import save_images_to_disk
    >>>
    >>> # Save only charts and tables
    >>> counts = save_images_to_disk(
    ...     response_data,
    ...     "./output/images",
    ...     save_charts=True,
    ...     save_tables=True,
    ...     save_page_images=False
    ... )
    >>> print(f"Saved {counts['chart']} charts and {counts['table']} tables")
    """

    if not response_data:
        logger.warning("No response data provided")
        return {}

    # Initialize counters
    image_counts = {"chart": 0, "table": 0, "infographic": 0, "page_image": 0, "image": 0, "total": 0}

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    for doc_idx, document in enumerate(response_data):
        try:
            metadata = document.get("metadata", {})
            doc_type = document.get("document_type", "unknown")

            # Skip documents without image content
            image_content = metadata.get("content")
            if not image_content:
                continue

            # Get document info for naming
            source_metadata = metadata.get("source_metadata", {})
            source_id = source_metadata.get("source_id", f"document_{doc_idx}")
            clean_source_name = get_valid_filename(os.path.basename(source_id))

            content_metadata = metadata.get("content_metadata", {})
            subtype = content_metadata.get("subtype", "image")
            page_number = content_metadata.get("page_number", 0)

            # Apply filtering based on image subtype and user preferences
            should_save = False
            if subtype == "chart" and save_charts:
                should_save = True
            elif subtype == "table" and save_tables:
                should_save = True
            elif subtype == "infographic" and save_infographics:
                should_save = True
            elif subtype == "page_image" and save_page_images:
                should_save = True
            elif (
                doc_type == "image"
                and subtype not in ["chart", "table", "infographic", "page_image"]
                and save_raw_images
            ):
                should_save = True
                subtype = "image"  # Normalize subtype for consistent counting

            if not should_save:
                continue

            # Determine image file format - use configured output_format
            image_type = output_format.lower()
            if image_type not in ["png", "jpeg", "jpg"]:
                logger.warning(f"Unsupported output format '{output_format}', falling back to JPEG")
                image_type = "jpeg"
            # Normalize jpg to jpeg for internal consistency
            elif image_type == "jpg":
                image_type = "jpeg"

            # Create descriptive filename with source, page, and index information
            # Use consistent file extension (jpg for jpeg)
            file_ext = "jpg" if image_type == "jpeg" else image_type

            if organize_by_type:
                # Organize into subdirectories by image type
                type_dir = os.path.join(output_directory, subtype)
                os.makedirs(type_dir, exist_ok=True)
                image_filename = f"{clean_source_name}_p{page_number}_{doc_idx}.{file_ext}"
                image_path = os.path.join(type_dir, image_filename)
            else:
                # Flat directory structure with type in filename
                image_filename = f"{clean_source_name}_{subtype}_p{page_number}_{doc_idx}.{file_ext}"
                image_path = os.path.join(output_directory, image_filename)

            # Save image using efficient method
            try:
                # Determine JPEG quality based on image type (higher for important content)
                quality = 95  # High quality for charts, tables, infographics
                if subtype == "page_image":
                    quality = 85  # Slightly lower for page images
                elif subtype == "image":
                    quality = 90  # Good quality for raw images

                # Save image using efficient method (direct write when possible)
                success = _save_image_efficiently(image_content, image_path, image_type, quality=quality)

                if success:
                    # Update image type counters
                    image_counts[subtype] += 1
                    image_counts["total"] += 1
                    logger.debug(f"Saved {subtype} image: {image_path}")
                else:
                    logger.error(f"Failed to save {subtype} image for {clean_source_name}")

            except Exception as e:
                logger.error(f"Failed to save {subtype} image for {clean_source_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to process document {doc_idx}: {e}")
            continue

    # Log summary statistics if requested
    if count_images and image_counts["total"] > 0:
        logger.info(f"Successfully saved {image_counts['total']} images to {output_directory}")
        for img_type, count in image_counts.items():
            if img_type != "total" and count > 0:
                logger.info(f"  - {img_type}: {count}")
    elif count_images:
        logger.info("No images were saved (none met filter criteria)")

    return image_counts


def save_images_from_response(response: Dict[str, Any], output_directory: str, **kwargs) -> Dict[str, int]:
    """
    Convenience function to save images from a full API response.

    Parameters
    ----------
    response : Dict[str, Any]
        Full API response containing a "data" field with document results.
    output_directory : str
        Directory where images will be saved.
    **kwargs
        Additional arguments passed to save_images_to_disk().
        Includes output_format ("png" or "jpeg") and other filtering options.

    Returns
    -------
    Dict[str, int]
        Dictionary with counts of images saved by type.
    """

    if "data" not in response or not response["data"]:
        logger.warning("No data found in response")
        return {}

    return save_images_to_disk(response["data"], output_directory, **kwargs)


def save_images_from_ingestor_results(
    results: List[List[Dict[str, Any]]], output_directory: str, **kwargs
) -> Dict[str, int]:
    """
    Save images from Ingestor.ingest() results.

    Parameters
    ----------
    results : List[List[Dict[str, Any]]]
        Results from Ingestor.ingest(), where each inner list contains
        document results for one source file. Can also handle LazyLoadedList
        objects when save_to_disk=True is used.
    output_directory : str
        Directory where images will be saved.
    **kwargs
        Additional arguments passed to save_images_to_disk().
        Includes output_format ("png" or "jpeg") and other filtering options.

    Returns
    -------
    Dict[str, int]
        Dictionary with counts of images saved by type.
    """

    # Flatten results from multiple documents into single list
    all_documents = []
    for doc_results in results:
        if isinstance(doc_results, list):
            # Standard list of document results
            all_documents.extend(doc_results)
        elif hasattr(doc_results, "__iter__") and hasattr(doc_results, "__len__"):
            # Handle LazyLoadedList or other sequence-like objects
            try:
                all_documents.extend(list(doc_results))
            except Exception as e:
                logger.warning(f"Failed to process document results: {e}")
                continue
        else:
            # Handle single document case
            all_documents.append(doc_results)

    return save_images_to_disk(all_documents, output_directory, **kwargs)
