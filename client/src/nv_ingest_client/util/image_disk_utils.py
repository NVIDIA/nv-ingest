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

import logging
import os
from typing import Any, Dict, List

from nv_ingest_client.client.util.processing import get_valid_filename
from nv_ingest_api.util.image_processing.transforms import save_image_to_disk, _detect_base64_image_format

logger = logging.getLogger(__name__)


def _detect_extension_from_content(image_content: str) -> str:
    """
    Get file extension by detecting original image format.
    Falls back to .jpeg if detection fails or format is unknown.
    """
    DEFAULT_EXT = "jpg"  # must be either "jpg" or "png"
    try:
        fmt = _detect_base64_image_format(image_content).upper()
    except Exception:
        logger.warning("Image format detection failed; falling back to default '%s'.", DEFAULT_EXT)
        return DEFAULT_EXT
    ext_map = {
        "JPEG": "jpg",
        "JPG": "jpg",
        "PNG": "png",
    }
    ext = ext_map.get(fmt, None)
    if ext:
        return ext
    logger.warning("Unsupported image format '%s'; falling back to default '%s'.", fmt, DEFAULT_EXT)
    return DEFAULT_EXT


def save_images_to_disk(
    response_data: List[Dict[str, Any]],
    output_directory: str,
    save_charts: bool = True,
    save_tables: bool = True,
    save_infographics: bool = True,
    save_page_images: bool = False,
    save_raw_images: bool = False,
    organize_by_type: bool = True,
    output_format: str = "auto",
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
    organize_by_type : bool, optional
        Whether to organize images into subdirectories by type. Default is True.
    output_format : str, optional
        Output image format for saved files. Default is "auto".
        - "auto": Preserve original format (fastest, no conversion)
        - "jpeg": Convert to JPEG (smaller files, good compression)
        - "png": Convert to PNG (lossless quality)
        Use "auto" for maximum speed by avoiding format conversion.

    Returns
    -------
    Dict[str, int]
        Dictionary with counts of images saved by type.

    Raises
    ------
    ValueError
        If output_format is not supported.

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

    # Validate format upfront to fail fast
    normalized_format = output_format.lower()
    if normalized_format not in ["auto", "png", "jpeg", "jpg"]:
        raise ValueError(
            f"Unsupported output format: '{output_format}'. Supported formats: 'auto', 'png', 'jpeg', 'jpg'"
        )

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

            # Determine file extension and target format (format already validated upfront)
            if normalized_format in ["jpeg", "jpg"]:
                file_ext, target_format = "jpeg", "jpeg"
            elif normalized_format == "png":
                file_ext, target_format = "png", "png"
            else:  # normalized_format == "auto" - detect once and use result
                detected_ext = _detect_extension_from_content(image_content)
                if detected_ext == "png":
                    file_ext, target_format = "png", "png"
                else:  # detected_ext == "jpeg"
                    file_ext, target_format = "jpeg", "jpeg"

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

            # Save image using centralized API function
            try:
                success = save_image_to_disk(image_content, image_path, target_format)

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

    # Log summary statistics
    if image_counts["total"] > 0:
        logger.info(f"Successfully saved {image_counts['total']} images to {output_directory}")
        for img_type, count in image_counts.items():
            if img_type != "total" and count > 0:
                logger.info(f"  - {img_type}: {count}")
    else:
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
        Includes output_format ("auto", "png", or "jpeg") and other filtering options.

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
        Includes output_format ("auto", "png", or "jpeg") and other filtering options.

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
