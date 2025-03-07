# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
from typing import Dict, Any

import pandas as pd
import pytest
from datetime import datetime

from nv_ingest_api.interface.mutate import deduplicate_images, filter_images
from nv_ingest_api.internal.enums.common import ContentTypeEnum


def create_image_row(source: str, content: str, width: int, height: int) -> Dict[str, Any]:
    """
    Create a simulated image row dictionary for testing the image filter.

    Parameters
    ----------
    source : str
        A unique identifier for the image (e.g., file name).
    content : str
        A string representing the image content (for hashing).
    width : int
        Simulated image width.
    height : int
        Simulated image height.

    Returns
    -------
    Dict[str, Any]
        A dictionary representing an image row with metadata.
        Note: The 'document_type' is set to ContentTypeEnum.IMAGE so that the filter function
        recognizes the row as an image.
    """
    return {
        "source_name": source,
        "source_id": source,
        "content": content,
        "document_type": ContentTypeEnum.IMAGE,  # Important: use "image" to trigger filtering.
        "metadata": {
            "content": content,
            "content_url": "",
            "embedding": None,
            "source_metadata": {
                "source_name": source,
                "source_id": source,
                "source_location": "",
                "source_type": "png",
                "collection_id": "",
                "date_created": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "summary": "",
                "partition_id": -1,
                "access_level": "unknown",
            },
            "content_metadata": {
                "type": ContentTypeEnum.IMAGE,
                "description": "",
                "page_number": -1,
                "hierarchy": {
                    "page_count": -1,
                    "page": -1,
                    "block": -1,
                    "line": -1,
                    "span": -1,
                    "nearby_objects": {
                        "text": {"content": [], "bbox": [], "type": []},
                        "images": {"content": [], "bbox": [], "type": []},
                        "structured": {"content": [], "bbox": [], "type": []},
                    },
                },
                "subtype": "",
            },
            "audio_metadata": None,
            "text_metadata": None,
            # Initialize image_metadata as a dict containing dimensions.
            "image_metadata": {"caption": "", "width": width, "height": height},
            "table_metadata": None,
            "chart_metadata": None,
            "error_metadata": None,
            "info_message_metadata": None,
            "debug_metadata": None,
            "raise_on_failure": False,
        },
    }


def create_text_row(source: str, content: str) -> Dict[str, Any]:
    """
    Create a simulated text row dictionary for testing the image filter.

    Parameters
    ----------
    source : str
        A unique identifier for the text document.
    content : str
        The text content.

    Returns
    -------
    Dict[str, Any]
        A dictionary representing a text row with metadata.
    """
    return {
        "source_name": source,
        "source_id": source,
        "content": content,
        "document_type": "text",
        "metadata": {
            "content": content,
            "content_url": "",
            "embedding": None,
            "source_metadata": {
                "source_name": source,
                "source_id": source,
                "source_location": "",
                "source_type": "txt",
                "collection_id": "",
                "date_created": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "summary": "",
                "partition_id": -1,
                "access_level": "unknown",
            },
            "content_metadata": {
                "type": "text",
                "description": "",
                "page_number": -1,
                "hierarchy": {},
                "subtype": "",
            },
            "audio_metadata": None,
            "text_metadata": None,
            "image_metadata": None,
            "table_metadata": None,
            "chart_metadata": None,
            "error_metadata": None,
            "info_message_metadata": None,
            "debug_metadata": None,
            "raise_on_failure": False,
        },
    }


@pytest.mark.integration
def test_filter_images_integration() -> None:
    """
    Integration test for the filter_images function.

    This test constructs a DataFrame containing four rows:

      - Three image rows with simulated dimensions:
          1. A "good" image row: width=300, height=100
             (average size = 200, aspect ratio = 3.0) → Meets criteria.
          2. A "small" image row: width=50, height=50
             (average size = 50, aspect ratio = 1.0) → Fails min_size and min_aspect_ratio.
          3. A "wide" image row: width=500, height=50
             (average size = 275, aspect ratio = 10.0) → Fails max_aspect_ratio.
      - One non-image row (text) which should remain unaffected.

    The filter_images function is invoked with:
        - min_size = 128,
        - max_aspect_ratio = 5.0,
        - min_aspect_ratio = 2.0.

    Expected Outcome:
      - Only the "good" image row should be retained among images.
      - The text row should be preserved.
      - The final DataFrame should have 2 rows.
      - The internal hash column (_image_content_hash) should not be present.

    Raises
    ------
    AssertionError
        If the resulting DataFrame does not match the expected row count or if filtering fails.
    """
    # Create simulated image rows.
    good_image = create_image_row("good.png", "duplicate_image", width=300, height=100)
    small_image = create_image_row("small.png", "duplicate_image", width=50, height=50)
    wide_image = create_image_row("wide.png", "duplicate_image", width=500, height=50)

    # Create a non-image (text) row.
    text_row = create_text_row("doc.txt", "unique_text")

    # Build the input DataFrame using the provided utility (if available) or construct manually.
    df_input = pd.DataFrame([good_image, small_image, wide_image, text_row])

    # Call the filter_images function.
    result_df = filter_images(
        df_ledger=df_input,
        min_size=128,
        max_aspect_ratio=5.0,
        min_aspect_ratio=2.0,
    )

    # Expected outcome: 1 good image + 1 text row = 2 rows.
    expected_total_rows = 2
    assert (
        len(result_df) == expected_total_rows
    ), f"Expected {expected_total_rows} rows after filtering, got {len(result_df)}."

    # Verify that non-image rows are preserved.
    non_image_df = result_df[result_df["document_type"] != ContentTypeEnum.IMAGE]
    assert not non_image_df.empty, "Non-image rows should be preserved."

    # Verify that only one unique image row remains.
    image_df = result_df[result_df["document_type"] == ContentTypeEnum.IMAGE]
    assert len(image_df) == 1, f"Expected 1 unique image row after filtering, got {len(image_df)}."

    # Ensure the internal hash column is not present.
    assert "_image_content_hash" not in result_df.columns, "Internal hash column should not be present in the output."


@pytest.mark.integration
def test_deduplicate_images_integration():
    """
    Integration test for deduplicate_images.

    This test creates a DataFrame containing image and non-image rows to validate that
    duplicate images are removed based on content hashes, while non-image rows are preserved.

    Test Data:
        - Two image rows (document_type set to "image") with identical image content in
          metadata["content"] ("duplicate_image"). These are duplicates.
        - One text row (document_type "text") with unique content ("unique_text").
          This row should remain unaffected.

    Expected Behavior:
        - The deduplication process computes a hash for each image's metadata.content using the specified
          hashing algorithm (default "md5").
        - Duplicate image rows (with identical content) are removed.
        - Non-image rows are preserved.
        - The output DataFrame should have the same structure as the input, without an internal column
          (e.g. "_image_content_hash").

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the deduplicated DataFrame does not match the expected row count or if duplicate removal fails.
    """
    # Create dummy content for image and text rows.
    duplicate_image_content = "duplicate_image"  # This string is used to simulate image content.
    unique_text_content = "unique_text"

    # Build two duplicate image rows.
    image_metadata_template = {
        "content": duplicate_image_content,
        "content_url": "",
        "embedding": None,
        "source_metadata": {
            "source_name": "img1.png",
            "source_id": "img1.png",
            "source_location": "",
            "source_type": "png",
            "collection_id": "",
            "date_created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "summary": "",
            "partition_id": -1,
            "access_level": "unknown",
        },
        "content_metadata": {
            "type": ContentTypeEnum.IMAGE,
            "description": "",
            "page_number": -1,
            "hierarchy": {
                "page_count": -1,
                "page": -1,
                "block": -1,
                "line": -1,
                "span": -1,
                "nearby_objects": {
                    "text": {"content": [], "bbox": [], "type": []},
                    "images": {"content": [], "bbox": [], "type": []},
                    "structured": {"content": [], "bbox": [], "type": []},
                },
            },
            "subtype": "",
        },
        "audio_metadata": None,
        "text_metadata": None,
        # For image rows, image_metadata is initialized as an empty dict so that captions can be added.
        "image_metadata": {},
        "table_metadata": None,
        "chart_metadata": None,
        "error_metadata": None,
        "info_message_metadata": None,
        "debug_metadata": None,
        "raise_on_failure": False,
    }
    image_row_1 = {
        "source_name": "img1.png",
        "source_id": "img1.png",
        "content": duplicate_image_content,
        "document_type": ContentTypeEnum.IMAGE,
        "metadata": copy.deepcopy(image_metadata_template),
    }
    image_row_2 = {
        "source_name": "img2.png",
        "source_id": "img2.png",
        "content": duplicate_image_content,
        "document_type": ContentTypeEnum.IMAGE,
        "metadata": copy.deepcopy(image_metadata_template),
    }

    # Build a non-image (text) row.
    text_metadata = {
        "content": unique_text_content,
        "content_url": "",
        "embedding": None,
        "source_metadata": {
            "source_name": "doc1.txt",
            "source_id": "doc1.txt",
            "source_location": "",
            "source_type": "txt",
            "collection_id": "",
            "date_created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "summary": "",
            "partition_id": -1,
            "access_level": "unknown",
        },
        "content_metadata": {
            "type": "text",
            "description": "",
            "page_number": -1,
            "hierarchy": {},
            "subtype": "",
        },
        "audio_metadata": None,
        "text_metadata": None,
        "image_metadata": None,
        "table_metadata": None,
        "chart_metadata": None,
        "error_metadata": None,
        "info_message_metadata": None,
        "debug_metadata": None,
        "raise_on_failure": False,
    }
    text_row = {
        "source_name": "doc1.txt",
        "source_id": "doc1.txt",
        "content": unique_text_content,
        "document_type": "text",
        "metadata": text_metadata,
    }

    # Construct the input DataFrame.
    df_input = pd.DataFrame([image_row_1, image_row_2, text_row])

    # Call the deduplication function using the wrapper.
    # Use the default "md5" hashing algorithm.
    dedup_df = deduplicate_images(df_ledger=df_input, hash_algorithm="md5")

    # Expected: Only one unique image row (from the duplicates) plus the text row should remain.
    expected_row_count = 2
    assert (
        len(dedup_df) == expected_row_count
    ), f"Expected {expected_row_count} rows after deduplication, got {len(dedup_df)}."

    # Verify that non-image rows are preserved.
    non_image_df = dedup_df[dedup_df["document_type"] != ContentTypeEnum.IMAGE]
    assert not non_image_df.empty, "Non-image rows should be preserved."

    # Verify that only one unique image row remains.
    image_df = dedup_df[dedup_df["document_type"] == ContentTypeEnum.IMAGE]
    assert len(image_df) == 1, f"Expected 1 unique image row after deduplication, got {len(image_df)}."

    # Ensure that the internal hash column used for deduplication is not present.
    assert "_image_content_hash" not in dedup_df.columns, "Internal hash column should not be present in the output."
