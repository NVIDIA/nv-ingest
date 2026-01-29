# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pytest
import pandas as pd
from typing import Dict, Union

from nv_ingest_api.internal.mutate.deduplicate import (
    _hash_content,
    calculate_iou,
    deduplicate_by_bbox_internal,
    deduplicate_images_internal,
)


# Dummy enumeration for testing purposes.
class DummyContentTypeEnum:
    IMAGE = "IMAGE"
    INFO_MSG = "INFO_MSG"
    STRUCTURED = "STRUCTURED"


# Patch the global ContentTypeEnum in the deduplicate module.
@pytest.fixture(autouse=True)
def patch_globals(monkeypatch):
    import nv_ingest_api.internal.mutate.deduplicate as dedup_mod

    monkeypatch.setattr(dedup_mod, "ContentTypeEnum", DummyContentTypeEnum)


# Tests for _hash_content


def test_hash_content_md5():
    x = {"content": "test content"}
    expected = hashlib.new("md5", x["content"].encode()).digest()
    result = _hash_content(x, algorithm="md5")
    assert result == expected


def test_hash_content_sha256():
    x = {"content": "test content"}
    expected = hashlib.new("sha256", x["content"].encode()).digest()
    result = _hash_content(x, algorithm="sha256")
    assert result == expected


def test_hash_content_missing_key():
    x = {"no_content": "data"}
    with pytest.raises(Exception):
        _hash_content(x, algorithm="md5")


# Tests for deduplicate_images_internal


def test_deduplicate_missing_required_columns():
    # DataFrame missing 'document_type' and 'metadata' columns.
    df = pd.DataFrame({"wrong_column": [1, 2, 3]})
    task_config: Dict[str, Union[int, float, bool, str]] = {"hash_algorithm": "md5", "filter": True}
    with pytest.raises(ValueError):
        deduplicate_images_internal(df, task_config, mutate_config=None, execution_trace_log=None)


def test_deduplicate_no_image_rows():
    # DataFrame with no IMAGE rows.
    df = pd.DataFrame({"document_type": ["TEXT", "TEXT"], "metadata": [{"content": "text1"}, {"content": "text2"}]})
    task_config = {"hash_algorithm": "md5", "filter": True}
    result = deduplicate_images_internal(df, task_config, mutate_config=None, execution_trace_log=None)
    pd.testing.assert_frame_equal(result, df)


def test_deduplicate_removes_duplicates_md5():
    # DataFrame with duplicate IMAGE rows.
    df = pd.DataFrame(
        {
            "document_type": [DummyContentTypeEnum.IMAGE, DummyContentTypeEnum.IMAGE, "TEXT"],
            "metadata": [{"content": "duplicate"}, {"content": "duplicate"}, {"content": "non-image"}],
        }
    )
    task_config = {"hash_algorithm": "md5", "filter": True}
    result = deduplicate_images_internal(df, task_config, mutate_config=None, execution_trace_log=None)
    # Expect one IMAGE row from duplicates plus the non-image row.
    expected = pd.DataFrame(
        {
            "document_type": [DummyContentTypeEnum.IMAGE, "TEXT"],
            "metadata": [{"content": "duplicate"}, {"content": "non-image"}],
        },
        index=[0, 2],
    )
    pd.testing.assert_frame_equal(result.sort_index(), expected.sort_index())


def test_deduplicate_with_sha256():
    # DataFrame with duplicate IMAGE rows using SHA256.
    df = pd.DataFrame(
        {
            "document_type": [DummyContentTypeEnum.IMAGE, DummyContentTypeEnum.IMAGE],
            "metadata": [{"content": "same content"}, {"content": "same content"}],
        }
    )
    task_config = {"hash_algorithm": "sha256", "filter": True}
    result = deduplicate_images_internal(df, task_config, mutate_config=None, execution_trace_log=None)
    expected = pd.DataFrame(
        {"document_type": [DummyContentTypeEnum.IMAGE], "metadata": [{"content": "same content"}]}, index=[0]
    )
    pd.testing.assert_frame_equal(result.sort_index(), expected.sort_index())


# Tests for calculate_iou


@pytest.mark.parametrize(
    "bbox1, bbox2, expected_iou",
    [
        # No overlap
        ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),
        # Full overlap (identical boxes)
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
        # Partial overlap (50% overlap)
        ((0, 0, 10, 10), (5, 0, 15, 10), 1 / 3),
    ],
)
def test_calculate_iou(bbox1, bbox2, expected_iou):
    result = calculate_iou(bbox1, bbox2)
    assert abs(result - expected_iou) < 1e-6


# Tests for deduplicate_by_bbox_internal


def test_deduplicate_by_bbox_removes_overlapping_image():
    """Test that overlapping images are removed when prefer_structured=True."""
    df = pd.DataFrame(
        {
            "document_type": [DummyContentTypeEnum.IMAGE, DummyContentTypeEnum.STRUCTURED],
            "metadata": [
                {
                    "content": "image_content",
                    "content_metadata": {"page_number": 1},
                    "image_metadata": {"image_location": (0, 0, 100, 100), "image_location_max_dimensions": (100, 100)},
                },
                {
                    "content": "table_content",
                    "content_metadata": {"page_number": 1, "subtype": "table"},
                    "table_metadata": {"table_location": (0, 0, 100, 100), "table_location_max_dimensions": (100, 100)},
                },
            ],
        }
    )
    result = deduplicate_by_bbox_internal(df, iou_threshold=0.5, prefer_structured=True)
    # Image should be removed, only structured remains
    assert len(result) == 1
    assert result.iloc[0]["document_type"] == DummyContentTypeEnum.STRUCTURED


def test_deduplicate_by_bbox_removes_overlapping_structured():
    """Test that overlapping structured elements are removed when prefer_structured=False."""
    df = pd.DataFrame(
        {
            "document_type": [DummyContentTypeEnum.IMAGE, DummyContentTypeEnum.STRUCTURED],
            "metadata": [
                {
                    "content": "image_content",
                    "content_metadata": {"page_number": 1},
                    "image_metadata": {"image_location": (0, 0, 100, 100), "image_location_max_dimensions": (100, 100)},
                },
                {
                    "content": "table_content",
                    "content_metadata": {"page_number": 1, "subtype": "table"},
                    "table_metadata": {"table_location": (0, 0, 100, 100), "table_location_max_dimensions": (100, 100)},
                },
            ],
        }
    )
    result = deduplicate_by_bbox_internal(df, iou_threshold=0.5, prefer_structured=False)
    # Structured should be removed, only image remains
    assert len(result) == 1
    assert result.iloc[0]["document_type"] == DummyContentTypeEnum.IMAGE


def test_deduplicate_by_bbox_no_overlap():
    """Test that non-overlapping elements on same page are kept."""
    df = pd.DataFrame(
        {
            "document_type": [DummyContentTypeEnum.IMAGE, DummyContentTypeEnum.STRUCTURED],
            "metadata": [
                {
                    "content": "image_content",
                    "content_metadata": {"page_number": 1},
                    "image_metadata": {"image_location": (0, 0, 50, 50), "image_location_max_dimensions": (100, 100)},
                },
                {
                    "content": "table_content",
                    "content_metadata": {"page_number": 1, "subtype": "table"},
                    "table_metadata": {
                        "table_location": (60, 60, 100, 100),
                        "table_location_max_dimensions": (100, 100),
                    },
                },
            ],
        }
    )
    result = deduplicate_by_bbox_internal(df, iou_threshold=0.5, prefer_structured=True)
    # Both should be kept since no overlap
    assert len(result) == 2
