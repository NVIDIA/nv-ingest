# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import pandas as pd
from typing import Any, Dict, Union, Optional, List

from nv_ingest_api.interface.mutate import filter_images
from nv_ingest_api.interface.mutate import deduplicate_images


# Dummy functions to replace filter_images_internal during testing.
def dummy_success(
    df: pd.DataFrame, task_params: Dict[str, Union[int, float, bool]], trace_log: Optional[List[Any]] = None
) -> pd.DataFrame:
    # Simulate processing by adding a new column.
    return df.assign(processed=True)


def dummy_capture(
    df: pd.DataFrame, task_params: Dict[str, Union[int, float, bool]], trace_log: Optional[List[Any]] = None
) -> pd.DataFrame:
    # Capture the parameters for later inspection.
    dummy_capture.captured_params = task_params
    dummy_capture.captured_trace = trace_log
    return df


def dummy_exception(
    df: pd.DataFrame, task_params: Dict[str, Union[int, float, bool]], trace_log: Optional[List[Any]] = None
) -> pd.DataFrame:
    raise ValueError("Dummy error in filter_images_internal")


# Initialize captured variables.
dummy_capture.captured_params = {}
dummy_capture.captured_trace = None


def test_filter_images_success(monkeypatch):
    sample_df = pd.DataFrame({"document_type": ["IMAGE", "TEXT"], "metadata": [{"dummy": "data1"}, {"dummy": "data2"}]})

    # Override filter_images_internal with dummy_success.
    monkeypatch.setattr("nv_ingest_api.interface.mutate.filter_images_internal", dummy_success)

    result = filter_images(sample_df, min_size=100, max_aspect_ratio=3, min_aspect_ratio=1)
    expected = sample_df.assign(processed=True)
    pd.testing.assert_frame_equal(result, expected)


def test_filter_images_parameters(monkeypatch):
    sample_df = pd.DataFrame({"document_type": ["IMAGE"], "metadata": [{"dummy": "data"}]})

    monkeypatch.setattr("nv_ingest_api.interface.mutate.filter_images_internal", dummy_capture)

    trace_log = ["trace1", "trace2"]
    min_size = 150
    max_aspect_ratio = 4.5
    min_aspect_ratio = 2.5

    result = filter_images(
        sample_df,
        min_size=min_size,
        max_aspect_ratio=max_aspect_ratio,
        min_aspect_ratio=min_aspect_ratio,
        execution_trace_log=trace_log,
    )

    expected_params = {
        "min_size": min_size,
        "max_aspect_ratio": max_aspect_ratio,
        "min_aspect_ratio": min_aspect_ratio,
        "filter": True,
    }
    assert dummy_capture.captured_params == expected_params
    assert dummy_capture.captured_trace == trace_log
    pd.testing.assert_frame_equal(result, sample_df)


def test_filter_images_exception(monkeypatch):
    sample_df = pd.DataFrame({"document_type": ["IMAGE"], "metadata": [{"dummy": "data"}]})

    monkeypatch.setattr("nv_ingest_api.interface.mutate.filter_images_internal", dummy_exception)

    with pytest.raises(ValueError) as exc_info:
        filter_images(sample_df)
    assert (
        "filter_images: Error applying deduplication filter. Original error: Dummy error in filter_images_internal"
        in str(exc_info.value)
    )


@pytest.fixture(autouse=True)
def patch_deduplicate_globals(monkeypatch):
    import nv_ingest_api.internal.mutate.deduplicate as dedup_mod

    class DummyContentTypeEnum:
        IMAGE = "IMAGE"
        INFO_MSG = "INFO_MSG"

    monkeypatch.setattr(dedup_mod, "ContentTypeEnum", DummyContentTypeEnum)


def test_deduplicate_images_no_duplicates():
    """
    Test that a DataFrame with no duplicate IMAGE rows remains unchanged.
    """
    df = pd.DataFrame({"document_type": ["IMAGE", "TEXT"], "metadata": [{"content": "unique"}, {"content": "text"}]})
    result = deduplicate_images(df, hash_algorithm="md5")
    # Expect the same DataFrame as there are no duplicates.
    pd.testing.assert_frame_equal(result.sort_index().reset_index(drop=True), df.sort_index().reset_index(drop=True))


def test_deduplicate_images_with_duplicates_md5():
    """
    Test that duplicate IMAGE rows are removed using the md5 hash algorithm.
    """
    df = pd.DataFrame(
        {
            "document_type": ["IMAGE", "IMAGE", "TEXT"],
            "metadata": [{"content": "dup"}, {"content": "dup"}, {"content": "text"}],
        }
    )
    result = deduplicate_images(df, hash_algorithm="md5")
    # Expect one IMAGE row for the duplicate content and the TEXT row.
    expected = pd.DataFrame(
        {"document_type": ["IMAGE", "TEXT"], "metadata": [{"content": "dup"}, {"content": "text"}]}, index=[0, 2]
    )
    pd.testing.assert_frame_equal(
        result.sort_index().reset_index(drop=True), expected.sort_index().reset_index(drop=True)
    )


def test_deduplicate_images_with_duplicates_sha256():
    """
    Test that duplicate IMAGE rows are removed using the sha256 hash algorithm.
    """
    df = pd.DataFrame({"document_type": ["IMAGE", "IMAGE"], "metadata": [{"content": "dup"}, {"content": "dup"}]})
    result = deduplicate_images(df, hash_algorithm="sha256")
    expected = pd.DataFrame({"document_type": ["IMAGE"], "metadata": [{"content": "dup"}]}, index=[0])
    pd.testing.assert_frame_equal(
        result.sort_index().reset_index(drop=True), expected.sort_index().reset_index(drop=True)
    )


def test_deduplicate_images_preserves_non_image_rows():
    """
    Test that non-IMAGE rows are preserved after deduplication.
    """
    df = pd.DataFrame(
        {
            "document_type": ["IMAGE", "IMAGE", "TEXT", "PDF"],
            "metadata": [{"content": "dup"}, {"content": "dup"}, {"content": "text1"}, {"content": "pdf1"}],
        }
    )
    result = deduplicate_images(df, hash_algorithm="md5")
    # Expect one IMAGE row (deduplicated) plus all non-IMAGE rows.
    expected = pd.DataFrame(
        {
            "document_type": ["IMAGE", "TEXT", "PDF"],
            "metadata": [{"content": "dup"}, {"content": "text1"}, {"content": "pdf1"}],
        },
        index=[0, 2, 3],
    )
    pd.testing.assert_frame_equal(
        result.sort_index().reset_index(drop=True), expected.sort_index().reset_index(drop=True)
    )


def test_deduplicate_images_missing_required_columns():
    """
    Test that an exception is raised when the DataFrame is missing required columns.
    """
    df = pd.DataFrame({"wrong_column": [1, 2], "metadata": [{"content": "a"}, {"content": "b"}]})
    with pytest.raises(ValueError):
        deduplicate_images(df, hash_algorithm="md5")


def test_deduplicate_images_execution_trace_log_unused():
    """
    Test that the optional execution_trace_log parameter does not affect the output.
    """
    df = pd.DataFrame({"document_type": ["IMAGE", "IMAGE"], "metadata": [{"content": "dup"}, {"content": "dup"}]})
    trace_log = ["step1", "step2"]
    result = deduplicate_images(df, hash_algorithm="md5", execution_trace_log=trace_log)
    expected = pd.DataFrame({"document_type": ["IMAGE"], "metadata": [{"content": "dup"}]}, index=[0])
    pd.testing.assert_frame_equal(
        result.sort_index().reset_index(drop=True), expected.sort_index().reset_index(drop=True)
    )
