# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import pandas as pd
from typing import Any, Dict, Union, Optional, List

from nv_ingest_api.interface.mutate import filter_images


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
