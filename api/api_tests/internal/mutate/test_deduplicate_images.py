# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pytest
import pandas as pd
from typing import Dict, Union

from nv_ingest_api.internal.mutate.deduplicate import _hash_content, deduplicate_images_internal


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
