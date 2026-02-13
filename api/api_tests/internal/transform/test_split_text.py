# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from unittest.mock import patch, MagicMock

import pandas as pd

import nv_ingest_api.internal.transform.split_text as module_under_test

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@patch(f"{MODULE_UNDER_TEST}._get_tokenizer")
@patch(f"{MODULE_UNDER_TEST}.os.path.exists", return_value=False)
@patch(f"{MODULE_UNDER_TEST}.os.environ.get", return_value="/mock/model/path")
def test_transform_text_split_and_tokenize_internal_happy_path(mock_environ_get, mock_exists, mock_get_tokenizer):
    # Setup a mock tokenizer
    mock_tokenizer = MagicMock()
    # Produce fake offsets
    mock_tokenizer.encode_plus.return_value = {"offset_mapping": [(0, 4), (5, 9), (10, 19)]}
    mock_get_tokenizer.return_value = mock_tokenizer

    # Dummy dataframe
    df = pd.DataFrame(
        [
            {
                "metadata": {
                    "content": "This is a sample document for testing.",
                    "source_metadata": {"source_type": "text"},
                },
                "document_type": module_under_test.ContentTypeEnum.TEXT,
            },
            {
                "metadata": {
                    "content": "This won't be split.",
                    "source_metadata": {"source_type": "other"},
                },
                "document_type": module_under_test.ContentTypeEnum.TEXT,
            },
        ]
    )

    task_config = {
        "tokenizer": "mock_tokenizer",
        "chunk_size": 2,
        "chunk_overlap": 1,
        "params": {"split_source_types": ["text"]},
    }

    result = module_under_test.transform_text_split_and_tokenize_internal(
        df,
        task_config,
        transform_config=module_under_test.TextSplitterSchema(),
        execution_trace_log=None,
    )

    # Assert cached tokenizer was loaded with correct identifier
    mock_get_tokenizer.assert_called_once_with("mock_tokenizer", token=None)

    # Assert that splitting occurred for first row only
    assert not result.empty
    split_contents = result["metadata"].apply(lambda m: m.get("content", ""))
    assert any("This" in c for c in split_contents)
    assert any("won't" in c for c in split_contents)


@patch(f"{MODULE_UNDER_TEST}._get_tokenizer")
@patch(f"{MODULE_UNDER_TEST}.os.path.exists", return_value=False)
def test_transform_text_split_and_tokenize_internal_no_matching_source_type(mock_exists, mock_get_tokenizer):
    # Dummy dataframe with unmatched source_type
    df = pd.DataFrame(
        [
            {
                "metadata": {
                    "content": "Unmatched document.",
                    "source_metadata": {"source_type": "other"},
                },
                "document_type": module_under_test.ContentTypeEnum.TEXT,
            },
        ]
    )

    result = module_under_test.transform_text_split_and_tokenize_internal(
        df,
        task_config={"params": {"split_source_types": ["nonexistent"]}},
        transform_config=module_under_test.TextSplitterSchema(),
        execution_trace_log=None,
    )

    # Should return the same DataFrame without loading a tokenizer
    assert result.equals(df)
    mock_get_tokenizer.assert_not_called()


def test_split_into_chunks_empty_text():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode_plus.return_value = {"offset_mapping": []}

    chunks = module_under_test._split_into_chunks("", mock_tokenizer)
    assert chunks == []


def test_build_split_documents_skips_empty_chunks():
    row = SimpleNamespace(document_type="text", metadata={"existing": "value"})
    chunks = ["  ", None, "", "Valid chunk"]

    documents = module_under_test._build_split_documents(row, chunks)

    # Validate
    assert len(documents) == 1
    assert documents[0]["metadata"]["content"] == "Valid chunk"
    assert documents[0]["document_type"] == module_under_test.ContentTypeEnum.TEXT.value


def test_build_split_documents_handles_no_metadata():
    # Row without metadata attribute
    row = SimpleNamespace(document_type="text")
    chunks = ["Chunk A", "Chunk B"]

    documents = module_under_test._build_split_documents(row, chunks)

    assert len(documents) == 2
    for doc, chunk in zip(documents, chunks):
        assert doc["metadata"]["content"] == chunk


def test_split_into_chunks_happy_path():
    text = "This is a sample document for testing splitting."

    # Mock tokenizer behavior
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode_plus.return_value = {
        "offset_mapping": [(0, 8), (7, 16), (17, 19), (20, 27), (28, 36), (37, 45), (46, 55)]
    }

    chunks = module_under_test._split_into_chunks(text, mock_tokenizer, chunk_size=3, chunk_overlap=1)

    # Validate based on actual offset mapping boundaries
    expected_chunks = [
        text[0:17],  # (0,4) to (8,9) -> ends at 7
        text[17:28],  # (8,9) to (17,19) -> ends at 19
        text[28:46],  # (17,19) to (28,36) -> ends at 36
        text[46:46],  # (28,36) to (46,55) -> ends at 55
    ]

    assert len(chunks) == len(expected_chunks)
    for actual, expected in zip(chunks, expected_chunks):
        assert actual == expected
