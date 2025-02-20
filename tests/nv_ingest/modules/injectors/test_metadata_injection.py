# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

from nv_ingest.modules.injectors.metadata_injector import on_data
from nv_ingest.schemas.ingest_job_schema import DocumentTypeEnum
from nv_ingest.util.converters.type_mappings import doc_type_to_content_type
from nv_ingest_api.primitives.ingest_control_message import IngestControlMessage


# Dummy subclass to simulate the expected payload behavior.
class DummyIngestControlMessage(IngestControlMessage):
    def __init__(self, df):
        # No extra parameters required; simply store the DataFrame.
        self._df = df

    def payload(self, new_df=None):
        # If new_df is provided, update the stored DataFrame.
        if new_df is not None:
            self._df = new_df
        return self._df


def create_message(df):
    # Create an instance of the dummy subclass.
    return DummyIngestControlMessage(df)


def test_no_update_required():
    # Prepare a DataFrame where every row already has valid metadata.
    df = pd.DataFrame(
        [
            {
                "document_type": "pdf",
                "content": "content1",
                "source_id": 1,
                "source_name": "SourceA",
                "metadata": {
                    "content": "content1",
                    "other_info": "exists",
                },
            },
            {
                "document_type": "text",
                "content": "content2",
                "source_id": 2,
                "source_name": "SourceB",
                "metadata": {
                    "content": "content2",
                    "other_info": "exists",
                },
            },
        ]
    )
    msg = create_message(df)
    result = on_data(msg)
    # If no update was necessary, the payload remains unchanged.
    pd.testing.assert_frame_equal(result.payload(), df)


def test_update_required_missing_metadata():
    # Row missing the 'metadata' key.
    df = pd.DataFrame(
        [
            {
                "document_type": "pdf",
                "content": "pdf content",
                "source_id": 10,
                "source_name": "PDF_Source",
            }
        ]
    )
    msg = create_message(df)
    result = on_data(msg)
    updated_df = result.payload()
    metadata = updated_df.loc[0, "metadata"]

    expected_type = doc_type_to_content_type(DocumentTypeEnum("pdf")).name.lower()
    assert isinstance(metadata, dict)
    assert metadata["content"] == "pdf content"
    assert metadata["content_metadata"]["type"] == expected_type
    assert metadata["error_metadata"] is None
    # For non-image and non-text types, image_metadata and text_metadata should be None.
    assert metadata["image_metadata"] is None
    assert metadata["text_metadata"] is None
    assert metadata["source_metadata"] == {
        "source_id": 10,
        "source_name": "PDF_Source",
        "source_type": "pdf",
    }


def test_update_required_non_dict_metadata():
    # Row where existing metadata is not a dict.
    df = pd.DataFrame(
        [
            {
                "document_type": "png",
                "content": "image content",
                "source_id": 20,
                "source_name": "Image_Source",
                "metadata": "invalid_metadata",
            }
        ]
    )
    msg = create_message(df)
    result = on_data(msg)
    updated_df = result.payload()
    metadata = updated_df.loc[0, "metadata"]

    expected_type = doc_type_to_content_type(DocumentTypeEnum("png")).name.lower()
    assert metadata["content"] == "image content"
    assert metadata["content_metadata"]["type"] == expected_type
    assert metadata["error_metadata"] is None
    # For an image, image_metadata should be set.
    assert metadata["image_metadata"] == {"image_type": "png"}
    # text_metadata should remain None.
    assert metadata["text_metadata"] is None
    assert metadata["source_metadata"] == {
        "source_id": 20,
        "source_name": "Image_Source",
        "source_type": "png",
    }


def test_update_required_missing_content_in_metadata():
    # Row with a metadata dict that exists but is missing the 'content' key.
    df = pd.DataFrame(
        [
            {
                "document_type": "text",
                "content": "textual content",
                "source_id": 30,
                "source_name": "Text_Source",
                "metadata": {"other": "value"},
            }
        ]
    )
    msg = create_message(df)
    result = on_data(msg)
    updated_df = result.payload()
    metadata = updated_df.loc[0, "metadata"]

    expected_type = doc_type_to_content_type(DocumentTypeEnum("text")).name.lower()
    assert metadata["content"] == "textual content"
    assert metadata["content_metadata"]["type"] == expected_type
    # For text content, text_metadata should be set.
    assert metadata["text_metadata"] == {"text_type": "document"}
    # image_metadata should be None.
    assert metadata["image_metadata"] is None
    assert metadata["error_metadata"] is None
    assert metadata["source_metadata"] == {
        "source_id": 30,
        "source_name": "Text_Source",
        "source_type": "text",
    }


def test_empty_dataframe():
    # An empty DataFrame should be handled gracefully.
    df = pd.DataFrame([])
    msg = create_message(df)
    result = on_data(msg)
    pd.testing.assert_frame_equal(result.payload(), df)


def test_inner_exception_on_invalid_document_type():
    # If the document_type is invalid, DocumentTypeEnum() should raise an exception.
    df = pd.DataFrame(
        [
            {
                "document_type": "invalid",  # This value is not valid for DocumentTypeEnum.
                "content": "content",
                "source_id": 3,
                "source_name": "SourceX",
            }
        ]
    )
    msg = create_message(df)
    with pytest.raises(Exception):
        on_data(msg)


def test_outer_exception_when_payload_fails(monkeypatch):
    # Simulate a scenario where payload() fails by subclassing DummyIngestControlMessage.
    class FailingMessage(DummyIngestControlMessage):
        def payload(self, new_df=None):
            raise ValueError("Payload retrieval failed")

    msg = FailingMessage(pd.DataFrame([]))
    with pytest.raises(ValueError) as excinfo:
        on_data(msg)
    # Verify the augmented error message from on_data.
    assert "on_data: Failed to process IngestControlMessage" in str(excinfo.value)
