# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import pandas as pd
from pydantic import BaseModel, Field
from unittest.mock import patch, MagicMock

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
import nv_ingest_api.internal.store.embed_text_upload as module_under_test
from nv_ingest_api.internal.store.embed_text_upload import store_text_embeddings_internal

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


class DummyTaskConfigModel(BaseModel):
    params: dict = Field(default_factory=lambda: {"example_key": "example_value"})


@pytest.fixture
def dummy_df():
    return pd.DataFrame(
        [
            {
                "metadata": {
                    "embedding": [0.1, 0.2, 0.3],
                    "content": "dummy content",
                    "source_metadata": {"source_location": "dummy/location"},
                    "content_metadata": {"field": "value"},
                },
                "document_type": ContentTypeEnum.IMAGE,
            }
        ]
    )


@pytest.fixture
def dummy_task_config():
    return {
        "minio_access_key": "key",
        "minio_secret_key": "secret",
        "minio_endpoint": "mock_endpoint",
        "minio_bucket_name": "mock_bucket",
        "milvus_address": "mock_milvus",
        "milvus_host": "mock_host",
        "milvus_port": 1234,
        "collection_name": "mock_collection",
    }


def test_store_text_embeddings_internal_with_dict_config(dummy_df):
    mock_params = {"params": {"foo": "bar"}}

    with patch(f"{MODULE_UNDER_TEST}._upload_text_embeddings", return_value=dummy_df) as mock_upload:
        result = store_text_embeddings_internal(
            dummy_df,
            mock_params,
            store_config=EmbeddingStorageSchema(),
        )

        mock_upload.assert_called_once()
        args, kwargs = mock_upload.call_args
        assert args[0].equals(dummy_df)
        assert args[1]["foo"] == "bar"
        assert ContentTypeEnum.EMBEDDING in args[1]["content_types"]
        assert result.equals(dummy_df)


def test_store_text_embeddings_internal_with_model_config(dummy_df):
    with patch(f"{MODULE_UNDER_TEST}._upload_text_embeddings", return_value=dummy_df) as mock_upload:
        result = store_text_embeddings_internal(
            dummy_df,
            DummyTaskConfigModel(),
            store_config=EmbeddingStorageSchema(),
        )

        mock_upload.assert_called_once()
        args, kwargs = mock_upload.call_args
        assert args[0].equals(dummy_df)
        assert args[1]["example_key"] == "example_value"
        assert ContentTypeEnum.EMBEDDING in args[1]["content_types"]
        assert result.equals(dummy_df)


def test_store_text_embeddings_internal_raises_enriched_error(dummy_df):
    with patch(f"{MODULE_UNDER_TEST}._upload_text_embeddings", side_effect=RuntimeError("upload failed")):
        with pytest.raises(RuntimeError) as excinfo:
            store_text_embeddings_internal(
                dummy_df,
                {"params": {}},
                store_config=EmbeddingStorageSchema(),
            )
        assert "_store_embeddings: Failed to store embeddings: upload failed" in str(excinfo.value)


def test_store_text_embeddings_internal_ignores_unused_params(dummy_df):
    # Just verify it works when all optional params are given but ignored
    with patch(f"{MODULE_UNDER_TEST}._upload_text_embeddings", return_value=dummy_df) as mock_upload:
        result = store_text_embeddings_internal(
            dummy_df,
            {"params": {}},
            store_config=EmbeddingStorageSchema(),
            execution_trace_log={"trace_id": "abc123"},
        )
        mock_upload.assert_called_once()
        assert result.equals(dummy_df)


@patch(f"{MODULE_UNDER_TEST}.RemoteBulkWriter")
@patch(f"{MODULE_UNDER_TEST}.Collection")
@patch(f"{MODULE_UNDER_TEST}.connections.connect")
@patch(f"{MODULE_UNDER_TEST}.Minio")
def test_upload_text_embeddings_happy_path(
    mock_minio, mock_connect, mock_collection, mock_bulk_writer, dummy_df, dummy_task_config
):
    # Setup mocks
    mock_client = mock_minio.return_value
    mock_client.bucket_exists.return_value = False

    mock_writer = mock_bulk_writer.return_value
    mock_writer.append_row = MagicMock()
    mock_writer.commit = MagicMock()

    mock_collection.return_value.schema = "mock_schema"

    # Execute
    result = module_under_test._upload_text_embeddings(dummy_df.copy(), dummy_task_config)

    # Validate MinIO client init
    mock_minio.assert_called_once_with(
        "mock_endpoint",
        access_key="key",
        secret_key="secret",
        session_token=None,
        secure=False,
        region=None,
    )

    # Validate bucket check and creation
    mock_client.bucket_exists.assert_called_once_with("mock_bucket")
    mock_client.make_bucket.assert_called_once_with("mock_bucket")

    # Validate Milvus connection
    mock_connect.assert_called_once_with(
        address="mock_milvus",
        uri="http://milvus:19530:1234",
        host="mock_host",
        port=1234,
    )

    # Validate writer usage
    mock_bulk_writer.assert_called_once()
    mock_writer.append_row.assert_called_once()
    mock_writer.commit.assert_called_once()

    # Validate dataframe modification
    updated_metadata = result.iloc[0]["metadata"]
    assert "uploaded_embedding_url" in updated_metadata["embedding_metadata"]
    assert updated_metadata["embedding_metadata"]["uploaded_embedding_url"] == "embeddings"


@patch(f"{MODULE_UNDER_TEST}.RemoteBulkWriter")
@patch(f"{MODULE_UNDER_TEST}.Collection")
@patch(f"{MODULE_UNDER_TEST}.connections.connect")
@patch(f"{MODULE_UNDER_TEST}.Minio")
def test_upload_text_embeddings_raises_wrapped_error(
    mock_minio, mock_connect, mock_collection, mock_bulk_writer, dummy_df, dummy_task_config
):
    # Simulate an error in bucket check
    mock_client = mock_minio.return_value
    mock_client.bucket_exists.side_effect = RuntimeError("simulated error")

    with pytest.raises(RuntimeError) as excinfo:
        module_under_test._upload_text_embeddings(dummy_df.copy(), dummy_task_config)

    assert "upload_embeddings: Error uploading embeddings." in str(excinfo.value)
    assert "simulated error" in str(excinfo.value)
