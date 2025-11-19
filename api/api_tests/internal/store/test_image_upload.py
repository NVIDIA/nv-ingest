# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

import pytest
import pandas as pd
import nv_ingest_api.internal.store.image_upload as module_under_test
from nv_ingest_api.internal.store.image_upload import store_images_to_minio_internal
from unittest.mock import patch, MagicMock
from urllib.parse import quote

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@pytest.fixture
def dummy_df():
    encoded_content = base64.b64encode(b"dummy_image_content").decode()
    return pd.DataFrame(
        [
            {
                "metadata": {
                    "content": encoded_content,
                    "source_metadata": {"source_id": "abc123"},  # must be dict with source_id
                    "image_metadata": {},  # optional but needed if you want to assert image updates
                },
                "document_type": module_under_test.ContentTypeEnum.IMAGE,
            }
        ]
    )


@pytest.fixture
def dummy_task_config():
    return {
        "content_types": {module_under_test.ContentTypeEnum.IMAGE: True},
        "access_key": "key",
        "secret_key": "secret",
        "endpoint": "mock_endpoint",
    }


@pytest.fixture
def dummy_df_with_matching():
    return pd.DataFrame(
        [
            {
                "metadata": {"content": "image1"},
                "document_type": module_under_test.ContentTypeEnum.IMAGE,
            },
            {
                "metadata": {"content": "other"},
                "document_type": "TEXT",
            },
        ]
    )


@pytest.fixture
def dummy_df_no_matching():
    return pd.DataFrame(
        [
            {"metadata": {"content": "other"}, "document_type": "TEXT"},
        ]
    )


def test_ensure_bucket_exists_creates_bucket_if_missing():
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = False

    module_under_test._ensure_bucket_exists(mock_client, "my-bucket")

    mock_client.bucket_exists.assert_called_once_with("my-bucket")
    mock_client.make_bucket.assert_called_once_with("my-bucket")


def test_ensure_bucket_exists_skips_creation_if_exists():
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True

    module_under_test._ensure_bucket_exists(mock_client, "my-bucket")

    mock_client.bucket_exists.assert_called_once_with("my-bucket")
    mock_client.make_bucket.assert_not_called()


@patch(f"{MODULE_UNDER_TEST}._ensure_bucket_exists")
@patch(f"{MODULE_UNDER_TEST}.Minio")
def test_upload_images_to_minio_happy_path(mock_minio, mock_ensure_bucket_exists, dummy_df, dummy_task_config):
    mock_client = mock_minio.return_value
    mock_client.put_object = MagicMock()

    result = module_under_test._upload_images_to_minio(dummy_df.copy(), dummy_task_config)

    # Bucket ensured
    mock_ensure_bucket_exists.assert_called_once_with(mock_client, module_under_test._DEFAULT_BUCKET_NAME)

    # Assert put_object called
    mock_client.put_object.assert_called_once()
    args, kwargs = mock_client.put_object.call_args
    assert args[0] == module_under_test._DEFAULT_BUCKET_NAME
    assert isinstance(args[2], BytesIO)
    assert args[2].getvalue() == b"dummy_image_content"

    # Assert metadata updated
    updated_metadata = result.iloc[0]["metadata"]
    expected_url_prefix = (
        f"{module_under_test._DEFAULT_READ_ADDRESS}/{module_under_test._DEFAULT_BUCKET_NAME}/"
        f"{quote('abc123', safe='')}/0.png"
    )
    assert updated_metadata["source_metadata"]["source_location"] == expected_url_prefix
    assert updated_metadata["image_metadata"]["uploaded_image_url"] == expected_url_prefix


def test_upload_images_to_minio_raises_on_invalid_content_types(dummy_df):
    with pytest.raises(ValueError) as excinfo:
        module_under_test._upload_images_to_minio(dummy_df, {})
    assert "Invalid configuration: 'content_types'" in str(excinfo.value)


@patch(f"{MODULE_UNDER_TEST}._ensure_bucket_exists")
@patch(f"{MODULE_UNDER_TEST}.Minio")
def test_upload_images_to_minio_skips_invalid_rows(mock_minio, mock_ensure_bucket_exists):
    # Row with missing 'metadata'
    df = pd.DataFrame([{"document_type": "IMAGE", "metadata": None}])
    mock_client = mock_minio.return_value

    # Should log error but continue gracefully without raising
    result = module_under_test._upload_images_to_minio(
        df.copy(), {"content_types": {"IMAGE": True}, "access_key": "key", "secret_key": "secret"}
    )
    # Ensure the row is still unchanged
    assert result.iloc[0]["metadata"] is None

    # Ensure no put_object was called
    mock_client.put_object.assert_not_called()


def test_store_images_to_minio_internal_raises_on_missing_content_types(dummy_df_with_matching):
    with pytest.raises(ValueError) as excinfo:
        store_images_to_minio_internal(dummy_df_with_matching, {}, {})
    assert "Task configuration must include a valid 'content_types'" in str(excinfo.value)


def test_store_images_to_minio_internal_raises_on_missing_document_type(dummy_task_config):
    df_missing_column = pd.DataFrame([{"metadata": {"content": "image1"}}])
    with pytest.raises(ValueError) as excinfo:
        store_images_to_minio_internal(df_missing_column, dummy_task_config, {})
    assert "Input DataFrame must contain a 'document_type'" in str(excinfo.value)


@patch(f"{MODULE_UNDER_TEST}._upload_images_to_minio")
def test_store_images_to_minio_internal_no_matching_returns_df(mock_upload, dummy_df_no_matching, dummy_task_config):
    result = store_images_to_minio_internal(dummy_df_no_matching.copy(), dummy_task_config, {})
    # Upload should not be called
    mock_upload.assert_not_called()
    assert result.equals(dummy_df_no_matching)


@patch(f"{MODULE_UNDER_TEST}._upload_images_to_minio")
def test_store_images_to_minio_internal_matching_calls_upload(mock_upload, dummy_df_with_matching, dummy_task_config):
    # Mock return
    dummy_return_df = pd.DataFrame(
        [{"metadata": {"content": "updated"}, "document_type": module_under_test.ContentTypeEnum.IMAGE}]
    )
    mock_upload.return_value = dummy_return_df

    result = store_images_to_minio_internal(dummy_df_with_matching.copy(), dummy_task_config, {})

    # Correct assertion pattern to handle DataFrame
    mock_upload.assert_called_once()
    args, kwargs = mock_upload.call_args
    assert args[0].equals(dummy_df_with_matching)
    assert args[1] == dummy_task_config

    assert result.equals(dummy_return_df)
