# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import pandas as pd
import pytest
from unittest.mock import patch

import nv_ingest_api.internal.store.image_upload as module_under_test
from nv_ingest_api.internal.store.image_upload import store_images_to_minio_internal

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@pytest.fixture
def dummy_df():
    encoded_content = base64.b64encode(b"dummy_image_content").decode()
    return pd.DataFrame(
        [
            {
                "metadata": {
                    "content": encoded_content,
                    "source_metadata": {"source_id": "abc123"},
                    "image_metadata": {},
                },
                "document_type": module_under_test.ContentTypeEnum.IMAGE,
            }
        ]
    )


@pytest.fixture
def dummy_df_with_matching():
    return pd.DataFrame(
        [
            {
                "metadata": {"content": base64.b64encode(b"image1").decode(), "source_metadata": {"source_id": "img1"}},
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


def test_upload_images_via_fsspec_writes_files(tmp_path, dummy_df):
    config = {
        "content_types": {module_under_test.ContentTypeEnum.IMAGE: True},
        "storage_uri": tmp_path.as_uri(),
    }

    result = module_under_test._upload_images_via_fsspec(dummy_df.copy(), config)

    expected_file = tmp_path / "abc123" / "0.png"
    assert expected_file.exists()
    assert expected_file.read_bytes() == b"dummy_image_content"

    metadata = result.iloc[0]["metadata"]
    source_meta = metadata["source_metadata"]
    image_meta = metadata["image_metadata"]

    assert source_meta["source_location"] == expected_file.as_uri()
    assert source_meta["local_source_location"] == str(expected_file)
    assert "uploaded_image_url" not in image_meta
    assert image_meta["uploaded_image_local_path"] == str(expected_file)


def test_upload_images_via_fsspec_uses_public_base_url(tmp_path, dummy_df):
    config = {
        "content_types": {module_under_test.ContentTypeEnum.IMAGE: True},
        "storage_uri": tmp_path.as_uri(),
        "public_base_url": "http://public-assets",
    }

    result = module_under_test._upload_images_via_fsspec(dummy_df.copy(), config)

    expected_file = tmp_path / "abc123" / "0.png"
    public_url = "http://public-assets/abc123/0.png"

    metadata = result.iloc[0]["metadata"]
    source_meta = metadata["source_metadata"]
    image_meta = metadata["image_metadata"]

    assert source_meta["source_location"] == public_url
    assert source_meta["local_source_location"] == str(expected_file)
    assert image_meta["uploaded_image_url"] == public_url
    assert image_meta["uploaded_image_local_path"] == str(expected_file)


def test_upload_images_via_fsspec_raises_on_invalid_content_types(dummy_df):
    with pytest.raises(ValueError) as excinfo:
        module_under_test._upload_images_via_fsspec(dummy_df, {"storage_uri": "file:///tmp"})
    assert "Invalid configuration: 'content_types'" in str(excinfo.value)


def test_upload_images_via_fsspec_requires_storage_uri(dummy_df):
    with pytest.raises(ValueError) as excinfo:
        module_under_test._upload_images_via_fsspec(
            dummy_df,
            {"content_types": {module_under_test.ContentTypeEnum.IMAGE: True}},
        )
    assert "`storage_uri` must be provided" in str(excinfo.value)


def test_upload_images_via_fsspec_skips_invalid_rows():
    df = pd.DataFrame([{"document_type": "IMAGE", "metadata": None}])
    result = module_under_test._upload_images_via_fsspec(
        df.copy(),
        {
            "content_types": {"IMAGE": True},
            "storage_uri": "file:///tmp",
        },
    )
    assert result.iloc[0]["metadata"] is None


def test_store_images_to_minio_internal_raises_on_missing_content_types(dummy_df_with_matching):
    with pytest.raises(ValueError) as excinfo:
        store_images_to_minio_internal(dummy_df_with_matching, {}, {})
    assert "Task configuration must include a valid 'content_types'" in str(excinfo.value)


def test_store_images_to_minio_internal_raises_on_missing_document_type(tmp_path):
    df_missing_column = pd.DataFrame([{"metadata": {"content": "image1"}}])
    config = {
        "content_types": {module_under_test.ContentTypeEnum.IMAGE: True},
        "storage_uri": tmp_path.as_uri(),
    }
    with pytest.raises(ValueError) as excinfo:
        store_images_to_minio_internal(df_missing_column, config, {})
    assert "Input DataFrame must contain a 'document_type'" in str(excinfo.value)


@patch(f"{MODULE_UNDER_TEST}._upload_images_via_fsspec")
def test_store_images_to_minio_internal_no_matching_returns_df(mock_upload, dummy_df_no_matching, tmp_path):
    config = {
        "content_types": {module_under_test.ContentTypeEnum.IMAGE: True},
        "storage_uri": tmp_path.as_uri(),
    }
    result = store_images_to_minio_internal(dummy_df_no_matching.copy(), config, {})
    mock_upload.assert_not_called()
    assert result.equals(dummy_df_no_matching)


@patch(f"{MODULE_UNDER_TEST}._upload_images_via_fsspec")
def test_store_images_to_minio_internal_matching_calls_upload(mock_upload, dummy_df_with_matching, tmp_path):
    dummy_return_df = pd.DataFrame(
        [{"metadata": {"content": "updated"}, "document_type": module_under_test.ContentTypeEnum.IMAGE}]
    )
    mock_upload.return_value = dummy_return_df
    config = {
        "content_types": {module_under_test.ContentTypeEnum.IMAGE: True},
        "storage_uri": tmp_path.as_uri(),
    }

    result = store_images_to_minio_internal(dummy_df_with_matching.copy(), config, {})

    mock_upload.assert_called_once()
    args, kwargs = mock_upload.call_args
    assert args[0].equals(dummy_df_with_matching)
    assert args[1] == config
    assert result.equals(dummy_return_df)
