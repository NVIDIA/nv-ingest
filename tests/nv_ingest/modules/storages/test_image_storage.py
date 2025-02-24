# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from minio import Minio

from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.modules.storages.image_storage import upload_images
from nv_ingest_api.primitives.ingest_control_message import IngestControlMessage


class MockMinioClient:
    def __init__(self, *args, **kwargs):
        pass

    def make_bucket(self, *args, **kwargs):
        return

    def put_object(self, *args, **kwargs):
        return

    def bucket_exists(self, *args, **kwargs):
        return True


@pytest.fixture
def mock_minio(mocker):
    def mock_minio_init(
        cls,
        *args,
        **kwargs,
    ):
        return MockMinioClient(*args, **kwargs)

    patched = mocker.patch.object(Minio, "__new__", new=mock_minio_init)
    yield patched


def test_upload_images(mock_minio):
    df = pd.DataFrame(
        {
            "document_type": [
                ContentTypeEnum.TEXT.value,
                ContentTypeEnum.IMAGE.value,
            ],
            "metadata": [
                {"content": "some text"},
                {
                    "content": "image_content",
                    "image_metadata": {
                        "image_type": "png",
                    },
                    "source_metadata": {
                        "source_id": "foo",
                    },
                },
            ],
        }
    )
    params = {"content_types": {"image": True, "structured": True}}

    msg = IngestControlMessage()
    msg.payload(df)
    df = msg.payload()

    result = upload_images(df, params)
    uploaded_image_url = result.iloc[1]["metadata"]["image_metadata"]["uploaded_image_url"]
    assert uploaded_image_url == "http://minio:9000/nv-ingest/foo/1.png"
