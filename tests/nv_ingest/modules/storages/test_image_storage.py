# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from minio import Minio

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


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
