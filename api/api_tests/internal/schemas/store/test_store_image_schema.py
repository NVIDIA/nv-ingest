# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.store.store_image_schema import (
    ImageStorageModuleSchema,
    _DEFAULT_STORAGE_URI,
)


def test_image_storage_module_schema_defaults():
    schema = ImageStorageModuleSchema()
    assert schema.structured is True
    assert schema.images is True
    assert schema.storage_uri == _DEFAULT_STORAGE_URI
    assert schema.storage_options == {}
    assert schema.public_base_url is None
    assert schema.raise_on_failure is False


def test_image_storage_module_schema_accepts_explicit_values(tmp_path):
    storage_uri = f"file://{tmp_path}/disk"
    schema = ImageStorageModuleSchema(
        structured=False,
        images=False,
        storage_uri=storage_uri,
        storage_options={"foo": "bar"},
        public_base_url="http://public",
        raise_on_failure=True,
    )
    assert schema.structured is False
    assert schema.images is False
    assert schema.storage_uri == storage_uri
    assert schema.storage_options == {"foo": "bar"}
    assert schema.public_base_url == "http://public"
    assert schema.raise_on_failure is True


def test_image_storage_module_schema_accepts_truthy_values():
    schema = ImageStorageModuleSchema(structured=1, images="True", raise_on_failure=0)
    assert schema.structured is True
    assert schema.images is True
    assert schema.raise_on_failure is False


def test_image_storage_module_schema_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        ImageStorageModuleSchema(structured=True, images=True, extra_field="oops")
    assert "Extra inputs are not permitted" in str(excinfo.value)


def test_image_storage_module_schema_requires_storage_uri():
    with pytest.raises(ValidationError) as excinfo:
        ImageStorageModuleSchema(storage_uri="")
    assert "`storage_uri` must be provided" in str(excinfo.value)
