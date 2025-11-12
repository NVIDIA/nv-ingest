# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema


def test_image_storage_module_schema_defaults():
    schema = ImageStorageModuleSchema()
    assert schema.structured is True
    assert schema.images is True
    assert schema.enable_minio is True
    assert schema.enable_local_disk is False
    assert schema.local_output_path is None
    assert schema.raise_on_failure is False


def test_image_storage_module_schema_accepts_explicit_values(tmp_path):
    local_path = str(tmp_path / "disk")
    schema = ImageStorageModuleSchema(
        structured=False,
        images=False,
        enable_minio=False,
        enable_local_disk=True,
        local_output_path=local_path,
        raise_on_failure=True,
    )
    assert schema.structured is False
    assert schema.images is False
    assert schema.enable_minio is False
    assert schema.enable_local_disk is True
    assert schema.local_output_path == local_path
    assert schema.raise_on_failure is True


def test_image_storage_module_schema_accepts_truthy_values():
    schema = ImageStorageModuleSchema(structured=1, images="True", enable_local_disk=0, raise_on_failure=0)
    assert schema.structured is True
    assert schema.images is True
    assert schema.enable_local_disk is False
    assert schema.raise_on_failure is False


def test_image_storage_module_schema_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        ImageStorageModuleSchema(structured=True, images=True, extra_field="oops")
    assert "Extra inputs are not permitted" in str(excinfo.value)


def test_image_storage_module_schema_requires_at_least_one_backend():
    with pytest.raises(ValidationError) as excinfo:
        ImageStorageModuleSchema(enable_minio=False, enable_local_disk=False)
    assert "At least one storage backend must be enabled" in str(excinfo.value)


def test_image_storage_module_schema_requires_output_path_when_disk_enabled():
    with pytest.raises(ValidationError) as excinfo:
        ImageStorageModuleSchema(enable_minio=False, enable_local_disk=True)
    assert "`local_output_path` is required when `enable_local_disk` is True" in str(excinfo.value)
