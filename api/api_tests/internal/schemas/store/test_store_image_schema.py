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
    assert schema.raise_on_failure is False


def test_image_storage_module_schema_accepts_explicit_values():
    schema = ImageStorageModuleSchema(structured=False, images=False, raise_on_failure=True)
    assert schema.structured is False
    assert schema.images is False
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
