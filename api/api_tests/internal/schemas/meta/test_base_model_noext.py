# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.meta.base_model_noext import BaseModelNoExt


# Example subclass for testing
class ExampleSchema(BaseModelNoExt):
    name: str


def test_base_model_no_ext_accepts_defined_fields_only():
    schema = ExampleSchema(name="test")
    assert schema.name == "test"


def test_base_model_no_ext_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        ExampleSchema(name="test", extra_field="oops")
    assert "Extra inputs are not permitted" in str(excinfo.value)


def test_base_model_no_ext_missing_required_field():
    with pytest.raises(ValidationError):
        ExampleSchema()
