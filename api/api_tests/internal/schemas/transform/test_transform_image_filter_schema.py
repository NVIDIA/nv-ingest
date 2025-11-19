# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema


def test_image_filter_schema_defaults():
    schema = ImageFilterSchema()
    assert schema.raise_on_failure is False
    assert schema.cpu_only is False


def test_image_filter_schema_accepts_strict_true_false():
    schema = ImageFilterSchema(raise_on_failure=True, cpu_only=False)
    assert schema.raise_on_failure is True
    assert schema.cpu_only is False

    schema = ImageFilterSchema(raise_on_failure=False, cpu_only=True)
    assert schema.raise_on_failure is False
    assert schema.cpu_only is True


def test_image_filter_schema_rejects_non_bool_values():
    with pytest.raises(ValidationError) as excinfo:
        ImageFilterSchema(raise_on_failure="yes", cpu_only=1)
    message = str(excinfo.value)
    assert "Input should be a valid boolean" in message


def test_image_filter_schema_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        ImageFilterSchema(raise_on_failure=True, cpu_only=False, extra_field="oops")
    assert "Extra inputs are not permitted" in str(excinfo.value)
