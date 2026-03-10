# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema


def test_image_dedup_schema_defaults():
    schema = ImageDedupSchema()
    assert schema.raise_on_failure is False


def test_image_dedup_schema_accepts_true():
    schema = ImageDedupSchema(raise_on_failure=True)
    assert schema.raise_on_failure is True


def test_image_dedup_schema_rejects_non_bool():
    with pytest.raises(ValidationError) as excinfo:
        ImageDedupSchema(raise_on_failure="yes")  # Should reject string
    assert "Input should be a valid boolean" in str(excinfo.value)


def test_image_dedup_schema_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        ImageDedupSchema(raise_on_failure=True, extra_field="oops")
    assert "Extra inputs are not permitted" in str(excinfo.value)
