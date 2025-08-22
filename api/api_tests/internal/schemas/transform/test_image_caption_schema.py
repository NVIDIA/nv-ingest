# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema


def test_image_caption_extraction_schema_defaults():
    schema = ImageCaptionExtractionSchema()
    assert schema.api_key == ""
    assert schema.endpoint_url.startswith("https://")
    assert schema.prompt.startswith("Caption")
    assert schema.model_name.startswith("nvidia/")
    assert schema.raise_on_failure is False


def test_image_caption_extraction_schema_accepts_custom_values():
    schema = ImageCaptionExtractionSchema(
        api_key="mykey",
        endpoint_url="https://custom.endpoint",
        prompt="Describe this image in detail.",
        model_name="custom/model",
        raise_on_failure=True,
    )
    assert schema.api_key == "mykey"
    assert schema.endpoint_url == "https://custom.endpoint"
    assert schema.prompt == "Describe this image in detail."
    assert schema.model_name == "custom/model"
    assert schema.raise_on_failure is True


def test_image_caption_extraction_schema_accepts_truthy_values():
    schema = ImageCaptionExtractionSchema(raise_on_failure=1)
    assert schema.raise_on_failure is True

    schema = ImageCaptionExtractionSchema(raise_on_failure=0)
    assert schema.raise_on_failure is False


def test_image_caption_extraction_schema_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        ImageCaptionExtractionSchema(extra_field="oops")
    assert "Extra inputs are not permitted" in str(excinfo.value)
