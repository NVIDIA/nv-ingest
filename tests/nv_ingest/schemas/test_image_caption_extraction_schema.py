import pytest
from pydantic import ValidationError

from nv_ingest.schemas import ImageCaptionExtractionSchema


def test_valid_schema():
    # Test with all required fields and optional defaults
    valid_data = {
        "api_key": "your-api-key-here",
    }
    schema = ImageCaptionExtractionSchema(**valid_data)
    assert schema.api_key == "your-api-key-here"
    assert schema.endpoint_url == "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
    assert schema.prompt == "Caption the content of this image:"
    assert schema.raise_on_failure is False


def test_valid_schema_with_custom_values():
    # Test with all fields including custom values for optional fields
    valid_data = {
        "api_key": "your-api-key-here",
        "endpoint_url": "https://custom.api.endpoint",
        "prompt": "Describe the image:",
        "raise_on_failure": True,
    }
    schema = ImageCaptionExtractionSchema(**valid_data)
    assert schema.api_key == "your-api-key-here"
    assert schema.endpoint_url == "https://custom.api.endpoint"
    assert schema.prompt == "Describe the image:"
    assert schema.raise_on_failure is True


def test_missing_api_key():
    # Test with missing required field `api_key`
    with pytest.raises(ValidationError) as exc_info:
        ImageCaptionExtractionSchema()
    assert "field required" in str(exc_info.value)


def test_invalid_extra_field():
    # Test with an additional field that should be forbidden
    data_with_extra_field = {
        "api_key": "your-api-key-here",
        "extra_field": "should_not_be_allowed"
    }
    with pytest.raises(ValidationError) as exc_info:
        ImageCaptionExtractionSchema(**data_with_extra_field)
    assert "extra fields not permitted" in str(exc_info.value)


def test_invalid_field_types():
    # Test with wrong types for optional fields
    invalid_data = {
        "api_key": "your-api-key-here",
        "endpoint_url": 12345,  # invalid type
        "prompt": 123,  # invalid type
        "raise_on_failure": "not_boolean"  # invalid type
    }
    with pytest.raises(ValidationError) as exc_info:
        ImageCaptionExtractionSchema(**invalid_data)


def test_default_values():
    # Test that default values are correctly assigned when not provided
    data = {"api_key": "your-api-key-here"}
    schema = ImageCaptionExtractionSchema(**data)
    assert schema.endpoint_url == "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
    assert schema.prompt == "Caption the content of this image:"
    assert schema.raise_on_failure is False
