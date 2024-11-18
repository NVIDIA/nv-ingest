import pytest
from pydantic import ValidationError

from nv_ingest.schemas.image_extractor_schema import ImageConfigSchema, ImageExtractorSchema


def test_image_config_schema_valid():
    # Test valid data with both gRPC and HTTP endpoints
    config = ImageConfigSchema(
        auth_token="token123",
        yolox_endpoints=("grpc_service_url", "http_service_url"),
        yolox_infer_protocol="http"
    )
    assert config.auth_token == "token123"
    assert config.yolox_endpoints == ("grpc_service_url", "http_service_url")
    assert config.yolox_infer_protocol == "http"


def test_image_config_schema_valid_single_service():
    # Test valid data with only gRPC service
    config = ImageConfigSchema(
        yolox_endpoints=("grpc_service_url", None),
    )
    assert config.yolox_endpoints == ("grpc_service_url", None)
    assert config.yolox_infer_protocol == "grpc"

    # Test valid data with only HTTP service
    config = ImageConfigSchema(
        yolox_endpoints=(None, "http_service_url"),
    )
    assert config.yolox_endpoints == (None, "http_service_url")
    assert config.yolox_infer_protocol == "http"


def test_image_config_schema_invalid_both_services_empty():
    # Test invalid data with both gRPC and HTTP services empty
    with pytest.raises(ValidationError) as exc_info:
        ImageConfigSchema(yolox_endpoints=(None, None))
    errors = exc_info.value.errors()
    assert any("Both gRPC and HTTP services cannot be empty" in error['msg'] for error in errors)


def test_image_config_schema_empty_service_strings():
    # Test services that are empty strings or whitespace
    config = ImageConfigSchema(
        yolox_endpoints=(" ", "http_service_url")
    )
    assert config.yolox_endpoints == (None, "http_service_url")  # Cleaned empty strings are None


def test_image_config_schema_missing_infer_protocol():
    # Test infer_protocol default setting based on available service
    config = ImageConfigSchema(
        yolox_endpoints=("grpc_service_url", None)
    )
    assert config.yolox_infer_protocol == "grpc"


def test_image_config_schema_extra_field():
    # Test extra fields raise a validation error
    with pytest.raises(ValidationError):
        ImageConfigSchema(
            auth_token="token123",
            yolox_endpoints=("grpc_service_url", "http_service_url"),
            extra_field="should_not_be_allowed"
        )


def test_image_extractor_schema_valid():
    # Test valid data for ImageExtractorSchema with nested ImageConfigSchema
    config = ImageExtractorSchema(
        max_queue_size=10,
        n_workers=4,
        raise_on_failure=True,
        image_extraction_config=ImageConfigSchema(
            auth_token="token123",
            yolox_endpoints=("grpc_service_url", "http_service_url"),
            yolox_infer_protocol="http"
        )
    )
    assert config.max_queue_size == 10
    assert config.n_workers == 4
    assert config.raise_on_failure is True
    assert config.image_extraction_config.auth_token == "token123"


def test_image_extractor_schema_defaults():
    # Test default values for optional fields
    config = ImageExtractorSchema()
    assert config.max_queue_size == 1
    assert config.n_workers == 16
    assert config.raise_on_failure is False
    assert config.image_extraction_config is None


def test_image_extractor_schema_invalid_max_queue_size():
    # Test invalid type for max_queue_size
    with pytest.raises(ValidationError) as exc_info:
        ImageExtractorSchema(max_queue_size="invalid_type")
    errors = exc_info.value.errors()
    assert any(error['loc'] == ('max_queue_size',) and error['type'] == 'type_error.integer' for error in errors)


def test_image_extractor_schema_invalid_n_workers():
    # Test invalid type for n_workers
    with pytest.raises(ValidationError) as exc_info:
        ImageExtractorSchema(n_workers="invalid_type")
    errors = exc_info.value.errors()
    assert any(error['loc'] == ('n_workers',) and error['type'] == 'type_error.integer' for error in errors)


def test_image_extractor_schema_invalid_nested_config():
    # Test invalid nested image_extraction_config
    with pytest.raises(ValidationError) as exc_info:
        ImageExtractorSchema(
            image_extraction_config={"auth_token": "token123", "yolox_endpoints": (None, None)}
        )
    errors = exc_info.value.errors()
    assert any("Both gRPC and HTTP services cannot be empty" in error['msg'] for error in errors)
