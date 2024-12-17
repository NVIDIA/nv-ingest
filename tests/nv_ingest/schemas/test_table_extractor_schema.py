import pytest
from pydantic import ValidationError

from nv_ingest.schemas.table_extractor_schema import TableExtractorConfigSchema
from nv_ingest.schemas.table_extractor_schema import TableExtractorSchema


# Test cases for TableExtractorConfigSchema
def test_valid_config_with_grpc_only():
    config = TableExtractorConfigSchema(auth_token="valid_token", paddle_endpoints=("grpc://paddle_service", None))
    assert config.auth_token == "valid_token"
    assert config.paddle_endpoints == ("grpc://paddle_service", None)


def test_valid_config_with_http_only():
    config = TableExtractorConfigSchema(auth_token="valid_token", paddle_endpoints=(None, "http://paddle_service"))
    assert config.auth_token == "valid_token"
    assert config.paddle_endpoints == (None, "http://paddle_service")


def test_valid_config_with_both_services():
    config = TableExtractorConfigSchema(
        auth_token="valid_token", paddle_endpoints=("grpc://paddle_service", "http://paddle_service")
    )
    assert config.auth_token == "valid_token"
    assert config.paddle_endpoints == ("grpc://paddle_service", "http://paddle_service")


def test_invalid_config_empty_endpoints():
    with pytest.raises(ValidationError) as exc_info:
        TableExtractorConfigSchema(paddle_endpoints=(None, None))
    assert "Both gRPC and HTTP services cannot be empty for paddle_endpoints" in str(exc_info.value)


def test_invalid_extra_fields():
    with pytest.raises(ValidationError) as exc_info:
        TableExtractorConfigSchema(
            auth_token="valid_token", paddle_endpoints=("grpc://paddle_service", None), extra_field="invalid"
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_cleaning_empty_strings_in_endpoints():
    config = TableExtractorConfigSchema(paddle_endpoints=("   ", "http://paddle_service"))
    assert config.paddle_endpoints == (None, "http://paddle_service")

    config = TableExtractorConfigSchema(paddle_endpoints=("grpc://paddle_service", ""))
    assert config.paddle_endpoints == ("grpc://paddle_service", None)


def test_auth_token_is_none_by_default():
    config = TableExtractorConfigSchema(paddle_endpoints=("grpc://paddle_service", "http://paddle_service"))
    assert config.auth_token is None


# Test cases for TableExtractorSchema
def test_table_extractor_schema_defaults():
    config = TableExtractorSchema()
    assert config.max_queue_size == 1
    assert config.n_workers == 2
    assert config.raise_on_failure is False
    assert config.stage_config is None


def test_table_extractor_schema_with_custom_values():
    stage_config = TableExtractorConfigSchema(paddle_endpoints=("grpc://paddle_service", "http://paddle_service"))
    config = TableExtractorSchema(max_queue_size=15, n_workers=12, raise_on_failure=True, stage_config=stage_config)
    assert config.max_queue_size == 15
    assert config.n_workers == 12
    assert config.raise_on_failure is True
    assert config.stage_config == stage_config


def test_table_extractor_schema_without_stage_config():
    config = TableExtractorSchema(max_queue_size=20, n_workers=5, raise_on_failure=True)
    assert config.max_queue_size == 20
    assert config.n_workers == 5
    assert config.raise_on_failure is True
    assert config.stage_config is None


def test_invalid_table_extractor_schema_negative_queue_size():
    with pytest.raises(ValidationError):
        TableExtractorSchema(max_queue_size=-5)


def test_invalid_table_extractor_schema_zero_workers():
    with pytest.raises(ValidationError):
        TableExtractorSchema(n_workers=0)


def test_invalid_extra_fields_in_table_extractor_schema():
    with pytest.raises(ValidationError):
        TableExtractorSchema(max_queue_size=10, n_workers=5, extra_field="invalid")
