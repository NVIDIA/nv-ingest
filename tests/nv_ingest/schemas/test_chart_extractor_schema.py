import pytest
from pydantic import ValidationError

from nv_ingest.schemas.chart_extractor_schema import (
    ChartExtractorConfigSchema,
)  # Adjust the import as per your file structure
from nv_ingest.schemas.chart_extractor_schema import ChartExtractorSchema


# Test cases for ChartExtractorConfigSchema
def test_valid_config_with_grpc_only():
    config = ChartExtractorConfigSchema(
        auth_token="valid_token",
        cached_endpoints=("grpc://cached_service", None),
        deplot_endpoints=("grpc://deplot_service", None),
        paddle_endpoints=("grpc://paddle_service", None),
    )
    assert config.auth_token == "valid_token"
    assert config.cached_endpoints == ("grpc://cached_service", None)
    assert config.deplot_endpoints == ("grpc://deplot_service", None)
    assert config.paddle_endpoints == ("grpc://paddle_service", None)


def test_valid_config_with_http_only():
    config = ChartExtractorConfigSchema(
        auth_token="valid_token",
        cached_endpoints=(None, "http://cached_service"),
        deplot_endpoints=(None, "http://deplot_service"),
        paddle_endpoints=(None, "http://paddle_service"),
    )
    assert config.auth_token == "valid_token"
    assert config.cached_endpoints == (None, "http://cached_service")
    assert config.deplot_endpoints == (None, "http://deplot_service")
    assert config.paddle_endpoints == (None, "http://paddle_service")


def test_invalid_config_with_empty_services():
    with pytest.raises(ValidationError) as excinfo:
        ChartExtractorConfigSchema(
            cached_endpoints=(None, None), deplot_endpoints=(None, None), paddle_endpoints=(None, None)
        )
    assert "Both gRPC and HTTP services cannot be empty" in str(excinfo.value)


def test_valid_config_with_both_grpc_and_http():
    config = ChartExtractorConfigSchema(
        auth_token="another_token",
        cached_endpoints=("grpc://cached_service", "http://cached_service"),
        deplot_endpoints=("grpc://deplot_service", "http://deplot_service"),
        paddle_endpoints=("grpc://paddle_service", "http://paddle_service"),
    )
    assert config.auth_token == "another_token"
    assert config.cached_endpoints == ("grpc://cached_service", "http://cached_service")
    assert config.deplot_endpoints == ("grpc://deplot_service", "http://deplot_service")
    assert config.paddle_endpoints == ("grpc://paddle_service", "http://paddle_service")


def test_invalid_auth_token_none():
    config = ChartExtractorConfigSchema(
        cached_endpoints=("grpc://cached_service", None),
        deplot_endpoints=("grpc://deplot_service", None),
        paddle_endpoints=("grpc://paddle_service", None),
    )
    assert config.auth_token is None


def test_invalid_endpoint_format():
    with pytest.raises(ValidationError):
        ChartExtractorConfigSchema(
            cached_endpoints=("invalid_endpoint", None), deplot_endpoints=(None, "invalid_endpoint")
        )


# Test cases for ChartExtractorSchema
def test_chart_extractor_schema_defaults():
    config = ChartExtractorSchema()
    assert config.max_queue_size == 1
    assert config.n_workers == 2
    assert config.raise_on_failure is False
    assert config.stage_config is None


def test_chart_extractor_schema_with_custom_values():
    stage_config = ChartExtractorConfigSchema(
        cached_endpoints=("grpc://cached_service", "http://cached_service"),
        deplot_endpoints=("grpc://deplot_service", None),
        paddle_endpoints=(None, "http://paddle_service"),
    )
    config = ChartExtractorSchema(max_queue_size=10, n_workers=5, raise_on_failure=True, stage_config=stage_config)
    assert config.max_queue_size == 10
    assert config.n_workers == 5
    assert config.raise_on_failure is True
    assert config.stage_config == stage_config


def test_chart_extractor_schema_without_stage_config():
    config = ChartExtractorSchema(max_queue_size=3, n_workers=1, raise_on_failure=False)
    assert config.max_queue_size == 3
    assert config.n_workers == 1
    assert config.raise_on_failure is False
    assert config.stage_config is None


def test_invalid_chart_extractor_schema_negative_queue_size():
    with pytest.raises(ValidationError):
        ChartExtractorSchema(max_queue_size=-1)


def test_invalid_chart_extractor_schema_zero_workers():
    with pytest.raises(ValidationError):
        ChartExtractorSchema(n_workers=0)
