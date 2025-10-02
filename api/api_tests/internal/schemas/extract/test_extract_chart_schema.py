# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_chart_schema import (
    ChartExtractorConfigSchema,
    ChartExtractorSchema,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging


### Tests for ChartExtractorConfigSchema ###


def test_valid_yolox_only():
    config = ChartExtractorConfigSchema(yolox_endpoints=("grpc_service", None), ocr_endpoints=("grpc_ocr", "http_ocr"))
    assert config.yolox_endpoints == ("grpc_service", None)
    assert config.ocr_endpoints == ("grpc_ocr", "http_ocr")


def test_valid_ocr_only():
    config = ChartExtractorConfigSchema(yolox_endpoints=("grpc_service", None), ocr_endpoints=(None, "http_ocr"))
    assert config.ocr_endpoints == (None, "http_ocr")


def test_both_endpoints_provided():
    config = ChartExtractorConfigSchema(
        yolox_endpoints=("grpc_service", "http_service"), ocr_endpoints=("grpc_ocr", "http_ocr")
    )
    assert config.yolox_endpoints == ("grpc_service", "http_service")
    assert config.ocr_endpoints == ("grpc_ocr", "http_ocr")


def test_invalid_yolox_empty():
    with pytest.raises(ValidationError) as excinfo:
        ChartExtractorConfigSchema(yolox_endpoints=(None, None), ocr_endpoints=("grpc_ocr", None))
    assert "Both gRPC and HTTP services cannot be empty for yolox_endpoints." in str(excinfo.value)


def test_invalid_ocr_empty():
    with pytest.raises(ValidationError) as excinfo:
        ChartExtractorConfigSchema(yolox_endpoints=("grpc_service", None), ocr_endpoints=("  ", '   "  '))
    assert "Both gRPC and HTTP services cannot be empty for ocr_endpoints." in str(excinfo.value)


def test_extra_fields_forbidden_in_chart_extractor_config():
    with pytest.raises(ValidationError):
        ChartExtractorConfigSchema(
            yolox_endpoints=("grpc_service", None), ocr_endpoints=("grpc_ocr", None), extra_field="should_fail"
        )


### Tests for ChartExtractorSchema ###


def test_valid_extractor_defaults():
    schema = ChartExtractorSchema()
    assert schema.max_queue_size == 1
    assert schema.n_workers == 2
    assert schema.raise_on_failure is False
    assert schema.endpoint_config is None


def test_valid_extractor_with_config():
    endpoint_config = ChartExtractorConfigSchema(
        yolox_endpoints=("grpc_service", None), ocr_endpoints=("grpc_ocr", None)
    )
    schema = ChartExtractorSchema(
        max_queue_size=20,
        n_workers=15,
        raise_on_failure=True,
        endpoint_config=endpoint_config,
    )
    assert schema.max_queue_size == 20
    assert schema.n_workers == 15
    assert schema.raise_on_failure is True
    assert schema.endpoint_config == endpoint_config


def test_invalid_extractor_negative_values():
    with pytest.raises(ValidationError) as excinfo:
        ChartExtractorSchema(max_queue_size=0, n_workers=-5)
    assert "must be greater than 0" in str(excinfo.value)


def test_extra_fields_forbidden_in_extractor_schema():
    with pytest.raises(ValidationError):
        ChartExtractorSchema(max_queue_size=10, n_workers=10, invalid_field="oops")


def test_extractor_rejects_low_values_correctly():
    with pytest.raises(ValidationError):
        ChartExtractorSchema(max_queue_size=0, n_workers=15)
    with pytest.raises(ValidationError):
        ChartExtractorSchema(max_queue_size=-1, n_workers=10)


### Sanitization and Redaction Tests ###


def test_chart_config_repr_hides_sensitive_fields_and_sanitize_redacts():
    cfg = ChartExtractorConfigSchema(
        yolox_endpoints=("grpc_service", None),
        ocr_endpoints=("grpc_ocr", None),
        auth_token="chart_secret",
    )

    rep = repr(cfg)
    s = str(cfg)
    assert "chart_secret" not in rep
    assert "chart_secret" not in s

    sanitized = sanitize_for_logging(cfg)
    assert isinstance(sanitized, dict)
    assert sanitized.get("auth_token") == "***REDACTED***"


def test_chart_extractor_schema_sanitize_nested_config():
    cfg = ChartExtractorConfigSchema(
        yolox_endpoints=("grpc_service", None),
        ocr_endpoints=("grpc_ocr", None),
        auth_token="nested_chart_secret",
    )
    schema = ChartExtractorSchema(
        max_queue_size=3,
        n_workers=2,
        raise_on_failure=False,
        endpoint_config=cfg,
    )

    sanitized = sanitize_for_logging(schema)
    assert isinstance(sanitized, dict)
    assert sanitized["endpoint_config"]["auth_token"] == "***REDACTED***"
