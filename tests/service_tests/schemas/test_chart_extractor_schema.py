# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorConfigSchema, ChartExtractorSchema


# Test cases for ChartExtractorConfigSchema
def test_valid_config_with_grpc_only():
    config = ChartExtractorConfigSchema(
        auth_token="valid_token",
        yolox_endpoints=("grpc://yolox_service", None),
        ocr_endpoints=("grpc://ocr_service", None),
    )
    assert config.auth_token == "valid_token"
    assert config.yolox_endpoints == ("grpc://yolox_service", None)
    assert config.ocr_endpoints == ("grpc://ocr_service", None)


def test_valid_config_with_http_only():
    config = ChartExtractorConfigSchema(
        auth_token="valid_token",
        yolox_endpoints=(None, "http://yolox_service"),
        ocr_endpoints=(None, "http://ocr_service"),
    )
    assert config.auth_token == "valid_token"
    assert config.yolox_endpoints == (None, "http://yolox_service")
    assert config.ocr_endpoints == (None, "http://ocr_service")


def test_invalid_config_with_empty_services():
    config = ChartExtractorConfigSchema(yolox_endpoints=(None, None), ocr_endpoints=(None, None))
    assert config.yolox_infer_protocol == "local"
    assert config.ocr_infer_protocol == "local"


def test_valid_config_with_both_grpc_and_http():
    config = ChartExtractorConfigSchema(
        auth_token="another_token",
        yolox_endpoints=("grpc://yolox_service", "http://yolox_service"),
        ocr_endpoints=("grpc://ocr_service", "http://ocr_service"),
    )
    assert config.auth_token == "another_token"
    assert config.yolox_endpoints == ("grpc://yolox_service", "http://yolox_service")
    assert config.ocr_endpoints == ("grpc://ocr_service", "http://ocr_service")


def test_invalid_auth_token_none():
    config = ChartExtractorConfigSchema(
        yolox_endpoints=("grpc://yolox_service", None),
        ocr_endpoints=("grpc://ocr_service", None),
    )
    assert config.auth_token is None


def test_invalid_endpoint_format():
    with pytest.raises(ValidationError):
        ChartExtractorConfigSchema(
            yolox_endpoints=("invalid_endpoint", None),
            deplot_endpoints=(None, "invalid_endpoint"),
        )


# Test cases for ChartExtractorSchema
def test_chart_extractor_schema_defaults():
    config = ChartExtractorSchema()
    assert config.max_queue_size == 1
    assert config.n_workers == 2
    assert config.raise_on_failure is False


def test_chart_extractor_schema_with_custom_values():
    stage_config = ChartExtractorConfigSchema(
        yolox_endpoints=("grpc://yolox_service", "http://yolox_service"),
        ocr_endpoints=(None, "http://ocr_service"),
    )
    config = ChartExtractorSchema(max_queue_size=10, n_workers=5, raise_on_failure=True)
    assert config.max_queue_size == 10
    assert config.n_workers == 5
    assert config.raise_on_failure is True


def test_chart_extractor_schema_without_stage_config():
    config = ChartExtractorSchema(max_queue_size=3, n_workers=1, raise_on_failure=False)
    assert config.max_queue_size == 3
    assert config.n_workers == 1
    assert config.raise_on_failure is False


def test_invalid_chart_extractor_schema_negative_queue_size():
    with pytest.raises(ValidationError):
        ChartExtractorSchema(max_queue_size=-1)


def test_invalid_chart_extractor_schema_zero_workers():
    with pytest.raises(ValidationError):
        ChartExtractorSchema(n_workers=0)
