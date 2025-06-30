# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorConfigSchema, ChartExtractorSchema


### Tests for ChartExtractorConfigSchema ###


def test_valid_yolox_only():
    config = ChartExtractorConfigSchema(
        yolox_endpoints=("grpc_service", None), ocr_endpoints=("grpc_ocr", "http_ocr")
    )
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
        max_queue_size=20, n_workers=15, raise_on_failure=True, endpoint_config=endpoint_config
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
