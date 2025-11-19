# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorConfigSchema, TableExtractorSchema


# Test cases for TableExtractorConfigSchema
def test_valid_config_with_grpc_only():
    config = TableExtractorConfigSchema(
        auth_token="valid_token",
        yolox_endpoints=("grpc://yolox_service", None),
        ocr_endpoints=("grpc://ocr_service", None),
    )
    assert config.auth_token == "valid_token"
    assert config.yolox_endpoints == ("grpc://yolox_service", None)
    assert config.ocr_endpoints == ("grpc://ocr_service", None)


def test_valid_config_with_http_only():
    config = TableExtractorConfigSchema(
        auth_token="valid_token",
        yolox_endpoints=(None, "http://yolox_service"),
        ocr_endpoints=(None, "http://ocr_service"),
    )
    assert config.auth_token == "valid_token"
    assert config.yolox_endpoints == (None, "http://yolox_service")
    assert config.ocr_endpoints == (None, "http://ocr_service")


def test_valid_config_with_both_services():
    config = TableExtractorConfigSchema(
        auth_token="valid_token",
        yolox_endpoints=("grpc://yolox_service", "http://yolox_service"),
        ocr_endpoints=("grpc://ocr_service", "http://ocr_service"),
    )
    assert config.auth_token == "valid_token"
    assert config.yolox_endpoints == ("grpc://yolox_service", "http://yolox_service")
    assert config.ocr_endpoints == ("grpc://ocr_service", "http://ocr_service")


def test_invalid_config_empty_endpoints():
    with pytest.raises(ValidationError) as exc_info:
        TableExtractorConfigSchema(
            yolox_endpoints=("grpc://yolox_service", "http://yolox_service"),
            ocr_endpoints=(None, None),
        )
    assert "Both gRPC and HTTP services cannot be empty for ocr_endpoints" in str(exc_info.value)


def test_invalid_extra_fields():
    with pytest.raises(ValidationError) as exc_info:
        TableExtractorConfigSchema(
            auth_token="valid_token",
            yolox_endpoints=("grpc://yolox_service", None),
            ocr_endpoints=("grpc://ocr_service", None),
            extra_field="invalid",
        )
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_cleaning_empty_strings_in_endpoints():
    config = TableExtractorConfigSchema(
        yolox_endpoints=("grpc://yolox_service", " "),
        ocr_endpoints=("   ", "http://ocr_service"),
    )
    assert config.yolox_endpoints == ("grpc://yolox_service", None)
    assert config.ocr_endpoints == (None, "http://ocr_service")

    config = TableExtractorConfigSchema(
        yolox_endpoints=("", "http://yolox_service"),
        ocr_endpoints=("grpc://ocr_service", ""),
    )
    assert config.yolox_endpoints == (None, "http://yolox_service")
    assert config.ocr_endpoints == ("grpc://ocr_service", None)


def test_auth_token_is_none_by_default():
    config = TableExtractorConfigSchema(
        yolox_endpoints=("grpc://yolox_service", "http://yolox_service"),
        ocr_endpoints=("grpc://ocr_service", "http://ocr_service"),
    )
    assert config.auth_token is None


# Test cases for TableExtractorSchema
def test_table_extractor_schema_defaults():
    config = TableExtractorSchema()
    assert config.max_queue_size == 1
    assert config.n_workers == 2
    assert config.raise_on_failure is False
    assert config.endpoint_config is None


def test_table_extractor_schema_with_custom_values():
    endpoint_config = TableExtractorConfigSchema(
        yolox_endpoints=("grpc://yolox_service", "http://yolox_service"),
        ocr_endpoints=("grpc://ocr_service", "http://ocr_service"),
    )
    config = TableExtractorSchema(
        max_queue_size=15, n_workers=12, raise_on_failure=True, endpoint_config=endpoint_config
    )
    assert config.max_queue_size == 15
    assert config.n_workers == 12
    assert config.raise_on_failure is True
    assert config.endpoint_config == endpoint_config


def test_table_extractor_schema_without_stage_config():
    config = TableExtractorSchema(max_queue_size=20, n_workers=5, raise_on_failure=True)
    assert config.max_queue_size == 20
    assert config.n_workers == 5
    assert config.raise_on_failure is True
    assert config.endpoint_config is None


def test_invalid_table_extractor_schema_negative_queue_size():
    with pytest.raises(ValidationError):
        TableExtractorSchema(max_queue_size=-5)


def test_invalid_table_extractor_schema_zero_workers():
    with pytest.raises(ValidationError):
        TableExtractorSchema(n_workers=0)


def test_invalid_extra_fields_in_table_extractor_schema():
    with pytest.raises(ValidationError):
        TableExtractorSchema(max_queue_size=10, n_workers=5, extra_field="invalid")
