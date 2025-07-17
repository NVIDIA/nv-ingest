# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import (
    InfographicExtractorConfigSchema,
    InfographicExtractorSchema,
)


### Tests for InfographicExtractorConfigSchema ###


def test_valid_ocr_grpc_only():
    config = InfographicExtractorConfigSchema(ocr_endpoints=("grpc_service", None))
    assert config.ocr_endpoints == ("grpc_service", None)


def test_valid_ocr_http_only():
    config = InfographicExtractorConfigSchema(ocr_endpoints=(None, "http_service"))
    assert config.ocr_endpoints == (None, "http_service")


def test_valid_ocr_both():
    config = InfographicExtractorConfigSchema(ocr_endpoints=("grpc_service", "http_service"))
    assert config.ocr_endpoints == ("grpc_service", "http_service")


def test_invalid_ocr_both_empty():
    with pytest.raises(ValidationError) as excinfo:
        InfographicExtractorConfigSchema(ocr_endpoints=(None, None))
    assert "Both gRPC and HTTP services cannot be empty for ocr_endpoints." in str(excinfo.value)


def test_cleaning_ocr_endpoints_spaces_and_quotes():
    with pytest.raises(ValidationError) as excinfo:
        InfographicExtractorConfigSchema(ocr_endpoints=("  ", '  "  '))
    assert "Both gRPC and HTTP services cannot be empty for ocr_endpoints." in str(excinfo.value)


def test_extra_fields_forbidden_in_infographic_config():
    with pytest.raises(ValidationError):
        InfographicExtractorConfigSchema(ocr_endpoints=("grpc_service", None), extra_field="fail")


### Tests for InfographicExtractorSchema ###


def test_infographic_extractor_schema_defaults():
    schema = InfographicExtractorSchema()
    assert schema.max_queue_size == 1
    assert schema.n_workers == 2
    assert schema.raise_on_failure is False
    assert schema.endpoint_config is None


def test_infographic_extractor_with_config():
    config = InfographicExtractorConfigSchema(ocr_endpoints=("grpc_service", None))
    schema = InfographicExtractorSchema(max_queue_size=20, n_workers=15, raise_on_failure=True, endpoint_config=config)
    assert schema.max_queue_size == 20
    assert schema.n_workers == 15
    assert schema.raise_on_failure is True
    assert schema.endpoint_config == config


def test_infographic_extractor_forbid_extra_fields():
    with pytest.raises(ValidationError):
        InfographicExtractorSchema(max_queue_size=10, n_workers=5, invalid_field="fail")


def test_infographic_extractor_invalid_low_values():
    with pytest.raises(ValidationError) as excinfo:
        InfographicExtractorSchema(max_queue_size=0, n_workers=15)
    assert "must be greater than 0" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        InfographicExtractorSchema(max_queue_size=-1, n_workers=10)
    assert "must be greater than 0" in str(excinfo.value)


def test_infographic_extractor_invalid_types():
    with pytest.raises(ValidationError):
        InfographicExtractorSchema(max_queue_size="not_int", n_workers="also_not_int")
