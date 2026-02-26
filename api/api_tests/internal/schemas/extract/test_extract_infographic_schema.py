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
    config = InfographicExtractorConfigSchema(ocr_endpoints=(None, None))
    assert config.ocr_endpoints == (None, None)
    assert config.ocr_infer_protocol == "local"


def test_cleaning_ocr_endpoints_spaces_and_quotes():
    config = InfographicExtractorConfigSchema(ocr_endpoints=("  ", '  "  '))
    assert config.ocr_endpoints == (None, None)
    assert config.ocr_infer_protocol == "local"


def test_extra_fields_forbidden_in_infographic_config():
    with pytest.raises(ValidationError):
        InfographicExtractorConfigSchema(ocr_endpoints=("grpc_service", None), extra_field="fail")


def test_protocol_case_insensitive():
    """Test that protocol values are normalized to lowercase for case-insensitive handling."""
    config = InfographicExtractorConfigSchema(
        ocr_endpoints=("grpc_ocr", "http_ocr"),
        ocr_infer_protocol="HTTP",  # uppercase
    )
    assert config.ocr_infer_protocol == "http"


def test_protocol_mixed_case():
    """Test that mixed case protocol values are normalized to lowercase."""
    config = InfographicExtractorConfigSchema(
        ocr_endpoints=("grpc_ocr", None),
        ocr_infer_protocol="GrPc",  # mixed case
    )
    assert config.ocr_infer_protocol == "grpc"


def test_protocol_auto_inference_from_endpoints():
    """Test that protocol is auto-inferred from endpoints when not specified."""
    config = InfographicExtractorConfigSchema(
        ocr_endpoints=(None, "http_service"),
    )
    assert config.ocr_infer_protocol == "http"

    config2 = InfographicExtractorConfigSchema(
        ocr_endpoints=("grpc_service", None),
    )
    assert config2.ocr_infer_protocol == "grpc"


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
