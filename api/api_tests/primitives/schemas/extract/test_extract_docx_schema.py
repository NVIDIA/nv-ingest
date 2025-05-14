# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxConfigSchema, DocxExtractorSchema


### Tests for DocxConfigSchema ###


def test_valid_yolox_grpc_only():
    config = DocxConfigSchema(yolox_endpoints=("grpc_service", None))
    assert config.yolox_endpoints == ("grpc_service", None)
    assert config.yolox_infer_protocol == "grpc"


def test_valid_yolox_http_only():
    config = DocxConfigSchema(yolox_endpoints=(None, "http_service"))
    assert config.yolox_endpoints == (None, "http_service")
    assert config.yolox_infer_protocol == "http"


def test_valid_yolox_both_endpoints():
    config = DocxConfigSchema(yolox_endpoints=("grpc_service", "http_service"))
    assert config.yolox_endpoints == ("grpc_service", "http_service")
    assert config.yolox_infer_protocol == "http"  # Should default to http if both are provided


def test_invalid_yolox_both_empty():
    with pytest.raises(ValidationError) as excinfo:
        DocxConfigSchema(yolox_endpoints=(None, None))
    assert "Both gRPC and HTTP services cannot be empty for yolox_endpoints." in str(excinfo.value)


def test_cleaning_yolox_endpoints_spaces_and_quotes():
    with pytest.raises(ValidationError) as excinfo:
        DocxConfigSchema(yolox_endpoints=("  ", '  "  '))
    assert "Both gRPC and HTTP services cannot be empty for yolox_endpoints." in str(excinfo.value)


def test_custom_protocol_is_normalized():
    config = DocxConfigSchema(yolox_endpoints=("grpc_service", "http_service"), yolox_infer_protocol="GRPC")
    assert config.yolox_infer_protocol == "grpc"


def test_extra_fields_forbidden_in_docx_config():
    with pytest.raises(ValidationError):
        DocxConfigSchema(yolox_endpoints=("grpc_service", None), extra_field="fail")


### Tests for DocxExtractorSchema ###


def test_docx_extractor_schema_defaults():
    schema = DocxExtractorSchema()
    assert schema.max_queue_size == 1
    assert schema.n_workers == 16
    assert schema.raise_on_failure is False
    assert schema.docx_extraction_config is None


def test_docx_extractor_with_config():
    docx_config = DocxConfigSchema(yolox_endpoints=("grpc_service", None))
    schema = DocxExtractorSchema(
        max_queue_size=10, n_workers=5, raise_on_failure=True, docx_extraction_config=docx_config
    )
    assert schema.max_queue_size == 10
    assert schema.n_workers == 5
    assert schema.raise_on_failure is True
    assert schema.docx_extraction_config == docx_config


def test_docx_extractor_forbid_extra_fields():
    with pytest.raises(ValidationError):
        DocxExtractorSchema(max_queue_size=10, n_workers=5, invalid_field="fail")


def test_docx_extractor_invalid_types():
    with pytest.raises(ValidationError):
        DocxExtractorSchema(max_queue_size="not_int", n_workers="also_not_int")
