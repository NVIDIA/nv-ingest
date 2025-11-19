# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXConfigSchema, PPTXExtractorSchema


### Tests for PPTXConfigSchema ###


def test_pptx_valid_yolox_grpc_only():
    config = PPTXConfigSchema(yolox_endpoints=("grpc_service", None))
    assert config.yolox_endpoints == ("grpc_service", None)
    assert config.yolox_infer_protocol == "grpc"


def test_pptx_valid_yolox_http_only():
    config = PPTXConfigSchema(yolox_endpoints=(None, "http_service"))
    assert config.yolox_endpoints == (None, "http_service")
    assert config.yolox_infer_protocol == "http"


def test_pptx_valid_yolox_both_endpoints():
    config = PPTXConfigSchema(yolox_endpoints=("grpc_service", "http_service"))
    assert config.yolox_endpoints == ("grpc_service", "http_service")
    assert config.yolox_infer_protocol == "http"  # Defaults to http if both are provided


def test_pptx_invalid_yolox_both_empty():
    with pytest.raises(ValidationError) as excinfo:
        PPTXConfigSchema(yolox_endpoints=(None, None))
    assert "Both gRPC and HTTP services cannot be empty for yolox_endpoints." in str(excinfo.value)


def test_pptx_cleaning_yolox_endpoints_spaces_and_quotes():
    with pytest.raises(ValidationError):
        PPTXConfigSchema(yolox_endpoints=("  ", '  "  '))


def test_pptx_custom_protocol_is_normalized():
    config = PPTXConfigSchema(yolox_endpoints=("grpc_service", "http_service"), yolox_infer_protocol="GRPC")
    assert config.yolox_infer_protocol == "grpc"


def test_pptx_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        PPTXConfigSchema(yolox_endpoints=("grpc_service", None), extra_field="fail")


### Tests for PPTXExtractorSchema ###


def test_pptx_extractor_schema_defaults():
    schema = PPTXExtractorSchema()
    assert schema.max_queue_size == 1
    assert schema.n_workers == 16
    assert schema.raise_on_failure is False
    assert schema.pptx_extraction_config is None


def test_pptx_extractor_with_config():
    pptx_config = PPTXConfigSchema(yolox_endpoints=("grpc_service", None))
    schema = PPTXExtractorSchema(
        max_queue_size=10, n_workers=5, raise_on_failure=True, pptx_extraction_config=pptx_config
    )
    assert schema.max_queue_size == 10
    assert schema.n_workers == 5
    assert schema.raise_on_failure is True
    assert schema.pptx_extraction_config == pptx_config


def test_pptx_extractor_forbid_extra_fields():
    with pytest.raises(ValidationError):
        PPTXExtractorSchema(max_queue_size=10, n_workers=5, invalid_field="fail")


def test_pptx_extractor_invalid_types():
    with pytest.raises(ValidationError):
        PPTXExtractorSchema(max_queue_size="not_int", n_workers="also_not_int")
