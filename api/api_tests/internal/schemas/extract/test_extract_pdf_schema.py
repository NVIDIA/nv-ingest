# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import (
    PDFiumConfigSchema,
    NemoRetrieverParseConfigSchema,
    PDFExtractorSchema,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging


### Tests for PDFiumConfigSchema ###


def test_pdfium_valid_yolox_grpc_only():
    config = PDFiumConfigSchema(yolox_endpoints=("grpc_service", None))
    assert config.yolox_endpoints == ("grpc_service", None)
    assert config.yolox_infer_protocol == "grpc"


def test_pdfium_valid_yolox_http_only():
    config = PDFiumConfigSchema(yolox_endpoints=(None, "http_service"))
    assert config.yolox_endpoints == (None, "http_service")
    assert config.yolox_infer_protocol == "http"


def test_pdfium_invalid_yolox_both_empty():
    with pytest.raises(ValidationError) as excinfo:
        PDFiumConfigSchema(yolox_endpoints=(None, None))
    assert "Both gRPC and HTTP services cannot be empty for yolox_endpoints." in str(excinfo.value)


def test_pdfium_cleaning_yolox_endpoints_spaces_and_quotes():
    with pytest.raises(ValidationError):
        PDFiumConfigSchema(yolox_endpoints=("  ", '  "  '))


def test_pdfium_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        PDFiumConfigSchema(yolox_endpoints=("grpc_service", None), extra_field="fail")


### Tests for NemoRetrieverParseConfigSchema ###


def test_nemo_valid_parse_grpc_only():
    config = NemoRetrieverParseConfigSchema(nemoretriever_parse_endpoints=("grpc_service", None))
    assert config.nemoretriever_parse_endpoints == ("grpc_service", None)
    assert config.nemoretriever_parse_infer_protocol == "grpc"


def test_nemo_valid_parse_http_only():
    config = NemoRetrieverParseConfigSchema(nemoretriever_parse_endpoints=(None, "http_service"))
    assert config.nemoretriever_parse_endpoints == (None, "http_service")
    assert config.nemoretriever_parse_infer_protocol == "http"


def test_nemo_invalid_parse_both_empty():
    with pytest.raises(ValidationError) as excinfo:
        NemoRetrieverParseConfigSchema(nemoretriever_parse_endpoints=(None, None))
    assert "Both gRPC and HTTP services cannot be empty for nemoretriever_parse_endpoints." in str(excinfo.value)


def test_nemo_cleaning_parse_endpoints_spaces_and_quotes():
    with pytest.raises(ValidationError):
        NemoRetrieverParseConfigSchema(nemoretriever_parse_endpoints=("  ", '  "  '))


def test_nemo_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        NemoRetrieverParseConfigSchema(nemoretriever_parse_endpoints=("grpc_service", None), extra_field="fail")


### Tests for PDFExtractorSchema ###


def test_pdf_extractor_schema_defaults():
    schema = PDFExtractorSchema()
    assert schema.max_queue_size == 1
    assert schema.n_workers == 16
    assert schema.raise_on_failure is False
    assert schema.pdfium_config is None
    assert schema.nemoretriever_parse_config is None


def test_pdf_extractor_with_configs():
    pdfium_config = PDFiumConfigSchema(yolox_endpoints=("grpc_service", None))
    nemo_config = NemoRetrieverParseConfigSchema(nemoretriever_parse_endpoints=("grpc_service", None))
    schema = PDFExtractorSchema(
        max_queue_size=10,
        n_workers=8,
        raise_on_failure=True,
        pdfium_config=pdfium_config,
        nemoretriever_parse_config=nemo_config,
    )
    assert schema.max_queue_size == 10
    assert schema.n_workers == 8
    assert schema.raise_on_failure is True
    assert schema.pdfium_config == pdfium_config
    assert schema.nemoretriever_parse_config == nemo_config


def test_pdf_extractor_forbid_extra_fields():
    with pytest.raises(ValidationError):
        PDFExtractorSchema(max_queue_size=10, n_workers=8, invalid_field="fail")


def test_pdf_extractor_invalid_types():
    with pytest.raises(ValidationError):
        PDFExtractorSchema(max_queue_size="wrong_type", n_workers="also_wrong")


### Sanitization and Redaction Tests ###


def test_pdfium_config_repr_hides_sensitive_fields_and_sanitize_redacts():
    cfg = PDFiumConfigSchema(
        yolox_endpoints=("grpc_service", None),
        auth_token="pdfium_secret",
    )

    rep = repr(cfg)
    s = str(cfg)
    assert "pdfium_secret" not in rep
    assert "pdfium_secret" not in s

    sanitized = sanitize_for_logging(cfg)
    assert isinstance(sanitized, dict)
    assert sanitized.get("auth_token") == "***REDACTED***"


def test_nemo_config_repr_hides_sensitive_fields_and_sanitize_redacts():
    cfg = NemoRetrieverParseConfigSchema(
        nemoretriever_parse_endpoints=("grpc_service", None),
        auth_token="nemo_secret",
    )

    rep = repr(cfg)
    s = str(cfg)
    assert "nemo_secret" not in rep
    assert "nemo_secret" not in s

    sanitized = sanitize_for_logging(cfg)
    assert isinstance(sanitized, dict)
    assert sanitized.get("auth_token") == "***REDACTED***"


def test_pdf_extractor_schema_sanitize_nested_configs():
    pdfium_cfg = PDFiumConfigSchema(yolox_endpoints=("grpc_service", None), auth_token="tok1")
    nemo_cfg = NemoRetrieverParseConfigSchema(nemoretriever_parse_endpoints=("grpc_service", None), auth_token="tok2")
    schema = PDFExtractorSchema(
        max_queue_size=5,
        n_workers=3,
        raise_on_failure=False,
        pdfium_config=pdfium_cfg,
        nemoretriever_parse_config=nemo_cfg,
    )

    sanitized = sanitize_for_logging(schema)
    assert isinstance(sanitized, dict)
    assert sanitized["pdfium_config"]["auth_token"] == "***REDACTED***"
    assert sanitized["nemoretriever_parse_config"]["auth_token"] == "***REDACTED***"
