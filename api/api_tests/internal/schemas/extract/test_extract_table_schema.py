# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorConfigSchema, TableExtractorSchema


### Tests for TableExtractorConfigSchema ###


def test_table_valid_yolox_and_ocr_grpc_only():
    config = TableExtractorConfigSchema(yolox_endpoints=("grpc_yolox", None), ocr_endpoints=("grpc_ocr", None))
    assert config.yolox_endpoints == ("grpc_yolox", None)
    assert config.ocr_endpoints == ("grpc_ocr", None)


def test_table_valid_yolox_and_ocr_http_only():
    config = TableExtractorConfigSchema(yolox_endpoints=(None, "http_yolox"), ocr_endpoints=(None, "http_ocr"))
    assert config.yolox_endpoints == (None, "http_yolox")
    assert config.ocr_endpoints == (None, "http_ocr")


def test_table_invalid_yolox_both_empty():
    with pytest.raises(ValidationError) as excinfo:
        TableExtractorConfigSchema(yolox_endpoints=(None, None), ocr_endpoints=("grpc_ocr", None))
    assert "Both gRPC and HTTP services cannot be empty for yolox_endpoints." in str(excinfo.value)


def test_table_invalid_ocr_both_empty():
    with pytest.raises(ValidationError) as excinfo:
        TableExtractorConfigSchema(yolox_endpoints=("grpc_yolox", None), ocr_endpoints=("  ", '  "  '))
    assert "Both gRPC and HTTP services cannot be empty for ocr_endpoints." in str(excinfo.value)


def test_table_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        TableExtractorConfigSchema(
            yolox_endpoints=("grpc_service", None), ocr_endpoints=("grpc_ocr", None), extra_field="fail"
        )


### Tests for TableExtractorSchema ###


def test_table_extractor_schema_defaults():
    schema = TableExtractorSchema()
    assert schema.max_queue_size == 1
    assert schema.n_workers == 2
    assert schema.raise_on_failure is False
    assert schema.endpoint_config is None


def test_table_extractor_with_config():
    config = TableExtractorConfigSchema(yolox_endpoints=("grpc_yolox", None), ocr_endpoints=("grpc_ocr", None))
    schema = TableExtractorSchema(max_queue_size=20, n_workers=15, raise_on_failure=True, endpoint_config=config)
    assert schema.max_queue_size == 20
    assert schema.n_workers == 15
    assert schema.raise_on_failure is True
    assert schema.endpoint_config == config


def test_table_extractor_forbid_extra_fields():
    with pytest.raises(ValidationError):
        TableExtractorSchema(max_queue_size=10, n_workers=5, invalid_field="fail")


def test_table_extractor_invalid_low_values():
    with pytest.raises(ValidationError) as excinfo:
        TableExtractorSchema(max_queue_size=0, n_workers=15)
    assert "must be greater than 0" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        TableExtractorSchema(max_queue_size=-1, n_workers=10)
    assert "must be greater than 0" in str(excinfo.value)


def test_table_extractor_invalid_types():
    with pytest.raises(ValidationError):
        TableExtractorSchema(max_queue_size="not_int", n_workers="also_not_int")
