# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.interface import _build_config_from_schema, extraction_interface_relay_constructor
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFiumConfigSchema


@pytest.fixture
def valid_pdfium_args():
    return {
        "workers_per_progress_engine": 2,
        "yolox_endpoints": ("grpc_service", ""),
    }


# --- Tests for _build_config_from_schema ---
def test_build_config_from_schema_happy_path():
    args = {"workers_per_progress_engine": 2, "yolox_endpoints": ("grpc", "http")}
    result = _build_config_from_schema(PDFiumConfigSchema, args)
    assert result["workers_per_progress_engine"] == 2
    assert result["yolox_endpoints"] == ("grpc", "http")


def test_build_config_from_schema_extra_args_ignored():
    args = {"workers_per_progress_engine": 2, "yolox_endpoints": ("grpc", "http"), "unexpected": "value"}
    result = _build_config_from_schema(PDFiumConfigSchema, args)
    assert "unexpected" not in result
    assert result["workers_per_progress_engine"] == 2


def test_build_config_from_schema_invalid_args_raises():
    args = {"workers_per_progress_engine": "invalid"}
    with pytest.raises(ValidationError):
        _build_config_from_schema(PDFiumConfigSchema, args)


# --- Tests for extraction_interface_relay_constructor ---


def dummy_backend(ledger, task_config, extractor_config, execution_trace_log):
    return ledger, task_config, extractor_config, execution_trace_log


@extraction_interface_relay_constructor(dummy_backend, task_keys=["extract_text"])
def user_function(ledger, extract_method, extract_text, workers_per_progress_engine):
    pass


def test_extraction_interface_missing_extract_method_raises():
    with pytest.raises(ValueError, match="extract_method"):
        user_function("ledger", extract_text=True, workers_per_progress_engine=2)


def test_extraction_interface_unsupported_method_raises():
    with pytest.raises(ValueError, match="Unsupported extraction method"):
        user_function(
            "ledger",
            extract_method="unsupported",
            extract_text=True,
            workers_per_progress_engine=2,
        )


def test_extraction_interface_invalid_backend_signature_raises():
    def bad_backend(a, b):
        pass

    with pytest.raises(ValueError):
        extraction_interface_relay_constructor(bad_backend)
