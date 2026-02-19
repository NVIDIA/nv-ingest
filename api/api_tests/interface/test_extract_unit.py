# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the public PDF extraction interface layer (no live services required).

These tests cover the kwarg-name contract between the public API functions and the
internal Pydantic schemas, specifically the auth_token rename that fixed the silent
token-drop bug (see: pdf-extract-fix-token-kwarg).

Root-cause recap
----------------
The public functions previously declared ``yolox_auth_token`` while the Pydantic
schemas (PDFiumConfigSchema, NemotronParseConfigSchema) defined the field as
``auth_token``.  The decorator ``extraction_interface_relay_constructor`` filters
kwargs to only those present in the schema's model_fields, so the misnamed kwarg
was silently dropped — the token never reached the backend.

What is tested here
-------------------
1. Schema-field sanity: both schemas expose ``auth_token``, not ``yolox_auth_token``.
2. _build_config_from_schema correctly passes ``auth_token`` through for each schema.
3. _build_config_from_schema silently drops the old ``yolox_auth_token`` key
   (documents why the bug was undetectable at the Pydantic level).
4. The public wrapper functions (pdfium, nemotron_parse) raise ``TypeError`` when
   called with the stale ``yolox_auth_token`` kwarg — regression guard.
5. The general ``extract_primitives_from_pdf`` raises ``TypeError`` for the same
   stale kwarg.

Import note
-----------
Groups 1-3 only require the schema package (no heavy PDF dependencies).
Groups 4-5 import ``nv_ingest_api.interface.extract`` which pulls in ``pypdfium2``;
``pypdfium2>=4.30.0`` is a declared core dependency in ``pyproject.toml`` so it is
always available when the package is installed normally.
"""

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Always-available imports (no pypdfium2 required)
# ---------------------------------------------------------------------------
from nv_ingest_api.interface import _build_config_from_schema
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import (
    NemotronParseConfigSchema,
    PDFiumConfigSchema,
)

from nv_ingest_api.interface.extract import (
    extract_primitives_from_pdf,
    extract_primitives_from_pdf_nemotron_parse,
    extract_primitives_from_pdf_pdfium,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YOLOX_HTTP = "http://localhost:8000/v1/infer"
_NEMOTRON_HTTP = "http://localhost:8015/v1/chat/completions"


# ===========================================================================
# 1. Schema-field sanity checks
# ===========================================================================


def test_pdfium_config_schema_has_auth_token_field():
    """PDFiumConfigSchema must declare 'auth_token', not 'yolox_auth_token'."""
    assert "auth_token" in PDFiumConfigSchema.model_fields
    assert "yolox_auth_token" not in PDFiumConfigSchema.model_fields


def test_nemotron_parse_config_schema_has_auth_token_field():
    """NemotronParseConfigSchema must declare 'auth_token', not 'yolox_auth_token'."""
    assert "auth_token" in NemotronParseConfigSchema.model_fields
    assert "yolox_auth_token" not in NemotronParseConfigSchema.model_fields


# ===========================================================================
# 2. _build_config_from_schema passes auth_token through
# ===========================================================================


def test_build_config_passes_auth_token_to_pdfium_schema():
    """auth_token survives _build_config_from_schema when using PDFiumConfigSchema."""
    args = {
        "auth_token": "pdfium_secret_token",
        "yolox_endpoints": (None, _YOLOX_HTTP),
    }
    result = _build_config_from_schema(PDFiumConfigSchema, args)
    assert result["auth_token"] == "pdfium_secret_token"


def test_build_config_passes_auth_token_to_nemotron_parse_schema():
    """auth_token survives _build_config_from_schema when using NemotronParseConfigSchema."""
    args = {
        "auth_token": "nemotron_secret_token",
        "nemotron_parse_endpoints": (None, _NEMOTRON_HTTP),
    }
    result = _build_config_from_schema(NemotronParseConfigSchema, args)
    assert result["auth_token"] == "nemotron_secret_token"


def test_build_config_passes_none_auth_token():
    """auth_token=None (the default) is preserved in the config dict."""
    args = {
        "auth_token": None,
        "yolox_endpoints": (None, _YOLOX_HTTP),
    }
    result = _build_config_from_schema(PDFiumConfigSchema, args)
    assert "auth_token" in result
    assert result["auth_token"] is None


# ===========================================================================
# 3. Silent-drop regression: yolox_auth_token is not a schema field
# ===========================================================================


def test_build_config_silently_drops_yolox_auth_token_for_pdfium():
    """
    Regression: the stale kwarg 'yolox_auth_token' is absent from PDFiumConfigSchema
    and is silently filtered out by _build_config_from_schema.

    This test documents the exact mechanism of the original bug: passing
    yolox_auth_token caused the token to vanish before reaching the backend
    because _build_config_from_schema retains only schema-defined keys.
    """
    args = {
        "yolox_auth_token": "should_be_dropped",
        "yolox_endpoints": (None, _YOLOX_HTTP),
    }
    result = _build_config_from_schema(PDFiumConfigSchema, args)
    assert "yolox_auth_token" not in result
    assert result.get("auth_token") is None


def test_build_config_silently_drops_yolox_auth_token_for_nemotron_parse():
    """
    Regression: the stale kwarg 'yolox_auth_token' is absent from
    NemotronParseConfigSchema and is silently filtered out.
    """
    args = {
        "yolox_auth_token": "should_be_dropped",
        "nemotron_parse_endpoints": (None, _NEMOTRON_HTTP),
    }
    result = _build_config_from_schema(NemotronParseConfigSchema, args)
    assert "yolox_auth_token" not in result
    assert result.get("auth_token") is None


# ===========================================================================
# 4. Public-API signature regression: yolox_auth_token must be rejected
# ===========================================================================


def test_extract_primitives_from_pdf_pdfium_rejects_yolox_auth_token():
    """
    Regression: extract_primitives_from_pdf_pdfium no longer accepts
    yolox_auth_token.  Python raises TypeError before the function body runs.
    """
    df = pd.DataFrame()
    with pytest.raises(TypeError, match="yolox_auth_token"):
        extract_primitives_from_pdf_pdfium(df, yolox_auth_token="stale_token")


def test_extract_primitives_from_pdf_nemotron_parse_rejects_yolox_auth_token():
    """
    Regression: extract_primitives_from_pdf_nemotron_parse no longer accepts
    yolox_auth_token.  Python raises TypeError before the function body runs.
    """
    df = pd.DataFrame()
    with pytest.raises(TypeError, match="yolox_auth_token"):
        extract_primitives_from_pdf_nemotron_parse(df, yolox_auth_token="stale_token")


def test_extract_primitives_from_pdf_rejects_yolox_auth_token():
    """
    Regression: the general extract_primitives_from_pdf no longer accepts
    yolox_auth_token.  The unified_exception_handler re-raises the TypeError
    produced by the decorator's inspect.Signature.bind_partial call.
    """
    df = pd.DataFrame()
    with pytest.raises(TypeError):
        extract_primitives_from_pdf(
            df_extraction_ledger=df,
            extract_method="pdfium",
            yolox_auth_token="stale_token",
        )
