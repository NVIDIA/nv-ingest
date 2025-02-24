# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import Any, Dict, Optional, List, Tuple

from pydantic import BaseModel, ValidationError

from nv_ingest_api.interface import CONFIG_SCHEMAS, extraction_interface_relay_constructor, build_config_from_schema


# For testing, define two dummy Pydantic schemas.
class DummyPDFiumConfigSchema(BaseModel):
    yolox_auth_token: str
    yolox_endpoints: Tuple[Optional[str], Optional[str]]
    yolox_infer_protocol: str


class DummyTikaConfigSchema(BaseModel):
    tika_server_url: str


# Clear and register schemas.
CONFIG_SCHEMAS.clear()
CONFIG_SCHEMAS["pdfium"] = DummyPDFiumConfigSchema
CONFIG_SCHEMAS["tika"] = DummyTikaConfigSchema


# Create a dummy backend API function.
def mock_backend_api(
    ledger: Any, task_config: Dict[str, Any], extractor_config: Any, execution_trace_log: Optional[List[Any]] = None
) -> Tuple[Any, Any, Any, Any]:
    """
    Dummy backend API function that returns its input arguments.
    """
    return ledger, task_config, extractor_config, execution_trace_log


# Create a dummy user-facing interface function using the decorator.
@extraction_interface_relay_constructor(api_fn=mock_backend_api)
def dummy_extract_primitives_default(
    *,
    df_extraction_ledger: Any,
    extract_method: str,
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    extract_charts: bool,
    yolox_auth_token: Optional[str] = None,
    yolox_endpoints: Optional[Tuple[Optional[str], Optional[str]]] = None,
    yolox_infer_protocol: str = "http",
    tika_server_url: Optional[str] = None,
    execution_trace_log: Optional[List[Any]] = None,
):
    pass


# Create another dummy function using a custom list of task_keys.
custom_task_keys = ["custom_flag", "another_flag"]


@extraction_interface_relay_constructor(api_fn=mock_backend_api, task_keys=custom_task_keys)
def dummy_extract_primitives_custom(
    *,
    df_extraction_ledger: Any,
    extract_method: str,
    custom_flag: bool,
    another_flag: bool,
    yolox_auth_token: Optional[str] = None,
    yolox_endpoints: Optional[Tuple[Optional[str], Optional[str]]] = None,
    yolox_infer_protocol: str = "http",
    tika_server_url: Optional[str] = None,
    execution_trace_log: Optional[List[Any]] = None,
):
    pass


# -------------- Unit Tests ----------------


def test_build_config_from_schema_valid():
    args = {
        "yolox_auth_token": "secret",
        "yolox_endpoints": ("grpc_ep", "http_ep"),
        "yolox_infer_protocol": "https",
        "irrelevant": "ignore me",
    }
    result = build_config_from_schema(DummyPDFiumConfigSchema, args)
    expected = {
        "yolox_auth_token": "secret",
        "yolox_endpoints": ("grpc_ep", "http_ep"),
        "yolox_infer_protocol": "https",
    }
    assert result == expected


def test_build_config_from_schema_missing_required():
    args = {"yolox_endpoints": ("grpc_ep", "http_ep"), "yolox_infer_protocol": "https"}
    with pytest.raises(ValidationError):
        build_config_from_schema(DummyPDFiumConfigSchema, args)


def test_interface_default_task_keys_pdfium():
    ledger = {"df": "dummy ledger"}
    ret = dummy_extract_primitives_default(
        df_extraction_ledger=ledger,
        extract_method="pdfium",
        extract_text=True,
        extract_images=False,
        extract_tables=True,
        extract_charts=False,
        yolox_auth_token="secret_token",
        yolox_endpoints=("grpc_ep", "http_ep"),
        yolox_infer_protocol="https",
        execution_trace_log=None,
    )
    ret_ledger, task_config, extractor_config, trace_log = ret
    assert ret_ledger == ledger
    # Default task_keys should have been used.
    expected_task_config = {
        "params": {
            "extract_text": True,
            "extract_images": False,
            "extract_tables": True,
            "extract_charts": False,
        }
    }
    assert task_config == expected_task_config
    expected_extractor_config = {
        "pdfium_config": {
            "yolox_auth_token": "secret_token",
            "yolox_endpoints": ("grpc_ep", "http_ep"),
            "yolox_infer_protocol": "https",
        }
    }
    assert extractor_config == expected_extractor_config
    # execution_trace_log was None so should be replaced with {}.
    assert trace_log == {}


def test_interface_default_task_keys_tika():
    ledger = {"df": "dummy ledger"}
    ret = dummy_extract_primitives_default(
        df_extraction_ledger=ledger,
        extract_method="tika",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=True,
        tika_server_url="http://tika.server",
        execution_trace_log=["trace1"],
    )
    ret_ledger, task_config, extractor_config, trace_log = ret
    assert ret_ledger == ledger
    expected_task_config = {
        "params": {
            "extract_text": False,
            "extract_images": True,
            "extract_tables": False,
            "extract_charts": True,
        }
    }
    assert task_config == expected_task_config
    expected_extractor_config = {"tika_config": {"tika_server_url": "http://tika.server"}}
    assert extractor_config == expected_extractor_config
    assert trace_log == ["trace1"]


def test_interface_custom_task_keys():
    ledger = {"df": "dummy ledger"}
    ret = dummy_extract_primitives_custom(
        df_extraction_ledger=ledger,
        extract_method="pdfium",
        custom_flag=True,
        another_flag=False,
        yolox_auth_token="secret_token",
        yolox_endpoints=("grpc_ep", "http_ep"),
        yolox_infer_protocol="https",
        execution_trace_log=None,
    )
    ret_ledger, task_config, extractor_config, trace_log = ret
    assert ret_ledger == ledger
    # Custom task_keys should be used.
    expected_task_config = {
        "params": {
            "custom_flag": True,
            "another_flag": False,
        }
    }
    assert task_config == expected_task_config
    expected_extractor_config = {
        "pdfium_config": {
            "yolox_auth_token": "secret_token",
            "yolox_endpoints": ("grpc_ep", "http_ep"),
            "yolox_infer_protocol": "https",
        }
    }
    assert extractor_config == expected_extractor_config
    assert trace_log == {}


def test_missing_extract_method():
    ledger = {"df": "dummy ledger"}
    with pytest.raises(ValueError, match="The 'extract_method' parameter is required."):
        dummy_extract_primitives_default(
            df_extraction_ledger=ledger,
            # extract_method omitted intentionally
            extract_text=True,
            extract_images=True,
            extract_tables=True,
            extract_charts=True,
            yolox_auth_token="secret_token",
            yolox_endpoints=("grpc_ep", "http_ep"),
            yolox_infer_protocol="https",
            execution_trace_log=None,
        )


def test_unsupported_extract_method():
    ledger = {"df": "dummy ledger"}
    with pytest.raises(ValueError, match="Unsupported extraction method: unsupported"):
        dummy_extract_primitives_default(
            df_extraction_ledger=ledger,
            extract_method="unsupported",
            extract_text=True,
            extract_images=True,
            extract_tables=True,
            extract_charts=True,
            yolox_auth_token="secret_token",
            yolox_endpoints=("grpc_ep", "http_ep"),
            yolox_infer_protocol="https",
            execution_trace_log=None,
        )
