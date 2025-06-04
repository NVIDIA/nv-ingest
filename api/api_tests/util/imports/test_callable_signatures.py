# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import inspect
from pydantic import BaseModel

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.util.imports.callable_signatures import ingest_stage_callable_signature


# A simple BaseModel subclass for testing
class DummyConfig(BaseModel):
    foo: int = 0


# --- Helper functions with various signatures ---


def no_params() -> IngestControlMessage:
    """Zero-parameter function."""
    return IngestControlMessage()


def one_param(a: IngestControlMessage) -> IngestControlMessage:
    """Only one parameter."""
    return a


def missing_first_annotation(a, b: DummyConfig) -> IngestControlMessage:
    """First parameter lacks annotation."""
    return a


def missing_second_annotation(a: IngestControlMessage, b) -> IngestControlMessage:
    """Second parameter lacks annotation."""
    return a


def missing_return_annotation(a: IngestControlMessage, b: DummyConfig):
    """Missing return‐type annotation."""
    return a


def wrong_first_type(a: int, b: DummyConfig) -> IngestControlMessage:
    """First parameter annotated, but not IngestControlMessage."""
    return IngestControlMessage()


def wrong_second_type(a: IngestControlMessage, b: int) -> IngestControlMessage:
    """Second parameter annotated, but not a BaseModel subclass."""
    return a


def wrong_return_type(a: IngestControlMessage, b: DummyConfig) -> int:
    """Return annotation is not IngestControlMessage."""
    return 42


def extra_params(a: IngestControlMessage, b: DummyConfig, c: str) -> IngestControlMessage:
    """Three parameters instead of two."""
    return a


def valid_stage(a: IngestControlMessage, b: DummyConfig) -> IngestControlMessage:
    """Proper two‐parameter signature and return type."""
    return a


# --- Test Cases ---


def test_zero_parameters():
    sig = inspect.signature(no_params)
    with pytest.raises(TypeError, match="Expected exactly 2 parameters, got 0"):
        ingest_stage_callable_signature(sig)


def test_one_parameter():
    sig = inspect.signature(one_param)
    with pytest.raises(TypeError, match="Expected exactly 2 parameters, got 1"):
        ingest_stage_callable_signature(sig)


def test_extra_parameters():
    sig = inspect.signature(extra_params)
    with pytest.raises(TypeError, match="Expected exactly 2 parameters, got 3"):
        ingest_stage_callable_signature(sig)


def test_missing_first_annotation():
    sig = inspect.signature(missing_first_annotation)
    with pytest.raises(TypeError, match="First parameter must be annotated with IngestControlMessage"):
        ingest_stage_callable_signature(sig)


def test_missing_second_annotation():
    sig = inspect.signature(missing_second_annotation)
    with pytest.raises(TypeError, match="Second parameter must be annotated with a subclass of BaseModel"):
        ingest_stage_callable_signature(sig)


def test_missing_return_annotation():
    sig = inspect.signature(missing_return_annotation)
    with pytest.raises(TypeError, match="Return type must be annotated with IngestControlMessage"):
        ingest_stage_callable_signature(sig)


def test_wrong_first_type():
    sig = inspect.signature(wrong_first_type)
    with pytest.raises(TypeError, match=r"First parameter must be IngestControlMessage, got <class 'int'>"):
        ingest_stage_callable_signature(sig)


def test_wrong_second_type():
    sig = inspect.signature(wrong_second_type)
    with pytest.raises(TypeError, match=r"Second parameter must be a subclass of BaseModel, got <class 'int'>"):
        ingest_stage_callable_signature(sig)


def test_wrong_return_type():
    sig = inspect.signature(wrong_return_type)
    with pytest.raises(TypeError, match=r"Return type must be IngestControlMessage, got <class 'int'>"):
        ingest_stage_callable_signature(sig)


def test_valid_signature_passes():
    sig = inspect.signature(valid_stage)
    # Should not raise any exceptions
    assert ingest_stage_callable_signature(sig) is None
