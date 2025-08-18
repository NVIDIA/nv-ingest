# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import inspect
from pydantic import BaseModel

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.util.imports.callable_signatures import ingest_stage_callable_signature


# --- Test config model ---
class DummyConfig(BaseModel):
    foo: int = 0


# --- Helper functions for negative and positive cases ---


def valid_stage(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
    return control_message


# Mismatched names
def wrong_param_names(a: IngestControlMessage, b: DummyConfig) -> IngestControlMessage:
    return a


def swapped_param_names(stage_config: DummyConfig, control_message: IngestControlMessage) -> IngestControlMessage:
    return control_message


def correct_types_wrong_param_names(message: IngestControlMessage, config: DummyConfig) -> IngestControlMessage:
    return message


# --- Annotation/Type Failures (existing from your set) ---
def no_params() -> IngestControlMessage:
    return IngestControlMessage()


def one_param(control_message: IngestControlMessage) -> IngestControlMessage:
    return control_message


def extra_params(control_message: IngestControlMessage, stage_config: DummyConfig, extra: str) -> IngestControlMessage:
    return control_message


def missing_first_annotation(control_message, stage_config: DummyConfig) -> IngestControlMessage:
    return control_message


def missing_second_annotation(control_message: IngestControlMessage, stage_config) -> IngestControlMessage:
    return control_message


def missing_return_annotation(control_message: IngestControlMessage, stage_config: DummyConfig):
    return control_message


def wrong_first_type(control_message: int, stage_config: DummyConfig) -> IngestControlMessage:
    return IngestControlMessage()


def wrong_second_type(control_message: IngestControlMessage, stage_config: int) -> IngestControlMessage:
    return control_message


def wrong_return_type(control_message: IngestControlMessage, stage_config: DummyConfig) -> int:
    return 123


# --- TESTS ---


def test_valid_signature_passes():
    sig = inspect.signature(valid_stage)
    assert ingest_stage_callable_signature(sig) is None


def test_wrong_param_names():
    sig = inspect.signature(wrong_param_names)
    with pytest.raises(TypeError, match="Expected parameter names: 'control_message', 'stage_config'"):
        ingest_stage_callable_signature(sig)


def test_swapped_param_names():
    sig = inspect.signature(swapped_param_names)
    with pytest.raises(TypeError, match="Expected parameter names: 'control_message', 'stage_config'"):
        ingest_stage_callable_signature(sig)


def test_correct_types_wrong_param_names():
    sig = inspect.signature(correct_types_wrong_param_names)
    with pytest.raises(TypeError, match="Expected parameter names: 'control_message', 'stage_config'"):
        ingest_stage_callable_signature(sig)


# Existing error cases


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
