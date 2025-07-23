# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import ray
from pydantic import BaseModel
import pandas as pd

from nv_ingest.framework.orchestration.ray.util.pipeline.tools import (
    wrap_callable_as_stage,
)
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


@pytest.fixture(scope="session", autouse=True)
def ray_startup_and_shutdown():
    # Only start if not already started (lets you run under pytest-xdist or interactively)
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


class DummyConfig(BaseModel):
    foo: int
    bar: str = "baz"


def test_stage_processes_message_with_model_config(ray_startup_and_shutdown):
    def fn(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        print(f"DEBUG: fn called with stage_config.bar = {stage_config.bar}")
        try:
            control_message.set_metadata("bar", stage_config.bar)
            print(f"DEBUG: set_metadata completed, metadata now: {control_message.get_metadata('bar')}")
            return control_message
        except Exception as e:
            print(f"DEBUG: Exception in fn: {e}")
            raise

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    cfg = DummyConfig(foo=5, bar="quux")
    stage = Stage.remote(config=cfg)
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    print(f"DEBUG: Before calling on_data, message metadata: {message.get_metadata('bar')}")
    out = ray.get(stage.on_data.remote(message))
    print(f"DEBUG: After calling on_data, out metadata: {out.get_metadata('bar')}")
    something = out
    assert out.get_metadata("bar") == "quux"


def test_stage_processes_message_with_dict_config(ray_startup_and_shutdown):
    """Given a valid config dict, on_data returns the value from the wrapped function."""

    def fn(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        control_message.set_metadata("result", stage_config.foo)
        return control_message

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    cfg = {"foo": 7}
    stage = Stage.remote(config=cfg)
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out = ray.get(stage.on_data.remote(message))

    assert out is not None  # can't do "is msg" unless you handle object identity roundtripping
    assert out.get_metadata("result") == 7


def test_stage_returns_original_message_on_error(ray_startup_and_shutdown):
    def fn(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        raise RuntimeError("fail!")

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    stage = Stage.remote(config={"foo": 1})
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out = ray.get(stage.on_data.remote(message))
    # Out may be a copy, check some invariant
    assert isinstance(out, IngestControlMessage)


def test_stage_can_chain_calls(ray_startup_and_shutdown):
    def fn1(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        print(f"fn1: {stage_config}")
        control_message.set_metadata("foo", stage_config.foo)
        return control_message

    def fn2(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        print(f"fn2: {stage_config}")
        control_message.set_metadata("bar", stage_config.bar)
        return control_message

    Stage1 = wrap_callable_as_stage(fn1, DummyConfig)
    Stage2 = wrap_callable_as_stage(fn2, DummyConfig)
    stage1 = Stage1.remote(config={"foo": 42})
    stage2 = Stage2.remote(config={"foo": 0, "bar": "chained!"})
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out1 = ray.get(stage1.on_data.remote(message))
    out2 = ray.get(stage2.on_data.remote(out1))

    assert out1.get_metadata("foo") == 42
    assert out2.get_metadata("bar") == "chained!"
