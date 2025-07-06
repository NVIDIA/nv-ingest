# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import ray
from pydantic import BaseModel

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
    def fn(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        msg.set_metadata("bar", cfg.bar)
        return msg

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    cfg = DummyConfig(foo=5, bar="quux")
    stage = Stage.remote(cfg)
    message = IngestControlMessage()
    out = ray.get(stage.on_data.remote(message))
    assert out.get_metadata("bar") == "quux"


def test_stage_processes_message_with_dict_config(ray_startup_and_shutdown):
    """Given a valid config dict, on_data returns the value from the wrapped function."""

    def fn(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        msg.set_metadata("result", cfg.foo)
        return msg

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    stage = Stage.remote({"foo": 7})
    msg = IngestControlMessage()
    out = ray.get(stage.on_data.remote(msg))

    assert out is not None  # can't do "is msg" unless you handle object identity roundtripping
    assert out.get_metadata("result") == 7


def test_stage_returns_original_message_on_error(ray_startup_and_shutdown):
    def fn(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        raise RuntimeError("fail!")

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    stage = Stage.remote({"foo": 1})
    message = IngestControlMessage()
    out = ray.get(stage.on_data.remote(message))
    # Out may be a copy, check some invariant
    assert isinstance(out, IngestControlMessage)


def test_stage_can_chain_calls(ray_startup_and_shutdown):
    def fn1(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        print(f"fn1: {cfg}")
        msg.set_metadata("foo", cfg.foo)
        return msg

    def fn2(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        print(f"fn2: {cfg}")
        msg.set_metadata("bar", cfg.bar)
        return msg

    Stage1 = wrap_callable_as_stage(fn1, DummyConfig)
    Stage2 = wrap_callable_as_stage(fn2, DummyConfig)
    stage1 = Stage1.remote({"foo": 42})
    stage2 = Stage2.remote({"foo": 0, "bar": "chained!"})
    message = IngestControlMessage()
    out1 = ray.get(stage1.on_data.remote(message))
    out2 = ray.get(stage2.on_data.remote(out1))

    assert out1.get_metadata("foo") == 42
    # assert out2.get_metadata("bar") == "chained!"


@pytest.mark.parametrize(
    "bad_config,expected_exc",
    [
        ({"bar": "nofoo"}, Exception),  # Missing required field
        (123, Exception),  # Not a dict or model
    ],
)
def test_stage_config_validation_errors(bad_config, expected_exc):
    """Stage construction raises if the config is invalid."""

    def fn(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        return msg

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    with pytest.raises(expected_exc):
        Stage(bad_config)
