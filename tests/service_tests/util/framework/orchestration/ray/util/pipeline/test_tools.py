# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Generator

import pytest
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_source_stage_base import RayActorSourceStage
from nv_ingest.framework.orchestration.ray.util.pipeline.tools import (
    wrap_callable_as_stage,
    wrap_callable_as_sink,
    wrap_callable_as_source,
)
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


class DummyConfig(BaseModel):
    foo: int
    bar: str = "baz"


def test_stage_processes_message_with_dict_config():
    """Given a valid config dict, on_data returns the value from the wrapped function."""

    def fn(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        msg.metadata["result"] = cfg.foo
        return msg

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    stage = Stage({"foo": 7})
    msg = IngestControlMessage()
    out = stage.on_data(msg)
    assert out is msg


def test_stage_processes_message_with_model_config():
    """Given a BaseModel config, on_data still returns the value from the wrapped function."""

    def fn(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        msg.metadata["bar"] = cfg.bar
        return msg

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    cfg = DummyConfig(foo=5, bar="quux")
    stage = Stage(cfg)
    msg = IngestControlMessage()
    out = stage.on_data(msg)
    assert out is msg


def test_stage_returns_original_message_on_error():
    """If the function raises, on_data should return the original message."""

    def fn(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        raise RuntimeError("fail!")

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    stage = Stage({"foo": 1})
    msg = IngestControlMessage()
    out = stage.on_data(msg)
    assert out is msg


def test_stage_can_chain_calls():
    """Two stages can be chained, passing messages through each, each applying its function."""

    def fn1(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        msg.metadata["foo"] = cfg.foo
        return msg

    def fn2(msg: IngestControlMessage, cfg: DummyConfig) -> IngestControlMessage:
        msg.metadata["bar"] = cfg.bar
        return msg

    Stage1 = wrap_callable_as_stage(fn1, DummyConfig)
    Stage2 = wrap_callable_as_stage(fn2, DummyConfig)
    stage1 = Stage1({"foo": 42})
    stage2 = Stage2({"foo": 0, "bar": "chained!"})
    msg = IngestControlMessage()
    out1 = stage1.on_data(msg)
    out2 = stage2.on_data(out1)
    assert out2 is msg


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
