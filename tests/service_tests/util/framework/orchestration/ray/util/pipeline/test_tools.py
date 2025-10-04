# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel
import pandas as pd
import sys
import types

from nv_ingest.framework.orchestration.ray.util.pipeline.tools import wrap_callable_as_stage
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


@pytest.fixture(autouse=True)
def fake_ray(monkeypatch):
    """Stub ray.remote and ray.get so tests run without a Ray runtime.

    - ray.remote(Class) returns a shim class exposing .remote(**kwargs) -> instance
    - instance methods expose .remote(...) that invoke the underlying method
    - ray.get(x) returns x directly
    """
    # Ensure a 'ray' module exists even if not installed
    if "ray" not in sys.modules:
        sys.modules["ray"] = types.SimpleNamespace()
    ray_mod = sys.modules["ray"]
    # Minimal is_initialized and get
    setattr(ray_mod, "is_initialized", lambda: True)
    setattr(ray_mod, "get", lambda x: x)

    def _remote(cls_or_fn):
        # wrap classes only (wrap_callable_as_stage returns a class)
        if isinstance(cls_or_fn, type):
            orig_cls = cls_or_fn

            class _ActorShim(orig_cls):
                @classmethod
                def remote(cls, *args, **kwargs):
                    inst = cls(*args, **kwargs)

                    class _Handle:
                        def __init__(self, obj):
                            self._obj = obj

                        def __getattr__(self, name):
                            attr = getattr(self._obj, name)
                            if callable(attr):

                                class _Method:
                                    def __init__(self, fn):
                                        self._fn = fn

                                    def remote(self, *a, **kw):
                                        return self._fn(*a, **kw)

                                return _Method(attr)
                            return attr

                    return _Handle(inst)

                @classmethod
                def options(cls, **_opts):
                    class _Opts:
                        def remote(self, *a, **kw):
                            return cls.remote(*a, **kw)

                    return _Opts()

            return _ActorShim
        # function path (unused here) — pass-through
        return cls_or_fn

    setattr(ray_mod, "remote", _remote)
    yield


class DummyConfig(BaseModel):
    foo: int
    bar: str = "baz"


def test_stage_processes_message_with_model_config():
    def fn(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        control_message.set_metadata("bar", stage_config.bar)
        return control_message

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    cfg = DummyConfig(foo=5, bar="quux")
    stage = Stage.remote(config=cfg)
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out = stage.on_data.remote(message)
    assert out.get_metadata("bar") == "quux"


def test_stage_processes_message_with_dict_config():
    """Given a valid config dict, on_data returns the value from the wrapped function."""

    def fn(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        control_message.set_metadata("result", stage_config.foo)
        return control_message

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    cfg = {"foo": 7}
    stage = Stage.remote(config=cfg)
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out = stage.on_data.remote(message)

    assert out is not None  # can't do "is msg" unless you handle object identity roundtripping
    assert out.get_metadata("result") == 7


def test_stage_returns_original_message_on_error():
    def fn(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        raise RuntimeError("fail!")

    Stage = wrap_callable_as_stage(fn, DummyConfig)
    stage = Stage.remote(config={"foo": 1})
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out = stage.on_data.remote(message)
    # Out may be a copy, check some invariant
    assert isinstance(out, IngestControlMessage)


def test_stage_can_chain_calls():
    def fn1(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        control_message.set_metadata("foo", stage_config.foo)
        return control_message

    def fn2(control_message: IngestControlMessage, stage_config: DummyConfig) -> IngestControlMessage:
        control_message.set_metadata("bar", stage_config.bar)
        return control_message

    Stage1 = wrap_callable_as_stage(fn1, DummyConfig)
    Stage2 = wrap_callable_as_stage(fn2, DummyConfig)
    stage1 = Stage1.remote(config={"foo": 42})
    stage2 = Stage2.remote(config={"foo": 0, "bar": "chained!"})
    message = IngestControlMessage()
    message.payload(pd.DataFrame())
    out1 = stage1.on_data.remote(message)
    out2 = stage2.on_data.remote(out1)

    assert out1.get_metadata("foo") == 42
    assert out2.get_metadata("bar") == "chained!"
