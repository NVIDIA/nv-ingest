# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Generator

import pandas as pd
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.util.pipeline.tools import wrap_callable_as_stage, wrap_callable_as_source
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


class ExampleSchema(BaseModel):
    multiplier: int


def test_stage_accepts_dict_config():
    def fn(cm: IngestControlMessage, config: ExampleSchema):
        return cm

    Stage = wrap_callable_as_stage(fn, ExampleSchema)
    stage = Stage({"multiplier": 2})
    assert stage.validated_config.multiplier == 2


def test_stage_accepts_model_config():
    def fn(cm: IngestControlMessage, config: ExampleSchema):
        return cm

    config = ExampleSchema(multiplier=3)
    Stage = wrap_callable_as_stage(fn, ExampleSchema)
    stage = Stage(config)
    assert stage.validated_config.multiplier == 3


def test_on_data_applies_fn():
    def fn(cm: IngestControlMessage, config: ExampleSchema):
        df = cm.payload()
        df["x"] = df["x"] * config.multiplier
        cm.payload(df)
        return cm

    Stage = wrap_callable_as_stage(fn, ExampleSchema)
    stage = Stage({"multiplier": 10})

    msg = IngestControlMessage()
    msg.payload(pd.DataFrame({"x": [1, 2, 3]}))

    result = stage.on_data(msg)
    pd.testing.assert_frame_equal(result.payload(), pd.DataFrame({"x": [10, 20, 30]}))


def test_on_data_catches_exception_and_returns_original_message():
    def fn(cm: IngestControlMessage, config: ExampleSchema):
        raise RuntimeError("intentional error")

    Stage = wrap_callable_as_stage(fn, ExampleSchema)
    stage = Stage({"multiplier": 1})
    msg = IngestControlMessage()

    result = stage.on_data(msg)
    assert result is msg
    assert stage.stats["errors"] == 1


class DummySourceSchema(BaseModel):
    count: int


def dummy_source_generator(config: DummySourceSchema) -> Generator[IngestControlMessage, None, None]:
    for i in range(config.count):
        msg = IngestControlMessage()
        msg.payload(pd.DataFrame({"x": [f"item_{i}"]}))
        yield msg


def test_lambda_source_emits_expected_messages():
    SourceStage = wrap_callable_as_source(dummy_source_generator, DummySourceSchema)
    stage = SourceStage(config={"count": 3})

    results = []
    for _ in range(3):
        msg = stage._read_input()
        assert isinstance(msg, IngestControlMessage)
        results.append(msg.payload()["x"].iloc[0])

    assert results == ["item_0", "item_1", "item_2"]


def test_lambda_source_stops_on_generator_exhaustion():
    SourceStage = wrap_callable_as_source(dummy_source_generator, DummySourceSchema)
    stage = SourceStage(config={"count": 2})

    _ = stage._read_input()
    _ = stage._read_input()
    final = stage._read_input()

    assert final is None
    assert stage._running is False


def test_lambda_source_pauses_and_resumes():
    SourceStage = wrap_callable_as_source(dummy_source_generator, DummySourceSchema)
    stage = SourceStage(config={"count": 1})

    stage.pause()
    assert stage.paused is True
    assert stage._read_input() is None

    stage.resume()
    assert stage.paused is False
    assert isinstance(stage._read_input(), IngestControlMessage)
