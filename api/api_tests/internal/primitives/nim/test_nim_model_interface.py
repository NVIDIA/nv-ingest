# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from nv_ingest_api.internal.primitives.nim import ModelInterface


# Example concrete implementation for testing purposes
class DummyModel(ModelInterface):
    def format_input(self, data: dict, protocol: str, max_batch_size: int):
        return {"formatted": data, "protocol": protocol, "max_batch_size": max_batch_size}

    def parse_output(self, response, protocol: str, data: Optional[dict] = None, **kwargs):
        return {"parsed": response, "protocol": protocol, "data": data}

    def prepare_data_for_inference(self, data: dict):
        return {"prepared": data}

    def process_inference_results(self, output_array, protocol: str, **kwargs):
        return {"processed": output_array, "protocol": protocol}

    def name(self) -> str:
        return "DummyModel"


def test_format_input():
    model = DummyModel()
    result = model.format_input({"x": 1}, "grpc", 8)
    assert result["formatted"] == {"x": 1}
    assert result["protocol"] == "grpc"
    assert result["max_batch_size"] == 8


def test_parse_output():
    model = DummyModel()
    result = model.parse_output("response_data", "http", {"y": 2})
    assert result["parsed"] == "response_data"
    assert result["protocol"] == "http"
    assert result["data"] == {"y": 2}


def test_prepare_data_for_inference():
    model = DummyModel()
    result = model.prepare_data_for_inference({"z": 3})
    assert result["prepared"] == {"z": 3}


def test_process_inference_results():
    model = DummyModel()
    result = model.process_inference_results("raw_output", "grpc")
    assert result["processed"] == "raw_output"
    assert result["protocol"] == "grpc"


def test_name():
    model = DummyModel()
    assert model.name() == "DummyModel"
