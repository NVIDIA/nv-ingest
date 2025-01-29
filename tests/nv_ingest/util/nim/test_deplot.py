# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from io import BytesIO
import base64
from PIL import Image

# Importing the class under test
from nv_ingest.util.nim.deplot import DeplotModelInterface


@pytest.fixture
def model_interface():
    return DeplotModelInterface()


def create_base64_image(width=256, height=256, color=(255, 0, 0)):
    buffer = BytesIO()
    image = Image.new("RGB", (width, height), color)
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_name_returns_deplot(model_interface):
    assert model_interface.name() == "Deplot"


def test_prepare_data_for_inference_valid(model_interface):
    """
    Test prepare_data_for_inference with a single 'base64_image'.
    We now expect the returned dict to contain 'image_arrays', a list with one item.
    """
    base64_img = create_base64_image()
    data = {"base64_image": base64_img}
    result = model_interface.prepare_data_for_inference(data)

    # Check that we now have "image_arrays"
    assert "image_arrays" in result, "Expected 'image_arrays' key after inference preparation"
    assert len(result["image_arrays"]) == 1, "Expected exactly one image array"

    # Extract the first array and verify shape/type
    arr = result["image_arrays"][0]
    assert isinstance(arr, np.ndarray), "Expected a NumPy array"
    assert arr.shape == (256, 256, 3), "Expected a (256,256,3) shape"
    assert arr.dtype == np.uint8, "Expected dtype of uint8"


def test_prepare_data_for_inference_missing_base64_image(model_interface):
    data = {}
    with pytest.raises(KeyError, match="'base64_image'"):
        model_interface.prepare_data_for_inference(data)


def test_prepare_data_for_inference_invalid_base64_image(model_interface):
    data = {"base64_image": "not_valid_base64"}
    with pytest.raises(ValueError):
        model_interface.prepare_data_for_inference(data)


def test_format_input_grpc(model_interface):
    base64_img = create_base64_image()
    prepared = model_interface.prepare_data_for_inference({"base64_image": base64_img})
    formatted = model_interface.format_input(prepared, "grpc")
    assert isinstance(formatted, np.ndarray)
    assert formatted.dtype == np.float32
    assert formatted.ndim == 4
    assert formatted.shape == (1, 256, 256, 3)
    assert 0.0 <= formatted.min() and formatted.max() <= 1.0


def test_format_input_http(model_interface):
    base64_img = create_base64_image()
    prepared = model_interface.prepare_data_for_inference({"base64_image": base64_img})
    formatted = model_interface.format_input(prepared, "http", max_tokens=600, temperature=0.7, top_p=0.95)
    assert isinstance(formatted, dict)
    assert formatted["model"] == "google/deplot"
    assert isinstance(formatted["messages"], list)
    assert len(formatted["messages"]) == 1
    message = formatted["messages"][0]
    assert message["role"] == "user"
    assert message["content"].startswith("Generate the underlying data table")
    assert formatted["max_tokens"] == 600
    assert formatted["temperature"] == 0.7
    assert formatted["top_p"] == 0.95
    assert formatted["stream"] is False


def test_format_input_http_defaults(model_interface):
    base64_img = create_base64_image()
    prepared = model_interface.prepare_data_for_inference({"base64_image": base64_img})
    formatted = model_interface.format_input(prepared, "http")
    assert formatted["max_tokens"] == 500
    assert formatted["temperature"] == 0.5
    assert formatted["top_p"] == 0.9
    assert formatted["stream"] is False


def test_format_input_invalid_protocol(model_interface):
    base64_img = create_base64_image()
    prepared = model_interface.prepare_data_for_inference({"base64_image": base64_img})
    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        model_interface.format_input(prepared, "invalid")


def test_parse_output_grpc_simple(model_interface):
    """
    Test parse_output for gRPC protocol with a simple 2-element response.
    The new code returns a list of strings (one for each batch element).
    """
    # Each element is [b"Hello"] or [b"World"], so parse_output should decode and build a list.
    response = [[b"Hello"], [b"World"]]
    output = model_interface.parse_output(response, "grpc")

    # Now we expect ["Hello", "World"], not a single string.
    assert output == ["Hello", "World"]


def test_parse_output_grpc_multiple_bytes(model_interface):
    """
    Test parse_output for gRPC protocol with three elements.
    The new code returns a list, each decoding/combining the bytes in that sublist.
    """
    # Here, the code sees [b"Hello"], [b"world"], [b"!"] -> becomes ["Hello", "world", "!"]
    response = [[b"Hello"], [b"world"], [b"!"]]
    output = model_interface.parse_output(response, "grpc")
    assert output == ["Hello", "world", "!"]


def test_parse_output_grpc_empty(model_interface):
    """
    Test parse_output for gRPC protocol with an empty response.
    We now expect an empty list rather than an empty string.
    """
    response = []
    output = model_interface.parse_output(response, "grpc")
    assert output == [], "Expected an empty list for an empty response"


def test_parse_output_http_valid(model_interface):
    response = {"choices": [{"message": {"content": "Data table result"}}]}
    output = model_interface.parse_output(response, "http")
    assert output == "Data table result"


def test_parse_output_http_multiple_choices(model_interface):
    # Should return the content of the first choice only
    response = {"choices": [{"message": {"content": "First choice"}}, {"message": {"content": "Second choice"}}]}
    output = model_interface.parse_output(response, "http")
    assert output == "First choice"


def test_parse_output_http_missing_choices(model_interface):
    response = {}
    with pytest.raises(RuntimeError, match="Unexpected response format: 'choices' key is missing or empty."):
        model_interface.parse_output(response, "http")


def test_parse_output_http_empty_choices(model_interface):
    response = {"choices": []}
    with pytest.raises(RuntimeError, match="Unexpected response format: 'choices' key is missing or empty."):
        model_interface.parse_output(response, "http")
