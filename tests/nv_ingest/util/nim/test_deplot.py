# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from io import BytesIO
import base64
from PIL import Image

from nv_ingest_api.primitives.nim.model_interface.deplot import DeplotModelInterface


# Importing the class under test


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
    """
    Test that for the gRPC protocol:
      - The image (decoded from a base64 string) is normalized and batched.
      - The returned formatted batch is a NumPy array of shape (B, H, W, C) with dtype float32.
      - The accompanying batch data contains the original image and its dimensions.
    """
    base64_img = create_base64_image()
    prepared = model_interface.prepare_data_for_inference({"base64_image": base64_img})
    # format_input returns a tuple: (formatted_batches, formatted_batch_data)
    batches, batch_data = model_interface.format_input(prepared, "grpc", max_batch_size=1)

    formatted = batches[0]
    # Check the formatted batch
    assert isinstance(formatted, np.ndarray)
    assert formatted.dtype == np.float32
    # Since prepare_data_for_inference decodes to (256,256,3), the grpc branch expands it to (1,256,256,3)
    assert formatted.ndim == 4
    assert formatted.shape == (1, 256, 256, 3)
    # Ensure normalization to [0, 1]
    assert 0.0 <= formatted.min() and formatted.max() <= 1.0

    # Verify accompanying batch data
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1
    bd = batch_data[0]
    assert "image_arrays" in bd and "image_dims" in bd
    assert isinstance(bd["image_arrays"], list)
    assert len(bd["image_arrays"]) == 1
    # The original image should have shape (256,256,3)
    assert bd["image_arrays"][0].shape == (256, 256, 3)
    # Dimensions should be recorded as (height, width)
    assert bd["image_dims"] == [(256, 256)]


def test_format_input_http(model_interface):
    """
    Test that for the HTTP protocol:
      - The formatted payload is a JSON-serializable dict built via _prepare_deplot_payload.
      - The payload includes the expected keys (model, messages, max_tokens, stream, temperature, top_p)
      - And the accompanying batch data reflects the original image and its dimensions.
    """
    base64_img = create_base64_image()
    prepared = model_interface.prepare_data_for_inference({"base64_image": base64_img})
    batches, batch_data = model_interface.format_input(
        prepared, "http", max_batch_size=1, max_tokens=600, temperature=0.7, top_p=0.95
    )
    formatted = batches[0]

    # Check the payload structure from _prepare_deplot_payload
    assert isinstance(formatted, dict)
    assert formatted["model"] == "google/deplot"
    assert "messages" in formatted
    assert isinstance(formatted["messages"], list)
    assert len(formatted["messages"]) == 1
    message = formatted["messages"][0]
    assert message["role"] == "user"
    # The content should start with the fixed prompt text
    assert message["content"].startswith("Generate the underlying data table")
    # Check that the payload parameters match the supplied arguments
    assert formatted["max_tokens"] == 600
    assert formatted["temperature"] == 0.7
    assert formatted["top_p"] == 0.95
    assert formatted["stream"] is False

    # Verify accompanying batch data
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1
    bd = batch_data[0]
    assert "image_arrays" in bd and "image_dims" in bd
    assert isinstance(bd["image_arrays"], list)
    assert len(bd["image_arrays"]) == 1
    assert bd["image_arrays"][0].shape == (256, 256, 3)
    assert bd["image_dims"] == [(256, 256)]


def test_format_input_http_defaults(model_interface):
    """
    Test the HTTP branch when default parameters are used.
      - The default max_tokens, temperature, and top_p values should be applied.
      - The stream flag should be False.
      - Also verify that batch data is correctly returned.
    """
    base64_img = create_base64_image()
    prepared = model_interface.prepare_data_for_inference({"base64_image": base64_img})
    batches, batch_data = model_interface.format_input(prepared, "http", max_batch_size=1)
    formatted = batches[0]

    # Check that default values are set
    assert formatted["max_tokens"] == 500
    assert formatted["temperature"] == 0.5
    assert formatted["top_p"] == 0.9
    assert formatted["stream"] is False

    # Verify accompanying batch data
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1
    bd = batch_data[0]
    assert "image_arrays" in bd and "image_dims" in bd
    assert isinstance(bd["image_arrays"], list)
    assert len(bd["image_arrays"]) == 1
    assert bd["image_arrays"][0].shape == (256, 256, 3)
    assert bd["image_dims"] == [(256, 256)]


def test_format_input_invalid_protocol(model_interface):
    base64_img = create_base64_image()
    prepared = model_interface.prepare_data_for_inference({"base64_image": base64_img})
    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        model_interface.format_input(prepared, "invalid", max_batch_size=1)


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
