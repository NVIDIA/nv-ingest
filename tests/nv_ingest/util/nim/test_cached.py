# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import numpy as np
import pytest

from io import BytesIO

from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.cached import CachedModelInterface
from PIL import Image


@pytest.fixture
def model_interface():
    """Fixture to instantiate the CachedModelInterface."""
    return CachedModelInterface()


def create_base64_image(width=64, height=64, color=(255, 0, 0)):
    """
    Creates a base64-encoded PNG image.

    Parameters:
    ----------
    width : int
        Width of the image.
    height : int
        Height of the image.
    color : tuple
        RGB color of the image.

    Returns:
    -------
    str
        Base64-encoded string of the image.
    """

    with BytesIO() as buffer:
        image = Image.new("RGB", (width, height), color)
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_name_returns_cached(model_interface):
    """Test that the name method returns 'Cached'."""
    assert model_interface.name() == "Cached"


def test_prepare_data_for_inference_valid(model_interface):
    """
    Test prepare_data_for_inference with a valid base64_image.
    Ensures that image_array is added to the data dictionary.
    """
    base64_img = create_base64_image()
    input_data = {"base64_image": base64_img}

    result = model_interface.prepare_data_for_inference(input_data)
    print(result)

    assert "image_arrays" in result
    assert isinstance(result["image_arrays"][0], np.ndarray)
    assert result["image_arrays"][0].shape == (64, 64, 3)  # Assuming RGB image
    assert result["image_arrays"][0].dtype == np.uint8  # Assuming image is loaded as uint8


def test_prepare_data_for_inference_invalid_base64(model_interface):
    """
    Test prepare_data_for_inference with an invalid base64_image.
    Expects an exception to be raised.
    """
    invalid_base64_img = "invalid_base64_string"
    input_data = {"base64_image": invalid_base64_img}

    with pytest.raises(Exception):
        model_interface.prepare_data_for_inference(input_data)


def test_prepare_data_for_inference_missing_base64_image(model_interface):
    """
    Test prepare_data_for_inference when 'base64_image' key is missing.
    Expects a KeyError to be raised.
    """
    input_data = {}

    with pytest.raises(KeyError, match="'base64_image'"):
        model_interface.prepare_data_for_inference(input_data)


def test_format_input_grpc_with_ndim_3(model_interface):
    """
    Test format_input for 'grpc' protocol with a 3-dimensional image array.
    Expects the image array to be expanded and cast to float32.
    """
    base64_img = create_base64_image()
    data = model_interface.prepare_data_for_inference({"base64_image": base64_img})

    formatted_input = model_interface.format_input(data, "grpc", max_batch_size=1)[0]

    assert isinstance(formatted_input, np.ndarray)
    assert formatted_input.dtype == np.float32
    assert formatted_input.shape == (1, 64, 64, 3)  # Expanded along axis 0


def test_format_input_grpc_with_ndim_other(model_interface):
    """
    Test format_input for 'grpc' protocol with a non-3-dimensional image array.
    Expects the image array to be cast to float32 without expansion.
    """
    # Create a grayscale image (2D array)
    with BytesIO() as buffer:
        image = Image.new("L", (64, 64), 128)  # 'L' mode for grayscale
        image.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    data = model_interface.prepare_data_for_inference({"base64_image": base64_img})

    formatted_input = model_interface.format_input(data, "grpc", max_batch_size=1)[0]

    assert isinstance(formatted_input, np.ndarray)
    assert formatted_input.dtype == np.float32
    assert formatted_input.shape == (64, 64)  # No expansion


def test_format_input_http(model_interface):
    """
    Test format_input for 'http' protocol under the new approach:
     - The code expects 'image_arrays' in data
     - Each array is re-encoded as PNG
     - A single Nim message is built with multiple images in the 'content' array
    """
    # 1) Create a small in-memory base64 image
    #    This is just a placeholder function to generate or load some base64 data
    base64_img = create_base64_image()

    # 2) Decode it into a NumPy array (mimicking prepare_data_for_inference)
    arr = base64_to_numpy(base64_img)  # or however your code does this

    # 3) Build the data dict with "image_arrays"
    data = {"image_arrays": [arr]}  # single array for a single test image

    # 4) Call format_input
    formatted_input = model_interface.format_input(data, "http", max_batch_size=1)[0]

    # 5) Verify the structure of the HTTP payload

    assert "messages" in formatted_input, "Expected 'messages' key in output"
    assert len(formatted_input["messages"]) == 1, "Expected exactly 1 message"

    message = formatted_input["messages"][0]
    assert "content" in message, "Expected 'content' key in the message"
    assert len(message["content"]) == 1, "Expected exactly 1 image in content for this test"

    content_item = message["content"][0]
    assert content_item["type"] == "image_url", "Expected 'type' == 'image_url'"
    assert "image_url" in content_item, "Expected 'image_url' key in content item"
    assert "url" in content_item["image_url"], "Expected 'url' key in 'image_url' dict"

    # 6) Optionally, check the prefix of the base64 URL
    url_value = content_item["image_url"]["url"]
    assert url_value.startswith(
        "data:image/png;base64,"
    ), f"URL should start with data:image/png;base64, but got {url_value[:30]}..."
    # And check that there's something after the prefix
    assert len(url_value) > len("data:image/png;base64,"), "Base64 string seems empty"


def test_format_input_invalid_protocol(model_interface):
    """
    Test format_input with an invalid protocol.
    Expects a ValueError to be raised.
    """

    base64_img = create_base64_image()
    data = model_interface.prepare_data_for_inference({"base64_image": base64_img})

    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        model_interface.format_input(data, "invalid_protocol", max_batch_size=1)


def test_parse_output_grpc(model_interface):
    """
    Test parse_output for 'grpc' protocol.
    Ensures that byte responses are correctly decoded into a list of strings.
    """
    # Suppose the new parse_output returns ["Hello", "World"] for this input
    response = [[b"Hello"], [b"World"]]  # Each output is a list of byte strings

    parsed_output = model_interface.parse_output(response, "grpc")

    # The updated code might now produce a list rather than a single concatenated string
    assert parsed_output == ["Hello", "World"]


def test_parse_output_http(model_interface):
    """
    Test parse_output for 'http' protocol.
    Ensures that content is correctly extracted from a valid HTTP JSON response
    and returned as a list of strings.
    """
    # Single "data" entry. The new code returns a list, even if there's only 1 item.
    json_response = {"data": [{"content": "Processed Content"}]}

    parsed_output = model_interface.parse_output(json_response, "http")

    # Expect a list with exactly one string in it
    assert parsed_output == ["Processed Content"]


def test_parse_output_http_missing_data_key(model_interface):
    """
    Test parse_output for 'http' protocol with missing 'data' key.
    Expects a RuntimeError to be raised.
    """
    json_response = {}

    with pytest.raises(RuntimeError, match="Unexpected response format: 'data' key missing or empty."):
        model_interface.parse_output(json_response, "http")


def test_parse_output_http_empty_data(model_interface):
    """
    Test parse_output for 'http' protocol with empty 'data' list.
    Expects a RuntimeError to be raised.
    """
    json_response = {"data": []}

    with pytest.raises(RuntimeError, match="Unexpected response format: 'data' key missing or empty."):
        model_interface.parse_output(json_response, "http")


def test_parse_output_invalid_protocol(model_interface):
    """
    Test parse_output with an invalid protocol.
    Expects a ValueError to be raised.
    """
    response = "Some response"

    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        model_interface.parse_output(response, "invalid_protocol")


def test_process_inference_results(model_interface):
    """
    Test process_inference_results method.
    Ensures that the method returns the output as-is.
    """
    output = "Processed Output"

    result = model_interface.process_inference_results(output, "http")

    assert result == output


def test_extract_content_from_nim_response_valid(model_interface):
    """
    Test the _extract_content_from_nim_response private method with valid response.
    Ensures that content is correctly extracted.
    """
    json_response = {"data": [{"content": "Extracted Content"}]}

    content = model_interface._extract_content_from_nim_response(json_response)

    assert content == "Extracted Content"


def test_extract_content_from_nim_response_missing_data(model_interface):
    """
    Test the _extract_content_from_nim_response private method with missing 'data' key.
    Expects a RuntimeError to be raised.
    """
    json_response = {}

    with pytest.raises(RuntimeError, match="Unexpected response format: 'data' key is missing or empty."):
        model_interface._extract_content_from_nim_response(json_response)


def test_extract_content_from_nim_response_empty_data(model_interface):
    """
    Test the _extract_content_from_nim_response private method with empty 'data' list.
    Expects a RuntimeError to be raised.
    """
    json_response = {"data": []}

    with pytest.raises(RuntimeError, match="Unexpected response format: 'data' key is missing or empty."):
        model_interface._extract_content_from_nim_response(json_response)
