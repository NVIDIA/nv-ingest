# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import numpy as np
import pytest

from io import BytesIO
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

    assert "image_array" in result
    assert isinstance(result["image_array"], np.ndarray)
    assert result["image_array"].shape == (64, 64, 3)  # Assuming RGB image
    assert result["image_array"].dtype == np.uint8  # Assuming image is loaded as uint8


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

    formatted_input = model_interface.format_input(data, "grpc")

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

    formatted_input = model_interface.format_input(data, "grpc")

    assert isinstance(formatted_input, np.ndarray)
    assert formatted_input.dtype == np.float32
    assert formatted_input.shape == (64, 64)  # No expansion


def test_format_input_http(model_interface):
    """
    Test format_input for 'http' protocol.
    Ensures that the HTTP payload is correctly formatted based on the base64_image.
    """
    base64_img = create_base64_image()
    data = {"base64_image": base64_img}
    expected_payload = {
        "messages": [{"content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}]}]
    }

    formatted_input = model_interface.format_input(data, "http")

    assert formatted_input == expected_payload


def test_format_input_invalid_protocol(model_interface):
    """
    Test format_input with an invalid protocol.
    Expects a ValueError to be raised.
    """

    base64_img = create_base64_image()
    data = model_interface.prepare_data_for_inference({"base64_image": base64_img})

    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        model_interface.format_input(data, "invalid_protocol")


def test_parse_output_grpc(model_interface):
    """
    Test parse_output for 'grpc' protocol.
    Ensures that byte responses are correctly decoded and concatenated.
    """
    response = [[b"Hello"], [b"World"]]  # Each output is a list containing a byte string

    parsed_output = model_interface.parse_output(response, "grpc")

    assert parsed_output == "Hello World"


def test_parse_output_http(model_interface):
    """
    Test parse_output for 'http' protocol.
    Ensures that content is correctly extracted from a valid HTTP JSON response.
    """
    json_response = {"data": [{"content": "Processed Content"}]}

    parsed_output = model_interface.parse_output(json_response, "http")

    assert parsed_output == "Processed Content"


def test_parse_output_http_missing_data_key(model_interface):
    """
    Test parse_output for 'http' protocol with missing 'data' key.
    Expects a RuntimeError to be raised.
    """
    json_response = {}

    with pytest.raises(RuntimeError, match="Unexpected response format: 'data' key is missing or empty."):
        model_interface.parse_output(json_response, "http")


def test_parse_output_http_empty_data(model_interface):
    """
    Test parse_output for 'http' protocol with empty 'data' list.
    Expects a RuntimeError to be raised.
    """
    # Arrange
    json_response = {"data": []}

    # Act & Assert
    with pytest.raises(RuntimeError, match="Unexpected response format: 'data' key is missing or empty."):
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


# Note: The following tests for private methods are optional and can be omitted
# in strict blackbox testing as they target internal implementations.


def test_prepare_nim_payload(model_interface):
    """
    Test the _prepare_nim_payload private method.
    Ensures that the NIM payload is correctly formatted.
    """
    base64_img = create_base64_image()
    expected_payload = {
        "messages": [{"content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}]}]
    }

    payload = model_interface._prepare_nim_payload(base64_img)

    assert payload == expected_payload


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
