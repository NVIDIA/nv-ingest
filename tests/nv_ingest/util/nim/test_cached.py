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
    Test format_input for the 'grpc' protocol when given a 3-dimensional image array.
    The test verifies that the image is expanded along a new batch dimension and cast to float32.
    It also confirms that the accompanying batch data reflects the original image and its dimensions.
    """
    # Assume create_base64_image() returns a base64-encoded image that decodes to a (64, 64, 3) array.
    base64_img = create_base64_image()
    data = model_interface.prepare_data_for_inference({"base64_image": base64_img})

    # format_input returns a tuple: (batched_inputs, formatted_batch_data)
    formatted_batches, batch_data = model_interface.format_input(data, "grpc", max_batch_size=1)

    # Check that the batched input is a single numpy array with a new batch dimension.
    assert isinstance(formatted_batches, list)
    assert len(formatted_batches) == 1
    batched_input = formatted_batches[0]
    assert isinstance(batched_input, np.ndarray)
    assert batched_input.dtype == np.float32
    # The original image shape (64,64,3) should have been expanded to (1,64,64,3).
    assert batched_input.shape == (1, 64, 64, 3)

    # Verify that batch data contains the original image and its dimensions.
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1
    bd = batch_data[0]
    assert "image_arrays" in bd and "image_dims" in bd
    # The original image should be unmodified (still 3D) in batch_data.
    assert len(bd["image_arrays"]) == 1
    # Expect dimensions to be (H, W) i.e. (64, 64).
    assert bd["image_dims"] == [(64, 64)]


def test_format_input_grpc_with_ndim_other(model_interface):
    """
    Test format_input for the 'grpc' protocol when given a non-3-dimensional image array.
    This test uses a grayscale image which decodes to a 2D array.
    The expected behavior is that the image is cast to float32 without being expanded.
    Batch data is also checked for correct original dimensions.
    """
    # Create a grayscale (L mode) image of size 64x64.
    with BytesIO() as buffer:
        image = Image.new("L", (64, 64), 128)
        image.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    data = model_interface.prepare_data_for_inference({"base64_image": base64_img})
    formatted_batches, batch_data = model_interface.format_input(data, "grpc", max_batch_size=1)

    # Check that the batched input is a numpy array without expansion.
    assert isinstance(formatted_batches, list)
    assert len(formatted_batches) == 1
    batched_input = formatted_batches[0]
    assert isinstance(batched_input, np.ndarray)
    assert batched_input.dtype == np.float32
    # For a 2D image (64,64), no extra batch dimension is added when max_batch_size=1.
    assert batched_input.shape == (64, 64)

    # Verify that batch data correctly reports the original image dimensions.
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1
    bd = batch_data[0]
    assert "image_arrays" in bd and "image_dims" in bd
    assert len(bd["image_arrays"]) == 1
    # The image dimensions should reflect a 2D image: (64, 64)
    assert bd["image_dims"] == [(64, 64)]


def test_format_input_http(model_interface):
    """
    Test format_input for the 'http' protocol.
    This test ensures that given data with key "image_arrays", the images are re-encoded as PNG,
    and a single payload is built with a proper Nim message containing the image content.
    Additionally, it verifies that the accompanying batch data contains the original images and their dimensions.
    """
    # Generate a base64-encoded image and decode it into a numpy array.
    base64_img = create_base64_image()
    arr = base64_to_numpy(base64_img)

    # Build the data dictionary directly with the "image_arrays" key.
    data = {"image_arrays": [arr]}

    payload_batches, batch_data = model_interface.format_input(data, "http", max_batch_size=1)

    # Verify the HTTP payload structure.
    assert isinstance(payload_batches, list)
    assert len(payload_batches) == 1
    payload = payload_batches[0]
    assert "messages" in payload
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 1
    message = messages[0]
    assert "content" in message
    content_list = message["content"]
    assert isinstance(content_list, list)
    assert len(content_list) == 1
    content_item = content_list[0]
    assert content_item["type"] == "image_url"
    assert "image_url" in content_item and "url" in content_item["image_url"]

    # Check that the URL starts with the expected PNG base64 prefix.
    url_value = content_item["image_url"]["url"]
    expected_prefix = "data:image/png;base64,"
    assert url_value.startswith(expected_prefix)
    assert len(url_value) > len(expected_prefix)

    # Verify that the batch data is correctly built.
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1
    bd = batch_data[0]
    assert "image_arrays" in bd and "image_dims" in bd
    assert len(bd["image_arrays"]) == 1
    # The expected dimensions should match the original array's height and width.
    expected_dims = [(arr.shape[0], arr.shape[1])]
    assert bd["image_dims"] == expected_dims


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
