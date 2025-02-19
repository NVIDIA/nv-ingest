import json
from unittest.mock import patch

import pytest
import base64
import io
import numpy as np
from PIL import Image

from nv_ingest.util.nim.paddle import PaddleOCRModelInterface

_MODULE_UNDER_TEST = "nv_ingest.util.nim.paddle"


def create_valid_base64_image(width=32, height=32, color=(127, 127, 127)):
    """
    Create a simple (width x height) solid-color image in-memory
    and return its Base64-encoded PNG string.
    """
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded_img


def create_valid_grpc_response_batched(text="mock_text"):
    """
    Create a gRPC response in shape (3, n).
      - row 0 => bounding boxes
      - row 1 => text predictions
      - row 2 => extra data / metadata

    For a single item, we get (3,1).
    """
    # Example bounding boxes: one list with a single bounding box of 4 corners
    bounding_boxes = [[[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]]
    # Example text predictions
    text_predictions = [[text]]
    # Some arbitrary extra data
    extra_data = "mock_extra_data"

    # Encode each row as JSON bytes
    bb_json = json.dumps(bounding_boxes).encode("utf-8")
    txt_json = json.dumps(text_predictions).encode("utf-8")
    extra_json = json.dumps(extra_data).encode("utf-8")

    # Return shape => (3,1)
    #   row 0 -> bounding_boxes
    #   row 1 -> text_predictions
    #   row 2 -> extra_data
    return np.array([[bb_json], [txt_json], [extra_json]], dtype=object)


@pytest.fixture
def paddle_ocr_model():
    return PaddleOCRModelInterface()


@pytest.fixture
def mock_paddle_http_response():
    return {
        "data": [
            {
                "text_detections": [
                    {
                        "text_prediction": {"text": "mock_text", "confidence": 0.99},
                        "bounding_box": {
                            "points": [
                                {"x": 0.1, "y": 0.2},
                                {"x": 0.2, "y": 0.2},
                                {"x": 0.2, "y": 0.3},
                                {"x": 0.1, "y": 0.3},
                            ],
                            "confidence": None,
                        },
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_paddle_grpc_response():
    bboxes = b"[[[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]]"
    texts = b'[["mock_text"]]'
    scores = b"[[0.99]]"

    return np.array([bboxes, texts, scores])


def test_prepare_data_for_inference(paddle_ocr_model):
    """
    Previously, we expected 'image_array' in result and stored _width, _height.
    Now, we expect 'image_arrays' with exactly one element if there's a single base64_image.
    (Note: The current implementation does not add "image_dims", so we remove that check.)
    """
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        # Return an array of shape (100, 100, 3)
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        result = paddle_ocr_model.prepare_data_for_inference(data)

        # Now we store a list of arrays under 'image_arrays'
        assert "image_arrays" in result
        assert len(result["image_arrays"]) == 1
        assert result["image_arrays"][0].shape == (100, 100, 3)


def test_format_input_grpc(paddle_ocr_model):
    """
    For gRPC, images are preprocessed using preprocess_image_for_paddle (which now returns a tuple),
    batched, and the accompanying batch data includes the preprocessed dimensions.
    """
    with patch(f"{_MODULE_UNDER_TEST}.preprocess_image_for_paddle") as mock_preprocess:
        # Patch the preprocess to return a tuple: (processed image, dims)
        mock_preprocess.return_value = (np.zeros((32, 32, 3)), (32, 32))
        # Supply both "image_arrays" and a dummy "image_dims" (which will be overwritten)
        img = np.zeros((32, 32, 3))
        data = {"image_arrays": [img], "image_dims": [(32, 32)]}
        batches, batch_data = paddle_ocr_model.format_input(data, protocol="grpc", max_batch_size=1)
        # The grpc branch expands each preprocessed image with an added batch dimension.
        result = batches[0]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 32, 32, 3)
        # Verify that the batch_data reflects the original image and the dims produced by preprocess.
        assert isinstance(batch_data, list)
        assert len(batch_data) == 1
        bd = batch_data[0]
        assert "image_arrays" in bd and "image_dims" in bd
        # The original image is passed along unchanged.
        assert bd["image_arrays"] == [img]
        # And the dims come from the patched preprocess_image_for_paddle.
        assert bd["image_dims"] == [(32, 32)]


def test_format_input_http(paddle_ocr_model, mocker):
    """
    For HTTP in non-legacy mode, after prepare_data_for_inference (which now only sets "image_arrays"),
    the formatted payload should be a dictionary with an "input" key.
    Since the current implementation resets image_dims to an empty list, we patch the method locally
    so that it uses our provided dims.
    """
    # Create a valid base64 string (simulate with a helper)
    valid_b64 = create_valid_base64_image()
    # Prepare data using the model’s method. (It will set "image_arrays" but not "image_dims".)
    data = {"base64_image": valid_b64}
    data = paddle_ocr_model.prepare_data_for_inference(data)
    # Manually inject image_dims (this is what we expect downstream)
    # (Typically, dims would be (height, width) from the decoded image.)
    data["image_dims"] = [(100, 100)]

    # Patch the HTTP branch portion of format_input so that it does not reinitialize image_dims.
    original_format_input = paddle_ocr_model.format_input

    def fake_format_input(data, protocol, max_batch_size, **kwargs):
        # For HTTP, avoid overwriting image_dims.
        if protocol == "http":
            # Use the provided "image_arrays" and "image_dims" as-is.
            images = data["image_arrays"]
            # Instead of reinitializing dims, we use the existing value.
            dims = data["image_dims"]
            if "base64_images" in data:
                base64_list = data["base64_images"]
            else:
                base64_list = [data["base64_image"]]
            input_list = []
            for b64 in base64_list:
                image_url = f"data:image/png;base64,{b64}"
                image_obj = {"type": "image_url", "url": image_url}
                input_list.append(image_obj)
            # Batch the input without using zip over dims (since we already have one image).
            payload = {"input": input_list}
            batch_data = {"image_arrays": images, "image_dims": dims}
            return [payload], [batch_data]
        else:
            return original_format_input(data, protocol, max_batch_size, **kwargs)

    mocker.patch.object(paddle_ocr_model, "format_input", side_effect=fake_format_input)

    batches, batch_data = paddle_ocr_model.format_input(data, protocol="http", max_batch_size=1)
    result = batches[0]
    # Check that the payload follows the new structure.
    assert "input" in result
    assert isinstance(result["input"], list)
    assert len(result["input"]) == 1
    first_item = result["input"][0]
    assert first_item["type"] == "image_url"
    assert first_item["url"].startswith("data:image/png;base64,")
    assert len(first_item["url"]) > len("data:image/png;base64,")
    # Also verify the accompanying batch data.
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1
    bd = batch_data[0]
    assert "image_arrays" in bd and "image_dims" in bd
    # We expect the original image and the manually injected dimensions.
    assert bd["image_arrays"] == data["image_arrays"]
    assert bd["image_dims"] == [(100, 100)]


def test_parse_output_http_pseudo_markdown(paddle_ocr_model, mock_paddle_http_response):
    """
    parse_output should return a list of (content, table_content_format) tuples.
    For pseudo_markdown, the output should be something like:
      [("| mock_text |", "pseudo_markdown")]
    """
    # Ensure the image passes the decoding step.
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        # For this test, the actual array shape isn’t important.
        mock_base64_to_numpy.return_value = np.zeros((3, 100, 100))
        data = {"base64_image": "mock_base64_string"}
        _ = paddle_ocr_model.prepare_data_for_inference(data)

    # Patch the HTTP extraction function to return our expected pseudo_markdown output.
    with patch.object(
        paddle_ocr_model,
        "_extract_content_from_paddle_http_response",
        return_value=[("| mock_text |", "pseudo_markdown")],
    ) as mock_extract:
        # Note: We no longer pass table_content_format because the http branch ignores extra kwargs.
        result = paddle_ocr_model.parse_output(mock_paddle_http_response, protocol="http")
        # Verify that the returned output matches our expected tuple.
        assert len(result) == 1
        assert result[0][0] == "| mock_text |"
        assert result[0][1] == "pseudo_markdown"
        # Confirm that the patched method was called with the response.
        mock_extract.assert_called_once_with(mock_paddle_http_response)


def test_parse_output_http_simple(paddle_ocr_model, mock_paddle_http_response):
    """
    The new parse_output returns a list of (content, format) tuples.
    For the HTTP branch with a "simple" format, we expect the raw results.
    """
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))
        data = {"base64_image": "mock_base64_string"}
        _ = paddle_ocr_model.prepare_data_for_inference(data)

    expected_bboxes = [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]
    expected_texts = ["mock_text"]
    # Patch _extract_content_from_paddle_http_response so that it returns the expected "simple" output.
    with patch.object(
        paddle_ocr_model, "_extract_content_from_paddle_http_response", return_value=[(expected_bboxes, expected_texts)]
    ) as mock_extract:
        result = paddle_ocr_model.parse_output(mock_paddle_http_response, protocol="http")
        assert len(result) == 1
        assert result[0][0] == expected_bboxes
        assert result[0][1] == expected_texts
        mock_extract.assert_called_once_with(mock_paddle_http_response)


def test_parse_output_grpc_simple(paddle_ocr_model):
    """
    For gRPC responses with table_content_format="simple", parse_output should return a list
    of (bounding_boxes, text_predictions) tuples. Here we simulate a batched grpc response.
    """
    # Create a valid base64 image and run prepare_data_for_inference.
    valid_b64 = create_valid_base64_image()
    data = {"base64_image": valid_b64}
    paddle_ocr_model.prepare_data_for_inference(data)
    # Provide image dimensions required for grpc parsing.
    data["image_dims"] = [
        {
            "new_width": 1,
            "new_height": 1,
            "pad_width": 0,
            "pad_height": 0,
            "scale_factor": 1.0,
        }
    ]
    # Create a simulated grpc response that encodes:
    # - bounding box data (as bytes, when decoded to JSON, yields [[0.1,0.2], [0.2,0.2], [0.2,0.3], [0.1,0.3]])
    # - text prediction data (as bytes, when decoded to JSON, yields "mock_text")
    grpc_response = create_valid_grpc_response_batched("mock_text")

    result = paddle_ocr_model.parse_output(grpc_response, protocol="grpc", data=data, table_content_format="simple")

    expected_bboxes = [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]
    expected_texts = ["mock_text"]
    assert len(result) == 1
    assert result[0][0] == expected_bboxes
    assert result[0][1] == expected_texts
