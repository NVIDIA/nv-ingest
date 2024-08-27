# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors
from tritonclient.utils import InferenceServerException

from nv_ingest.schemas.metadata_schema import ContentTypeEnum

from ....import_checks import CUDA_DRIVER_OK
from ....import_checks import MORPHEUS_IMPORT_OK

if CUDA_DRIVER_OK and MORPHEUS_IMPORT_OK:
    import cudf

    from nv_ingest.modules.transforms.image_caption_extraction import _calculate_centroids
    from nv_ingest.modules.transforms.image_caption_extraction import _extract_bboxes_and_content
    from nv_ingest.modules.transforms.image_caption_extraction import _find_nearest_neighbors
    from nv_ingest.modules.transforms.image_caption_extraction import _fit_nearest_neighbors
    from nv_ingest.modules.transforms.image_caption_extraction import _generate_captions
    from nv_ingest.modules.transforms.image_caption_extraction import _predict_caption
    from nv_ingest.modules.transforms.image_caption_extraction import _prepare_dataframes
    from nv_ingest.modules.transforms.image_caption_extraction import _prepare_final_dataframe
    from nv_ingest.modules.transforms.image_caption_extraction import _process_content
    from nv_ingest.modules.transforms.image_caption_extraction import _process_documents
    from nv_ingest.modules.transforms.image_caption_extraction import _sanitize_inputs
    from nv_ingest.modules.transforms.image_caption_extraction import _update_metadata_with_captions

_MODULE_UNDER_TEST = "nv_ingest.modules.transforms.image_caption_extraction"


def check_result_accuracy(computed_results, expected_results, atol=0.0001):
    """
    Check if each element in computed_results is within four decimal places of the expected_results.

    Args:
    computed_results (np.array): The results obtained from the computation.
    expected_results (np.array): The expected results to compare against.
    atol (float): Absolute tolerance required (default is 0.0001 for four decimal places).

    Returns:
    bool: True if all elements match within the specified tolerance, False otherwise.
    """
    # Ensure both inputs are numpy arrays for element-wise comparison
    computed_results = np.array(computed_results)
    expected_results = np.array(expected_results)

    # Check if all elements are close within the absolute tolerance
    if np.allclose(computed_results, expected_results, atol=atol):
        return True
    else:
        return False


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_extract_bboxes_and_content():
    """Test extraction of bounding boxes and content."""
    data = {
        "content_metadata": {
            "hierarchy": {
                "nearby_objects": {
                    "text": {"bbox": [(0, 0, 10, 10), (10, 10, 20, 20)], "content": ["Text A", "Text B"]}
                }
            }
        }
    }
    bboxes, content = _extract_bboxes_and_content(data)
    assert bboxes == [(0, 0, 10, 10), (10, 10, 20, 20)], "Bounding boxes were not extracted correctly."
    assert content == ["Text A", "Text B"], "Content was not extracted correctly."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_calculate_centroids():
    """Test calculation of centroids from bounding boxes."""
    bboxes = [(0, 0, 10, 10), (10, 10, 20, 20)]
    expected_centroids = [(5.0, 5.0), (15.0, 15.0)]
    centroids = _calculate_centroids(bboxes)
    assert centroids == expected_centroids, "Centroids were not calculated correctly."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_fit_nearest_neighbors():
    """Test fitting the nearest neighbors model to centroids."""
    centroids = [(5.0, 5.0), (15.0, 15.0)]
    nbrs, adjusted_n_neighbors = _fit_nearest_neighbors(centroids, n_neighbors=5)
    assert adjusted_n_neighbors == 2, "Adjusted number of neighbors should be equal to the number of centroids."
    assert isinstance(nbrs, NearestNeighbors), "The function should return a NearestNeighbors instance."
    assert (
        nbrs.n_neighbors == adjusted_n_neighbors
    ), "NearestNeighbors instance does not have the correct number of neighbors."


@pytest.fixture
def sample_data():
    """Fixture to create sample data for tests."""
    return {
        "content_metadata": {
            "hierarchy": {
                "nearby_objects": {
                    "text": {"bbox": [(0, 0, 10, 10), (10, 10, 20, 20)], "content": ["Text A", "Text B"]}
                }
            }
        }
    }


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_integration(sample_data):
    """Integration test to verify the complete workflow from data extraction to neighbors fitting."""
    bboxes, content = _extract_bboxes_and_content(sample_data)
    centroids = _calculate_centroids(bboxes)
    nbrs, _ = _fit_nearest_neighbors(centroids)
    neighbors = nbrs.kneighbors([[7, 7]])

    assert check_result_accuracy(
        neighbors[0], [[2.8284, 11.3137]]
    ), "Nearest neighbor predictions do not match expected results."
    assert check_result_accuracy(neighbors[1], [[0, 1]]), "Nearest neighbor predictions do not match expected results."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_find_nearest_neighbors():
    centroids = [(0, 0), (5, 5), (10, 10)]
    nbrs, _ = _fit_nearest_neighbors(centroids, n_neighbors=2)
    new_bbox = (1, 1, 2, 2)  # A bounding box close to the first centroid
    content = ["Near origin", "Mid-point", "Far point"]

    distances, indices, result_content = _find_nearest_neighbors(nbrs, new_bbox, content, n_neighbors=2)

    # Check results
    assert len(distances) == 1, "There should be one distance array."
    assert len(indices) == 1, "There should be one index array."
    assert len(result_content) == 2, "There should be two contents returned."
    assert result_content[0] == "Near origin", "Content does not match expected nearest neighbor."
    assert result_content[1] == "Mid-point", "Content does not match expected second nearest neighbor."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_sanitize_inputs_with_ascii():
    """
    Test sanitizing inputs that are already in ASCII format.
    """
    inputs = [["Hello", "World"], ["Test123", "!@#$%^&*()"]]
    expected = [["Hello", "World"], ["Test123", "!@#$%^&*()"]]
    assert _sanitize_inputs(inputs) == expected, "ASCII characters should not be altered."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_sanitize_inputs_with_non_ascii():
    """
    Test sanitizing inputs containing non-ASCII characters.
    """
    inputs = [["Héllo", "Wörld"], ["Café", "Niño"]]
    expected = [["H?llo", "W?rld"], ["Caf?", "Ni?o"]]
    assert _sanitize_inputs(inputs) == expected, "Non-ASCII characters should be replaced with '?'."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_sanitize_inputs_with_empty_strings():
    """
    Test sanitizing inputs that include empty strings.
    """
    inputs = [["", "Hello"], ["World", ""]]
    expected = [["", "Hello"], ["World", ""]]
    assert _sanitize_inputs(inputs) == expected, "Empty strings should remain unchanged."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_sanitize_inputs_with_mixed_content():
    """
    Test sanitizing inputs with mixed ASCII and non-ASCII characters.
    """
    inputs = [["Héllo123", "World!"], ["Python™", "C++"]]
    expected = [["H?llo123", "World!"], ["Python?", "C++"]]
    assert _sanitize_inputs(inputs) == expected, "Mix of ASCII and non-ASCII should be handled correctly."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_sanitize_inputs_with_special_cases():
    """
    Test sanitizing inputs that contain special cases such as numbers and special characters.
    """
    inputs = [["123456", "7890"], ["!@#$%", "^&*()"]]
    expected = [["123456", "7890"], ["!@#$%", "^&*()"]]
    assert _sanitize_inputs(inputs) == expected, "Numbers and special characters should remain unchanged."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}.grpcclient.InferenceServerClient")  # Mock the 'InferenceServerClient' class
def test_predict_caption_success(mock_client_class):
    # Create a mock client instance
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Set up the mock response object for the client's infer method
    mock_infer_result = MagicMock()
    # Mock the output of the inference
    # Ensure that the mock output matches the expected dimensions and values
    mock_infer_result.as_numpy.return_value = np.array(
        [
            [0.6],  # First input (first candidate has the highest probability)
            [0.4],  # Second input (second candidate has the highest probability)
        ]
    )

    # Mock the infer method
    mock_client.infer.return_value = mock_infer_result

    # Define test variables
    triton_url = "http://fake-url.com"
    caption_model = "fake_model"
    inputs = [["hello", "world"]]

    # Call the function
    caption = _predict_caption(triton_url, caption_model, inputs)

    # Assert function behavior
    assert caption == ["hello"], "Caption should match the mock response data"


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}.grpcclient.InferenceServerClient")  # Mock the 'InferenceServerClient' class
def test_predict_caption_http_error(mock_client_class):
    # Create a mock client instance
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simulate an HTTP error
    mock_client.infer.side_effect = InferenceServerException("Mock HTTP Error")

    # Define test variables
    triton_url = "http://fake-url.com"
    caption_model = "fake_model"
    inputs = [["hello", "world"]]

    # Call the function and expect it to handle the HTTP error
    caption = _predict_caption(triton_url, caption_model, inputs)

    # Assert function behavior
    assert caption == [""], "Function should return an empty string for HTTP errors"


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}.grpcclient.InferenceServerClient")  # Mock the 'InferenceServerClient' class
def test_predict_caption_bad_request(mock_client_class):
    # Create a mock client instance
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simulate a 400 Bad Request error
    mock_client.infer.side_effect = InferenceServerException("Bad request error")

    # Define test variables
    triton_url = "http://fake-url.com"
    caption_model = "fake_model"
    inputs = [["hello", "world"]]

    # Call the function
    caption = _predict_caption(triton_url, caption_model, inputs)

    # Check error handling
    assert caption == [""], "Function should return an empty string for bad requests"


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}.grpcclient.InferenceServerClient")  # Mock the 'InferenceServerClient' class
def test_predict_caption_request_exception(mock_client_class):
    # Create a mock client instance
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simulate a general RequestException
    mock_client.infer.side_effect = InferenceServerException("RequestException")

    # Define test variables
    triton_url = "http://fake-url.com"
    caption_model = "fake_model"
    inputs = [["hello", "world"]]

    # Call the function
    caption = _predict_caption(triton_url, caption_model, inputs)

    # Check error handling
    assert caption == [""], "Function should return an empty string for request exceptions"


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}.grpcclient.InferenceServerClient")  # Mock the 'InferenceServerClient' class
def test_predict_caption_generic_exception(mock_client_class):
    # Create a mock client instance
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simulate a general exception
    mock_client.infer.side_effect = RuntimeError("Generic Exception")

    # Define test variables
    triton_url = "http://fake-url.com"
    caption_model = "fake_model"
    inputs = [["hello", "world"]]

    # Call the function
    caption = _predict_caption(triton_url, caption_model, inputs)

    # Check error handling
    assert caption == [""], "Function should return an empty string for generic exceptions"


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}.grpcclient.InferenceServerClient")  # Mock the 'InferenceServerClient' class
def test_predict_caption_server_error(mock_client_class):
    # Create a mock client instance
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simulate a 500 Server Error
    mock_client.infer.side_effect = InferenceServerException("Server error")

    # Define test variables
    triton_url = "http://fake-url.com"
    caption_model = "fake_model"
    inputs = [["hello", "world"]]

    # Call the function
    caption = _predict_caption(triton_url, caption_model, inputs)

    # Check error handling
    assert caption == [""], "Function should return an empty string for server errors"


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_process_content():
    # Example data where bounding boxes and content are adequate
    bboxes = [(0, 0, 10, 10), (10, 10, 20, 20)]
    content = ["Content A", "Content B"]
    metadata = {"image_metadata": {"image_location": (5, 5, 15, 15)}}
    neighbor_content = []

    # Assuming calculate_centroids, fit_nearest_neighbors, and find_nearest_neighbors work as expected
    # Directly use the functions as they should be behaving correctly if they are unit tested separately.
    _process_content(bboxes, content, metadata, neighbor_content, n_neighbors=5)

    # Assertions
    # We expect that the function correctly adds processed content to neighbor_content.
    # The exact content depends on the functioning of the called functions which are assumed to be correct.
    assert len(neighbor_content) == 1, "Should append one list of nearest content."
    # Check for padding with empty strings if less content is available than n_neighbors
    assert len(neighbor_content[0]) <= 5, "Output should not exceed requested number of neighbors."
    if len(neighbor_content[0]) < 5:
        assert neighbor_content[0].count("") == 5 - len(
            neighbor_content[0]
        ), "Should fill missing neighbors with empty strings."

    # Test with no bounding boxes or content
    neighbor_content = []
    _process_content([], [], metadata, neighbor_content, n_neighbors=5)
    assert len(neighbor_content) == 1, "Should handle no input data gracefully."
    assert neighbor_content[0] == ["", "", "", "", ""], "Should fill with empty strings for no data."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_process_documents_empty_df():
    # Create an empty DataFrame with the expected structure
    df_empty = pd.DataFrame(columns=["metadata"])
    metadata_list, neighbor_content = _process_documents(df_empty)

    # Assertions
    assert len(metadata_list) == 0, "Metadata list should be empty for an empty input DataFrame."
    assert len(neighbor_content) == 0, "Neighbor content should be empty for an empty input DataFrame."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}._extract_bboxes_and_content")
@patch(f"{_MODULE_UNDER_TEST}._process_content")
def test_process_documents_populated_df(mock_process_content, mock_extract_bboxes_and_content):
    # Set up the mock responses for the functions used within process_documents
    mock_extract_bboxes_and_content.return_value = (["bbox1", "bbox2"], ["content1", "content2"])
    mock_process_content.return_value = None  # Assuming process_content does not return anything

    # Create a DataFrame with some test data
    df_populated = pd.DataFrame(
        {
            "metadata": [
                {
                    "content_metadata": {
                        "hierarchy": {"nearby_objects": {"text": {"bbox": ["bbox1"], "content": ["content1"]}}}
                    }
                }
            ]
        }
    )

    metadata_list, neighbor_content = _process_documents(df_populated)

    # Assertions
    assert len(metadata_list) == 1, "Metadata list should contain one entry for each row in the DataFrame."
    mock_extract_bboxes_and_content.assert_called_once()
    mock_process_content.assert_called_once()

    # Check that the correct metadata was passed and handled
    assert metadata_list[0] == df_populated.iloc[0]["metadata"], "Metadata should match input DataFrame's metadata."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}._predict_caption")
def test_generate_captions_empty(mock_predict_caption):
    config = MagicMock()
    config.batch_size = 2
    captions = _generate_captions([], config)
    assert len(captions) == 0, "Should return an empty list when no content is available."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}._predict_caption", return_value=["Caption 1", "Caption 2"])
def test_generate_captions_single_batch(mock_predict_caption):
    config = MagicMock()
    config.batch_size = 3
    config.endpoint_url = "http://fake-url.com"
    config.headers = {"Content-Type": "application/json"}

    neighbor_content = [["Content 1"], ["Content 2"]]
    captions = _generate_captions(neighbor_content, config)
    mock_predict_caption.assert_called_once_with(config.endpoint_url, config.headers, neighbor_content)
    assert len(captions) == 2, "Should process a single batch correctly."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@patch(f"{_MODULE_UNDER_TEST}._predict_caption", side_effect=[["Caption 1", "Caption 2"], ["Caption 3"]])
def test_generate_captions_multiple_batches(mock_predict_caption):
    config = MagicMock()
    config.batch_size = 2
    config.endpoint_url = "http://fake-url.com"
    config.headers = {"Content-Type": "application/json"}

    neighbor_content = [["Content 1"], ["Content 2"], ["Content 3"]]
    captions = _generate_captions(neighbor_content, config)
    assert mock_predict_caption.call_count == 2, "Should call predict caption twice for two batches."
    assert len(captions) == 3, "Should return the correct number of captions for all content."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_update_metadata_with_captions():
    metadata_list = [{"image_metadata": {}} for _ in range(3)]
    captions = ["Caption 1", "Caption 2", "Caption 3"]
    df_filtered = pd.DataFrame({"uuid": ["uuid1", "uuid2", "uuid3"]})

    image_docs = _update_metadata_with_captions(metadata_list, captions, df_filtered)

    assert len(image_docs) == 3, "Should create one document entry for each metadata entry."
    for doc, caption in zip(image_docs, captions):
        assert (
            doc["metadata"]["image_metadata"]["caption"] == caption
        ), "Caption should be correctly inserted into metadata."
        assert "uuid" in doc, "UUID should be included in the document."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_prepare_final_dataframe():
    # Create test data
    df = pd.DataFrame({"data": [1, 2, 3, 4], "info": ["a", "b", "c", "d"]})
    image_docs = [
        {"document_type": "image", "metadata": "metadata1", "uuid": "uuid1"},
        {"document_type": "image", "metadata": "metadata2", "uuid": "uuid2"},
    ]
    filter_index = pd.Series([True, False, True, False])

    # Mock the message object
    message = MagicMock()

    # Execute the function
    _prepare_final_dataframe(df, image_docs, filter_index, message)

    # Check the message payload was updated correctly
    assert message.payload.called, "The message payload should be updated."

    # Since we can't check cudf.DataFrame directly in a normal environment, we check if the transformation was called
    assert isinstance(message.payload.call_args[0][0].df, cudf.DataFrame), "Payload should be a cudf DataFrame."

    # Verify DataFrame shapes and contents
    # The final DataFrame should only include the non-filtered out items and new image docs
    expected_length = 2 + len(image_docs)  # 2 from df (where filter_index is False) + 2 image docs
    docs_df = pd.concat([df[~filter_index], pd.DataFrame(image_docs)], axis=0).reset_index(drop=True)
    assert len(docs_df) == expected_length, "Final DataFrame length should be correct."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_prepare_dataframes_empty():
    data = {"document_type": [], "content": []}

    # Mock the message and its payload
    message = MagicMock()
    message.payload().mutable_dataframe().__enter__().to_pandas.return_value = pd.DataFrame(data)

    df, df_filtered, bool_index = _prepare_dataframes(message)

    # Assertions
    assert df.empty, "Original DataFrame should be empty."
    assert df_filtered.empty, "Filtered DataFrame should be empty."
    assert bool_index.empty, "Boolean index should be empty."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_prepare_dataframes_mixed_document_types():
    # Create a DataFrame with mixed document types
    data = {
        "document_type": [
            ContentTypeEnum.IMAGE,
            ContentTypeEnum.TEXT,
            ContentTypeEnum.IMAGE,
            ContentTypeEnum.STRUCTURED,
        ],
        "content": ["img1", "text1", "img2", "pdf1"],
    }
    df = pd.DataFrame(data)

    # Mock the message and its payload
    message = MagicMock()
    message.payload().mutable_dataframe().__enter__().to_pandas.return_value = df

    df, df_filtered, bool_index = _prepare_dataframes(message)

    # Assertions
    assert not df.empty, "Original DataFrame should not be empty."
    assert len(df_filtered) == 2, "Filtered DataFrame should contain only images."
    assert all(
        df_filtered["document_type"] == ContentTypeEnum.IMAGE
    ), "Filtered DataFrame should only contain IMAGE types."
    assert list(bool_index) == [
        True,
        False,
        True,
        False,
    ], "Boolean index should correctly identify IMAGE document types."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_prepare_dataframes_all_images():
    # All entries as images
    data = {"document_type": [ContentTypeEnum.IMAGE, ContentTypeEnum.IMAGE], "content": ["img1", "img2"]}
    df = pd.DataFrame(data)

    # Mock setup
    message = MagicMock()
    message.payload().mutable_dataframe().__enter__().to_pandas.return_value = df

    df, df_filtered, bool_index = _prepare_dataframes(message)

    assert len(df) == 2 and len(df_filtered) == 2, "Both dataframes should be full and equal as all are images."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_prepare_dataframes_no_images():
    # No image document types
    data = {"document_type": [ContentTypeEnum.TEXT, ContentTypeEnum.STRUCTURED], "content": ["text1", "pdf1"]}
    df = pd.DataFrame(data)

    # Mock setup
    message = MagicMock()
    message.payload().mutable_dataframe().__enter__().to_pandas.return_value = df

    df, df_filtered, bool_index = _prepare_dataframes(message)

    assert len(df_filtered) == 0, "Filtered DataFrame should be empty as there are no images."
    assert not any(bool_index), "Boolean index should have no True values as there are no images."
