from unittest.mock import MagicMock

import pytest
import pandas as pd
import numpy as np

from nv_ingest.schemas.infographic_extractor_schema import InfographicExtractorConfigSchema
from nv_ingest.stages.nim.infographic_extraction import _update_metadata, _create_clients
from nv_ingest.stages.nim.infographic_extraction import _extract_infographic_data

MODULE_UNDER_TEST = "nv_ingest.stages.nim.infographic_extraction"


@pytest.fixture
def valid_infographic_extractor_config():
    """
    Returns a InfographicExtractorConfigSchema object with valid endpoints/protocols.
    This fixture can be adapted to your environment.
    """
    return InfographicExtractorConfigSchema(
        auth_token="fake_token",
        paddle_endpoints=("paddle_grpc_url", "paddle_http_url"),
        paddle_infer_protocol="grpc",
        workers_per_progress_engine=5,
    )


@pytest.fixture
def base64_image():
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


@pytest.fixture
def validated_config(valid_infographic_extractor_config):
    """
    If your code references validated_config.stage_config,
    we can make a simple object that has an attribute 'stage_config'
    pointing to our infographic config.
    """

    class FakeValidated:
        stage_config = valid_infographic_extractor_config

    return FakeValidated()


def test_update_metadata_empty_list():
    """
    If the base64_images list is empty, _update_metadata should return an empty list.
    With the updated implementation, both clients are still invoked (with an empty list)
    so we set their return values to [] and then verify the calls.
    """
    paddle_mock = MagicMock()
    trace_info = {}

    # When given an empty list, both clients return an empty list.
    paddle_mock.infer.return_value = []

    result = _update_metadata(
        base64_images=[],
        paddle_client=paddle_mock,
        worker_pool_size=1,
        trace_info=trace_info,
    )

    assert result == []

    # infer should be called once with an empty list.
    paddle_mock.infer.assert_called_once_with(
        data={"base64_images": []},
        model_name="paddle",
        stage_name="infographic_data_extraction",
        max_batch_size=2,
        trace_info=trace_info,
    )


def test_update_metadata_single_batch_single_worker(mocker, base64_image):
    """
    Test a simple scenario with a small list of base64_images using worker_pool_size=1.
    In the updated _update_metadata implementation, the paddle client is called once
    with the full list of images.
    """
    # Patch base64_to_numpy to simulate a valid image (e.g., 100x100 with 3 channels)
    mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy", return_value=np.ones((100, 100, 3)))

    # Mock out the clients.
    paddle_mock = MagicMock()
    # Set paddle protocol so that max_batch_size becomes 2 (non-grpc).
    paddle_mock.protocol = "http"

    # Simulate paddle returning bounding boxes and text predictions for two images.
    paddle_mock.infer.return_value = [([(0, 1, 2, 3)], ["paddle_res1"]), ([(4, 5, 6, 7)], ["paddle_res2"])]

    base64_images = [base64_image, base64_image]
    trace_info = {}

    result = _update_metadata(base64_images, paddle_mock, worker_pool_size=1, trace_info=trace_info)

    # Expect the result to combine each original image with its corresponding output.
    assert len(result) == 2
    assert result[0] == (base64_image, [(0, 1, 2, 3)], ["paddle_res1"])
    assert result[1] == (base64_image, [(4, 5, 6, 7)], ["paddle_res2"])

    # Verify that paddle.infer is called once with the full list of original images.
    paddle_mock.infer.assert_called_once_with(
        data={"base64_images": [base64_image, base64_image]},
        model_name="paddle",
        stage_name="infographic_data_extraction",
        max_batch_size=2,
        trace_info=trace_info,
    )


def test_update_metadata_multiple_batches_multi_worker(mocker, base64_image):
    """
    With the new _update_metadata implementation, both cached_client.infer and deplot_client.infer
    are called once with the full list of images. Their results are expected to be lists with one
    item per image.
    """
    # Patch base64_to_numpy to simulate valid images (e.g., 100x100 with 3 channels)
    mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy", return_value=np.ones((100, 100, 3)))

    paddle_mock = MagicMock()

    # Define a similar side effect for paddle.infer.
    def paddle_side_effect(**kwargs):
        base64_images_list = kwargs["data"]["base64_images"]
        return [([(i, i + 1, i + 2, i + 3)], [f"paddle_result_{i+1}"]) for i in range(len(base64_images_list))]

    paddle_mock.infer.side_effect = paddle_side_effect

    base64_images = [base64_image, base64_image, base64_image]
    trace_info = {}

    result = _update_metadata(
        base64_images,
        paddle_mock,
        worker_pool_size=2,
        trace_info=trace_info,
    )

    expected = [
        (base64_image, [(0, 1, 2, 3)], ["paddle_result_1"]),
        (base64_image, [(1, 2, 3, 4)], ["paddle_result_2"]),
        (base64_image, [(2, 3, 4, 5)], ["paddle_result_3"]),
    ]
    assert result == expected

    # infer method was called only once.
    assert paddle_mock.infer.call_count == 1


def test_update_metadata_exception_in_paddle_call(mocker, base64_image, caplog):
    """
    If the paddle call fails, we expect an exception to bubble up and the error to be logged.
    """
    # Ensure the image passes the filtering by patching base64_to_numpy to return a valid image array.
    mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy", return_value=np.ones((100, 100, 3)))

    paddle_mock = MagicMock()
    paddle_mock.infer.side_effect = Exception("Paddle error")

    with pytest.raises(Exception, match="Paddle error"):
        _update_metadata([base64_image], paddle_mock, trace_info={}, worker_pool_size=2)

    assert "Error calling paddle_client.infer: Paddle error" in caplog.text


def test_create_clients(mocker):
    """
    Verify that _create_clients calls create_inference_client for the paddle endpoint,
    returning the NimClient mock.
    """
    mock_create_inference_client = mocker.patch(f"{MODULE_UNDER_TEST}.create_inference_client")

    # Suppose it returns different mocks each time
    paddle_mock = MagicMock()
    mock_create_inference_client.side_effect = [paddle_mock]

    result = _create_clients(
        paddle_endpoints=("paddle_grpc", "paddle_http"),
        paddle_protocol="http",
        auth_token="xyz",
    )

    assert result == paddle_mock

    # Check calls
    assert mock_create_inference_client.call_count == 1

    mock_create_inference_client.assert_any_call(
        endpoints=("paddle_grpc", "paddle_http"), model_interface=mocker.ANY, auth_token="xyz", infer_protocol="http"
    )


def test_extract_infographic_data_empty_df(validated_config, mocker):
    """
    If df is empty, we just return df + trace_info as is; no calls to clients.
    """
    mock_create_clients = mocker.patch(f"{MODULE_UNDER_TEST}._create_clients")
    mock_update_metadata = mocker.patch(f"{MODULE_UNDER_TEST}._update_metadata")

    empty_df = pd.DataFrame()

    df_out, ti = _extract_infographic_data(empty_df, {}, validated_config)
    assert df_out.empty
    assert ti == {}

    mock_create_clients.assert_not_called()
    mock_update_metadata.assert_not_called()


def test_extract_infographic_data_no_valid_rows(validated_config, mocker):
    """
    A DataFrame with rows that do not meet the 'structured/infographic' criteria
    => skip everything, return df unchanged, no calls to _update_metadata.
    """
    mock_create = mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=MagicMock())
    mock_update = mocker.patch(f"{MODULE_UNDER_TEST}._update_metadata")

    df_in = pd.DataFrame(
        [
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "table"},
                    "table_metadata": {},
                    "content": "some_img",
                }
            },
            {"metadata": None},
        ]
    )
    df_out, trace_info = _extract_infographic_data(df_in, {}, validated_config)

    assert df_out.equals(df_in), "No changes should be made"
    assert "trace_info" in trace_info

    mock_create.assert_called_once()  # We still create clients
    mock_update.assert_not_called()


def test_extract_infographic_data_all_valid(validated_config, mocker):
    """
    All rows meet criteria => pass them all to _update_metadata in order.
    """
    # Mock out clients
    paddle_mock = MagicMock()

    mock_create_clients = mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=paddle_mock)

    # Suppose _update_metadata returns infographic content for each image
    mock_update_metadata = mocker.patch(
        f"{MODULE_UNDER_TEST}._update_metadata",
        return_value=[("imgA", [()], ["contentA"]), ("imgB", [()], ["contentB"])],
    )

    # Build a DataFrame with 2 valid rows
    df_in = pd.DataFrame(
        [
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "infographic"},
                    "table_metadata": {},
                    "content": "imgA",
                }
            },
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "infographic"},
                    "table_metadata": {},
                    "content": "imgB",
                }
            },
        ]
    )

    # Extract
    df_out, ti = _extract_infographic_data(df_in, {}, validated_config)
    assert df_out.at[0, "metadata"]["table_metadata"]["table_content"] == "contentA"
    assert df_out.at[1, "metadata"]["table_metadata"]["table_content"] == "contentB"

    mock_create_clients.assert_called_once_with(
        validated_config.stage_config.paddle_endpoints,
        validated_config.stage_config.paddle_infer_protocol,
        validated_config.stage_config.auth_token,
    )

    # Check _update_metadata call
    mock_update_metadata.assert_called_once_with(
        base64_images=["imgA", "imgB"],
        paddle_client=paddle_mock,
        worker_pool_size=validated_config.stage_config.workers_per_progress_engine,
        trace_info=ti.get("trace_info"),
    )


def test_extract_infographic_data_mixed_rows(validated_config, mocker):
    """
    Some rows are valid, some not. We only pass valid images to _update_metadata,
    and only those rows get updated.
    """
    paddle_mock = MagicMock()

    mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=paddle_mock)

    mock_update = mocker.patch(
        f"{MODULE_UNDER_TEST}._update_metadata",
        return_value=[
            ("base64img1", [()], ["stuff1"]),
            ("base64img2", [()], ["stuff2"]),
        ],
    )

    df_in = pd.DataFrame(
        [
            {  # valid row
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "infographic"},
                    "table_metadata": {},
                    "content": "base64img1",
                }
            },
            {  # invalid row
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "table"},
                    "table_metadata": {},
                    "content": "whatever",
                }
            },
            {  # valid row
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "infographic"},
                    "table_metadata": {},
                    "content": "base64img2",
                }
            },
        ]
    )

    df_out, trace = _extract_infographic_data(df_in, {}, validated_config)

    assert df_out.at[0, "metadata"]["table_metadata"]["table_content"] == "stuff1"
    # row1 => no update
    assert "table_content" not in df_out.at[1, "metadata"]["table_metadata"]
    assert df_out.at[2, "metadata"]["table_metadata"]["table_content"] == "stuff2"

    mock_update.assert_called_once_with(
        base64_images=["base64img1", "base64img2"],
        paddle_client=paddle_mock,
        worker_pool_size=validated_config.stage_config.workers_per_progress_engine,
        trace_info=trace.get("trace_info"),
    )


def test_extract_infographic_data_exception_raised(validated_config, mocker):
    """
    If something goes wrong, we still close the clients and re-raise the exception.
    """
    c_mock = MagicMock()
    mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=c_mock)

    # Suppose _update_metadata raises an exception
    mocker.patch(f"{MODULE_UNDER_TEST}._update_metadata", side_effect=RuntimeError("Test error"))

    df_in = pd.DataFrame(
        [
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "infographic"},
                    "table_metadata": {},
                    "content": "imgZ",
                }
            }
        ]
    )

    with pytest.raises(RuntimeError, match="Test error"):
        _extract_infographic_data(df_in, {}, validated_config)

    c_mock.close.assert_called_once()
