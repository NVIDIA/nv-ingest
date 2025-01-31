from unittest.mock import MagicMock

import pytest
import pandas as pd
import numpy as np

from nv_ingest.schemas.chart_extractor_schema import ChartExtractorConfigSchema
from nv_ingest.stages.nim.chart_extraction import _update_metadata, _create_clients
from nv_ingest.stages.nim.chart_extraction import _extract_chart_data
from nv_ingest.util.image_processing.transforms import base64_to_numpy

MODULE_UNDER_TEST = "nv_ingest.stages.nim.chart_extraction"


@pytest.fixture
def valid_chart_extractor_config():
    """
    Returns a ChartExtractorConfigSchema object with valid endpoints/protocols.
    This fixture can be adapted to your environment.
    """
    return ChartExtractorConfigSchema(
        auth_token="fake_token",
        yolox_endpoints=("yolox_grpc_url", "yolox_http_url"),
        yolox_infer_protocol="grpc",
        paddle_endpoints=("paddle_grpc_url", "paddle_http_url"),
        paddle_infer_protocol="grpc",
        nim_batch_size=2,
        workers_per_progress_engine=5,
    )


@pytest.fixture
def base64_image():
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


@pytest.fixture
def validated_config(valid_chart_extractor_config):
    """
    If your code references validated_config.stage_config,
    we can make a simple object that has an attribute 'stage_config'
    pointing to our chart config.
    """

    class FakeValidated:
        stage_config = valid_chart_extractor_config

    return FakeValidated()


def test_update_metadata_empty_list():
    """
    If the base64_images list is empty, _update_metadata should
    skip all logic and return an empty list.
    """
    yolox_mock = MagicMock()
    paddle_mock = MagicMock()
    trace_info = {}

    result = _update_metadata(
        base64_images=[],
        yolox_client=yolox_mock,
        paddle_client=paddle_mock,
        trace_info=trace_info,
        batch_size=1,
        worker_pool_size=1,
    )

    assert result == []
    yolox_mock.infer.assert_not_called()


def test_update_metadata_single_batch_single_worker(mocker, base64_image):
    """
    Test a simple scenario with a small list of base64_images, batch_size=2,
    worker_pool_size=1. We verify that yolox is called once per batch,
    and paddle is called once per image in that batch.
    """
    # Mock out the clients
    yolox_mock = MagicMock()
    paddle_mock = MagicMock()

    # Suppose yolox returns ["yolox_res1", "yolox_res2"] for 2 images
    yolox_mock.infer.return_value = ["yolox_res1", "yolox_res2"]

    # Suppose paddle returns ["paddle_res1", "paddle_res2"] for 2 images
    paddle_mock.infer.return_value = [[(), "paddle_res1"], [(), "paddle_res2"]]

    mock_join = mocker.patch(
        f"{MODULE_UNDER_TEST}.join_yolox_and_paddle_output",
        side_effect=[{"chart_title": "joined_1"}, {"chart_title": "joined_2"}],
    )

    base64_images = [
        base64_image,
        base64_image,
    ]
    trace_info = {}

    result = _update_metadata(base64_images, yolox_mock, paddle_mock, trace_info, batch_size=2, worker_pool_size=1)

    # We expect result => [("img1", "joined_1"), ("img2", "joined_2")]
    assert len(result) == 2
    assert result[0] == (base64_image, "joined_1")
    assert result[1] == (base64_image, "joined_2")

    # Check calls
    assert yolox_mock.infer.call_count == 1
    assert np.all(yolox_mock.infer.call_args.kwargs["data"]["images"][0] == base64_to_numpy(base64_image))
    assert np.all(yolox_mock.infer.call_args.kwargs["data"]["images"][1] == base64_to_numpy(base64_image))
    assert yolox_mock.infer.call_args.kwargs["model_name"] == "yolox"
    assert yolox_mock.infer.call_args.kwargs["stage_name"] == "chart_data_extraction"
    assert yolox_mock.infer.call_args.kwargs["trace_info"] == trace_info

    # paddle.infer called once per image
    assert paddle_mock.infer.call_count == 1
    paddle_mock.infer.assert_any_call(
        data={"base64_images": [base64_image, base64_image]},
        model_name="paddle",
        stage_name="chart_data_extraction",
        trace_info=trace_info,
    )

    # join_yolox_and_paddle_output called twice
    assert mock_join.call_count == 2


def test_update_metadata_multiple_batches_multi_worker(mocker, base64_image):
    """
    If batch_size=1 but we have multiple images, each image forms its own batch.
    We also can use worker_pool_size=2 for parallel calls.
    """
    yolox_mock = MagicMock()
    paddle_mock = MagicMock()
    mock_join = mocker.patch(
        f"{MODULE_UNDER_TEST}.join_yolox_and_paddle_output",
        side_effect=[{"chart_title": "joined_1"}, {"chart_title": "joined_2"}, {"chart_title": "joined_3"}],
    )

    # Suppose every yolox.infer call returns a 1-element list
    def yolox_side_effect(**kwargs):
        images = kwargs["data"]["images"]
        return [f"yolox_{images[0]}"]

    yolox_mock.infer.side_effect = yolox_side_effect

    # Suppose paddle.infer returns e.g. ["paddle_img1"], etc.
    def paddle_side_effect(**kwargs):
        img = kwargs["data"]["base64_images"]
        return [([], f"paddle_{img}")]

    paddle_mock.infer.side_effect = paddle_side_effect

    base64_images = [base64_image, base64_image, base64_image]
    trace_info = {}

    result = _update_metadata(
        base64_images,
        yolox_mock,
        paddle_mock,
        trace_info,
        batch_size=1,  # each image in its own batch
        worker_pool_size=2,
    )

    # Expect 3 results: [("imgA", "joined_1"), ("imgB", "joined_2"), ("imgC", "joined_3")]
    assert result == [(base64_image, "joined_1"), (base64_image, "joined_2"), (base64_image, "joined_3")]

    # We should have 3 calls to yolox.infer, each with one image
    assert yolox_mock.infer.call_count == 3
    # Also 3 calls to paddle.infer
    assert paddle_mock.infer.call_count == 3
    # 3 calls to join
    assert mock_join.call_count == 3


def test_update_metadata_exception_in_yolox_call(base64_image, caplog):
    """
    If the yolox call fails for a batch, we expect an exception to bubble up
    and the error logged.
    """
    yolox_mock = MagicMock()
    paddle_mock = MagicMock()
    yolox_mock.infer.side_effect = Exception("Yolox call error")

    with pytest.raises(Exception, match="Yolox call error"):
        _update_metadata([base64_image], yolox_mock, paddle_mock, trace_info={}, batch_size=1, worker_pool_size=1)

    # Check log
    assert f"Error processing batch: ['{base64_image}']" in caplog.text


def test_update_metadata_exception_in_paddle_call(base64_image, caplog):
    """
    If any paddle call fails for one of the images, the entire process fails
    and logs the error.
    """
    yolox_mock = MagicMock()
    yolox_mock.infer.return_value = ["yolox_result"]  # 1-element list
    paddle_mock = MagicMock()
    paddle_mock.infer.side_effect = Exception("Paddle error")

    with pytest.raises(Exception, match="Paddle error"):
        _update_metadata([base64_image], yolox_mock, paddle_mock, trace_info={}, batch_size=1, worker_pool_size=2)

    assert f"Error processing batch: ['{base64_image}']" in caplog.text


def test_create_clients(mocker):
    """
    Verify that _create_clients calls create_inference_client for
    both yolox and paddle endpoints, returning the pair of NimClient mocks.
    """
    mock_create_inference_client = mocker.patch(f"{MODULE_UNDER_TEST}.create_inference_client")

    # Suppose it returns different mocks each time
    yolox_mock = MagicMock()
    paddle_mock = MagicMock()
    mock_create_inference_client.side_effect = [yolox_mock, paddle_mock]

    result = _create_clients(
        yolox_endpoints=("yolox_grpc", "yolox_http"),
        yolox_protocol="grpc",
        paddle_endpoints=("paddle_grpc", "paddle_http"),
        paddle_protocol="http",
        auth_token="xyz",
    )

    # result => (yolox_mock, paddle_mock)
    assert result == (yolox_mock, paddle_mock)

    # Check calls
    assert mock_create_inference_client.call_count == 2

    mock_create_inference_client.assert_any_call(
        endpoints=("yolox_grpc", "yolox_http"), model_interface=mocker.ANY, auth_token="xyz", infer_protocol="grpc"
    )
    mock_create_inference_client.assert_any_call(
        endpoints=("paddle_grpc", "paddle_http"), model_interface=mocker.ANY, auth_token="xyz", infer_protocol="http"
    )


def test_extract_chart_data_empty_df(validated_config, mocker):
    """
    If df is empty, we just return df + trace_info as is; no calls to clients.
    """
    mock_create_clients = mocker.patch(f"{MODULE_UNDER_TEST}._create_clients")
    mock_update_metadata = mocker.patch(f"{MODULE_UNDER_TEST}._update_metadata")

    empty_df = pd.DataFrame()

    df_out, ti = _extract_chart_data(empty_df, {}, validated_config)
    assert df_out.empty
    assert ti == {}

    mock_create_clients.assert_not_called()
    mock_update_metadata.assert_not_called()


def test_extract_chart_data_no_valid_rows(validated_config, mocker):
    """
    A DataFrame with rows that do not meet the 'structured/chart' criteria
    => skip everything, return df unchanged, no calls to _update_metadata.
    """
    mock_create = mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=(MagicMock(), MagicMock()))
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
    df_out, trace_info = _extract_chart_data(df_in, {}, validated_config)

    assert df_out.equals(df_in), "No changes should be made"
    assert "trace_info" in trace_info

    mock_create.assert_called_once()  # We still create clients
    mock_update.assert_not_called()


def test_extract_chart_data_all_valid(validated_config, mocker):
    """
    All rows meet criteria => pass them all to _update_metadata in order.
    """
    # Mock out clients
    yolox_mock, paddle_mock = MagicMock(), MagicMock()
    mock_create_clients = mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=(yolox_mock, paddle_mock))

    # Suppose _update_metadata returns chart content for each image
    mock_update_metadata = mocker.patch(
        f"{MODULE_UNDER_TEST}._update_metadata",
        return_value=[("imgA", {"joined": "contentA"}), ("imgB", {"joined": "contentB"})],
    )

    # Build a DataFrame with 2 valid rows
    df_in = pd.DataFrame(
        [
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "chart"},
                    "table_metadata": {},
                    "content": "imgA",
                }
            },
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "chart"},
                    "table_metadata": {},
                    "content": "imgB",
                }
            },
        ]
    )

    # Extract
    df_out, ti = _extract_chart_data(df_in, {}, validated_config)
    assert df_out.at[0, "metadata"]["table_metadata"]["table_content"] == {"joined": "contentA"}
    assert df_out.at[1, "metadata"]["table_metadata"]["table_content"] == {"joined": "contentB"}

    mock_create_clients.assert_called_once_with(
        validated_config.stage_config.yolox_endpoints,
        validated_config.stage_config.yolox_infer_protocol,
        validated_config.stage_config.paddle_endpoints,
        validated_config.stage_config.paddle_infer_protocol,
        validated_config.stage_config.auth_token,
    )

    # Check _update_metadata call
    mock_update_metadata.assert_called_once_with(
        base64_images=["imgA", "imgB"],
        yolox_client=yolox_mock,
        paddle_client=paddle_mock,
        batch_size=validated_config.stage_config.nim_batch_size,
        worker_pool_size=validated_config.stage_config.workers_per_progress_engine,
        trace_info=ti.get("trace_info"),
    )


def test_extract_chart_data_mixed_rows(validated_config, mocker):
    """
    Some rows are valid, some not. We only pass valid images to _update_metadata,
    and only those rows get updated.
    """
    yolox_mock, paddle_mock = MagicMock(), MagicMock()
    mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=(yolox_mock, paddle_mock))

    mock_update = mocker.patch(
        f"{MODULE_UNDER_TEST}._update_metadata",
        return_value=[
            ("base64img1", {"chart": "stuff1"}),
            ("base64img2", {"chart": "stuff2"}),
        ],
    )

    df_in = pd.DataFrame(
        [
            {  # valid row
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "chart"},
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
                    "content_metadata": {"type": "structured", "subtype": "chart"},
                    "table_metadata": {},
                    "content": "base64img2",
                }
            },
        ]
    )

    df_out, trace = _extract_chart_data(df_in, {}, validated_config)

    assert df_out.at[0, "metadata"]["table_metadata"]["table_content"] == {"chart": "stuff1"}
    # row1 => no update
    assert "table_content" not in df_out.at[1, "metadata"]["table_metadata"]
    assert df_out.at[2, "metadata"]["table_metadata"]["table_content"] == {"chart": "stuff2"}

    mock_update.assert_called_once_with(
        base64_images=["base64img1", "base64img2"],
        yolox_client=yolox_mock,
        paddle_client=paddle_mock,
        batch_size=validated_config.stage_config.nim_batch_size,
        worker_pool_size=validated_config.stage_config.workers_per_progress_engine,
        trace_info=trace.get("trace_info"),
    )


def test_extract_chart_data_exception_raised(validated_config, mocker):
    """
    If something goes wrong, we still close the clients and re-raise the exception.
    """
    c_mock, d_mock = MagicMock(), MagicMock()
    mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=(c_mock, d_mock))

    # Suppose _update_metadata raises an exception
    mocker.patch(f"{MODULE_UNDER_TEST}._update_metadata", side_effect=RuntimeError("Test error"))

    df_in = pd.DataFrame(
        [
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "chart"},
                    "table_metadata": {},
                    "content": "imgZ",
                }
            }
        ]
    )

    with pytest.raises(RuntimeError, match="Test error"):
        _extract_chart_data(df_in, {}, validated_config)

    c_mock.close.assert_called_once()
    d_mock.close.assert_called_once()
