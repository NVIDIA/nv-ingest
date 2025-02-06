from unittest.mock import MagicMock

import pytest
import pandas as pd

from nv_ingest.schemas.chart_extractor_schema import ChartExtractorConfigSchema
from nv_ingest.stages.nim.chart_extraction import _update_metadata, _create_clients
from nv_ingest.stages.nim.chart_extraction import _extract_chart_data

MODULE_UNDER_TEST = "nv_ingest.stages.nim.chart_extraction"


@pytest.fixture
def valid_chart_extractor_config():
    """
    Returns a ChartExtractorConfigSchema object with valid endpoints/protocols.
    This fixture can be adapted to your environment.
    """
    return ChartExtractorConfigSchema(
        auth_token="fake_token",
        cached_endpoints=("cached_grpc_url", "cached_http_url"),
        cached_infer_protocol="grpc",
        deplot_endpoints=("deplot_grpc_url", "deplot_http_url"),
        deplot_infer_protocol="http",
        paddle_endpoints=("paddle_grpc_url", "paddle_http_url"),
        paddle_infer_protocol="grpc",
        nim_batch_size=2,
        workers_per_progress_engine=5,
    )


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
    cached_mock = MagicMock()
    deplot_mock = MagicMock()
    trace_info = {}

    result = _update_metadata(
        base64_images=[],
        cached_client=cached_mock,
        deplot_client=deplot_mock,
        trace_info=trace_info,
        batch_size=1,
        worker_pool_size=1,
    )

    assert result == []
    cached_mock.infer.assert_not_called()
    deplot_mock.infer.assert_not_called()


def test_update_metadata_single_batch_single_worker(mocker):
    """
    Test a simple scenario with a small list of base64_images, batch_size=2,
    worker_pool_size=1. We verify that cached is called once per batch,
    and deplot is called once per image in that batch.
    """
    # Mock out the clients
    cached_mock = MagicMock()
    deplot_mock = MagicMock()

    # Suppose cached returns ["cached_res1", "cached_res2"] for 2 images
    cached_mock.infer.return_value = ["cached_res1", "cached_res2"]

    # Suppose deplot is called for each image, returning single string
    deplot_mock.infer.side_effect = [["deplot_res1"], ["deplot_res2"]]

    mock_join = mocker.patch(f"{MODULE_UNDER_TEST}.join_cached_and_deplot_output", side_effect=["joined_1", "joined_2"])

    base64_images = ["img1", "img2"]
    trace_info = {}

    result = _update_metadata(base64_images, cached_mock, deplot_mock, trace_info, batch_size=2, worker_pool_size=1)

    # We expect result => [("img1", "joined_1"), ("img2", "joined_2")]
    assert len(result) == 2
    assert result[0] == ("img1", "joined_1")
    assert result[1] == ("img2", "joined_2")

    # Check calls
    cached_mock.infer.assert_called_once_with(
        data={"base64_images": ["img1", "img2"]},
        model_name="cached",
        stage_name="chart_data_extraction",
        max_batch_size=2,
        trace_info=trace_info,
    )

    # deplot.infer called once per image
    assert deplot_mock.infer.call_count == 2
    deplot_mock.infer.assert_any_call(
        data={"base64_image": "img1"},
        model_name="deplot",
        stage_name="chart_data_extraction",
        max_batch_size=1,
        trace_info=trace_info,
    )
    deplot_mock.infer.assert_any_call(
        data={"base64_image": "img2"},
        model_name="deplot",
        stage_name="chart_data_extraction",
        max_batch_size=1,
        trace_info=trace_info,
    )

    # join_cached_and_deplot_output called twice
    assert mock_join.call_count == 2


def test_update_metadata_multiple_batches_multi_worker(mocker):
    """
    If batch_size=1 but we have multiple images, each image forms its own batch.
    We also can use worker_pool_size=2 for parallel calls.
    """
    cached_mock = MagicMock()
    deplot_mock = MagicMock()
    mock_join = mocker.patch(
        f"{MODULE_UNDER_TEST}.join_cached_and_deplot_output", side_effect=["joined_1", "joined_2", "joined_3"]
    )

    # Suppose every cached.infer call returns a 1-element list
    def cached_side_effect(**kwargs):
        images = kwargs["data"]["base64_images"]
        return [f"cached_{images[0]}"]

    cached_mock.infer.side_effect = cached_side_effect

    # Suppose deplot.infer returns e.g. ["deplot_img1"], etc.
    def deplot_side_effect(**kwargs):
        img = kwargs["data"]["base64_image"]
        return [f"deplot_{img}"]

    deplot_mock.infer.side_effect = deplot_side_effect

    base64_images = ["imgA", "imgB", "imgC"]
    trace_info = {}

    result = _update_metadata(
        base64_images,
        cached_mock,
        deplot_mock,
        trace_info,
        batch_size=1,  # each image in its own batch
        worker_pool_size=2,
    )

    # Expect 3 results: [("imgA", "joined_1"), ("imgB", "joined_2"), ("imgC", "joined_3")]
    assert result == [("imgA", "joined_1"), ("imgB", "joined_2"), ("imgC", "joined_3")]

    # We should have 3 calls to cached.infer, each with one image
    assert cached_mock.infer.call_count == 3
    # Also 3 calls to deplot.infer
    assert deplot_mock.infer.call_count == 3
    # 3 calls to join
    assert mock_join.call_count == 3


def test_update_metadata_exception_in_cached_call(caplog):
    """
    If the cached call fails for a batch, we expect an exception to bubble up
    and the error logged.
    """
    cached_mock = MagicMock()
    deplot_mock = MagicMock()
    cached_mock.infer.side_effect = Exception("Cached call error")

    with pytest.raises(Exception, match="Cached call error"):
        _update_metadata(["some_img"], cached_mock, deplot_mock, trace_info={}, batch_size=1, worker_pool_size=1)

    # Check log
    assert "Error processing batch: ['some_img']" in caplog.text


def test_update_metadata_exception_in_deplot_call(caplog):
    """
    If any deplot call fails for one of the images, the entire process fails
    and logs the error.
    """
    cached_mock = MagicMock()
    cached_mock.infer.return_value = ["cached_result"]  # 1-element list
    deplot_mock = MagicMock()
    deplot_mock.infer.side_effect = Exception("Deplot error")

    with pytest.raises(Exception, match="Deplot error"):
        _update_metadata(["some_img"], cached_mock, deplot_mock, trace_info={}, batch_size=1, worker_pool_size=2)

    assert "Error processing batch: ['some_img']" in caplog.text


def test_create_clients(mocker):
    """
    Verify that _create_clients calls create_inference_client for
    both cached and deplot endpoints, returning the pair of NimClient mocks.
    """
    mock_create_inference_client = mocker.patch(f"{MODULE_UNDER_TEST}.create_inference_client")

    # Suppose it returns different mocks each time
    cached_mock = MagicMock()
    deplot_mock = MagicMock()
    mock_create_inference_client.side_effect = [cached_mock, deplot_mock]

    result = _create_clients(
        cached_endpoints=("cached_grpc", "cached_http"),
        cached_protocol="grpc",
        deplot_endpoints=("deplot_grpc", "deplot_http"),
        deplot_protocol="http",
        auth_token="xyz",
    )

    # result => (cached_mock, deplot_mock)
    assert result == (cached_mock, deplot_mock)

    # Check calls
    assert mock_create_inference_client.call_count == 2

    mock_create_inference_client.assert_any_call(
        endpoints=("cached_grpc", "cached_http"), model_interface=mocker.ANY, auth_token="xyz", infer_protocol="grpc"
    )
    mock_create_inference_client.assert_any_call(
        endpoints=("deplot_grpc", "deplot_http"), model_interface=mocker.ANY, auth_token="xyz", infer_protocol="http"
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
    cached_mock, deplot_mock = MagicMock(), MagicMock()
    mock_create_clients = mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=(cached_mock, deplot_mock))

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
        validated_config.stage_config.cached_endpoints,
        validated_config.stage_config.cached_infer_protocol,
        validated_config.stage_config.deplot_endpoints,
        validated_config.stage_config.deplot_infer_protocol,
        validated_config.stage_config.auth_token,
    )

    # Check _update_metadata call
    mock_update_metadata.assert_called_once_with(
        base64_images=["imgA", "imgB"],
        cached_client=cached_mock,
        deplot_client=deplot_mock,
        batch_size=validated_config.stage_config.nim_batch_size,
        worker_pool_size=validated_config.stage_config.workers_per_progress_engine,
        trace_info=ti.get("trace_info"),
    )


def test_extract_chart_data_mixed_rows(validated_config, mocker):
    """
    Some rows are valid, some not. We only pass valid images to _update_metadata,
    and only those rows get updated.
    """
    cached_mock, deplot_mock = MagicMock(), MagicMock()
    mocker.patch(f"{MODULE_UNDER_TEST}._create_clients", return_value=(cached_mock, deplot_mock))

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
        cached_client=cached_mock,
        deplot_client=deplot_mock,
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
