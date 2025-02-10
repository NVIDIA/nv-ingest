import base64
from io import BytesIO
from unittest.mock import Mock, MagicMock
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nv_ingest.stages.nim.table_extraction import _extract_table_data
from nv_ingest.stages.nim.table_extraction import _update_metadata
from nv_ingest.stages.nim.table_extraction import PADDLE_MIN_WIDTH, PADDLE_MIN_HEIGHT
from nv_ingest.util.nim.helpers import NimClient
from nv_ingest.util.nim.paddle import PaddleOCRModelInterface

MODULE_UNDER_TEST = "nv_ingest.stages.nim.table_extraction"


@pytest.fixture
def paddle_mock():
    """
    Fixture that returns a MagicMock for the paddle_client,
    which we'll pass to _update_metadata.
    """
    return MagicMock()


@pytest.fixture
def validated_config():
    """
    Fixture that returns a minimal validated_config object
    with a `stage_config` containing the necessary fields.
    """

    class FakeStageConfig:
        # Values that _extract_table_data expects
        workers_per_progress_engine = 5
        auth_token = "fake-token"
        # For _create_paddle_client
        paddle_endpoints = ("grpc_url", "http_url")
        paddle_infer_protocol = "grpc"

    class FakeValidatedConfig:
        stage_config = FakeStageConfig()

    return FakeValidatedConfig()


def test_extract_table_data_empty_df(mocker, validated_config):
    """
    If df is empty, return the df + an empty trace_info without creating a client or calling _update_metadata.
    """
    mock_create_client = mocker.patch(f"{MODULE_UNDER_TEST}._create_paddle_client")
    mock_update_metadata = mocker.patch(f"{MODULE_UNDER_TEST}._update_metadata")

    df_in = pd.DataFrame()

    df_out, trace_info = _extract_table_data(df_in, {}, validated_config)
    assert df_out.empty
    assert trace_info == {}
    mock_create_client.assert_not_called()
    mock_update_metadata.assert_not_called()


def test_extract_table_data_no_valid_rows(mocker, validated_config):
    """
    Rows exist, but none meet the "structured/table" criteria =>
    skip _update_metadata, still create/close the client,
    and return the DataFrame unmodified with a trace_info.
    """
    mock_client = MagicMock()
    mock_create_client = mocker.patch(f"{MODULE_UNDER_TEST}._create_paddle_client", return_value=mock_client)
    mock_update_metadata = mocker.patch(f"{MODULE_UNDER_TEST}._update_metadata")

    df_in = pd.DataFrame(
        [
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "chart"},
                    "table_metadata": {},
                    "content": "some_base64",
                }
            },
            {"metadata": None},  # also invalid
        ]
    )

    df_out, trace_info = _extract_table_data(df_in, {}, validated_config)
    assert df_out.equals(df_in)
    assert "trace_info" in trace_info
    mock_create_client.assert_called_once()  # We do create a client
    mock_update_metadata.assert_not_called()  # But never call _update_metadata
    mock_client.close.assert_called_once()  # Must close client


def test_extract_table_data_all_valid(mocker, validated_config):
    """
    All rows are valid => we pass all base64 images to _update_metadata once,
    then write the returned content/format back into each row.
    """
    mock_client = MagicMock()
    mock_create_client = mocker.patch(f"{MODULE_UNDER_TEST}._create_paddle_client", return_value=mock_client)
    mock_update_metadata = mocker.patch(
        f"{MODULE_UNDER_TEST}._update_metadata",
        return_value=[
            ("imgA", ("tableA", "fmtA")),
            ("imgB", ("tableB", "fmtB")),
        ],
    )

    df_in = pd.DataFrame(
        [
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "table"},
                    "table_metadata": {},
                    "content": "imgA",
                }
            },
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "table"},
                    "table_metadata": {},
                    "content": "imgB",
                }
            },
        ]
    )

    df_out, trace_info = _extract_table_data(df_in, {}, validated_config)

    # Each valid row updated
    assert df_out.at[0, "metadata"]["table_metadata"]["table_content"] == "tableA"
    assert df_out.at[0, "metadata"]["table_metadata"]["table_content_format"] == "fmtA"
    assert df_out.at[1, "metadata"]["table_metadata"]["table_content"] == "tableB"
    assert df_out.at[1, "metadata"]["table_metadata"]["table_content_format"] == "fmtB"

    # Check calls
    mock_create_client.assert_called_once()
    mock_update_metadata.assert_called_once_with(
        base64_images=["imgA", "imgB"],
        paddle_client=mock_client,
        worker_pool_size=validated_config.stage_config.workers_per_progress_engine,
        trace_info=trace_info.get("trace_info"),
    )
    mock_client.close.assert_called_once()


def test_extract_table_data_mixed_rows(mocker, validated_config):
    """
    Some rows valid, some invalid => only valid rows get updated.
    """
    mock_client = MagicMock()
    mock_create_client = mocker.patch(f"{MODULE_UNDER_TEST}._create_paddle_client", return_value=mock_client)
    mock_update_metadata = mocker.patch(
        f"{MODULE_UNDER_TEST}._update_metadata",
        return_value=[("good1", ("table1", "fmt1")), ("good2", ("table2", "fmt2"))],
    )

    df_in = pd.DataFrame(
        [
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "table"},
                    "table_metadata": {},
                    "content": "good1",
                }
            },
            {
                # invalid => subtype=chart
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "chart"},
                    "table_metadata": {},
                    "content": "chart_b64",
                }
            },
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "table"},
                    "table_metadata": {},
                    "content": "good2",
                }
            },
        ]
    )

    df_out, trace_info = _extract_table_data(df_in, {}, validated_config)

    # row0 => updated with table1/fmt1
    assert df_out.at[0, "metadata"]["table_metadata"]["table_content"] == "table1"
    assert df_out.at[0, "metadata"]["table_metadata"]["table_content_format"] == "fmt1"
    # row1 => invalid => no table_content
    assert "table_content" not in df_out.at[1, "metadata"]["table_metadata"]
    # row2 => updated => table2/fmt2
    assert df_out.at[2, "metadata"]["table_metadata"]["table_content"] == "table2"
    assert df_out.at[2, "metadata"]["table_metadata"]["table_content_format"] == "fmt2"

    mock_update_metadata.assert_called_once_with(
        base64_images=["good1", "good2"],
        paddle_client=mock_client,
        worker_pool_size=validated_config.stage_config.workers_per_progress_engine,
        trace_info=trace_info.get("trace_info"),
    )
    mock_client.close.assert_called_once()


def test_extract_table_data_update_error(mocker, validated_config):
    """
    If _update_metadata raises an exception, we should re-raise
    but still close the paddle_client.
    """
    # Mock the paddle client so we don't make real calls or wait.
    mock_client = MagicMock()
    mock_create_client = mocker.patch(f"{MODULE_UNDER_TEST}._create_paddle_client", return_value=mock_client)

    # Mock _update_metadata to raise an error
    mock_update_metadata = mocker.patch(f"{MODULE_UNDER_TEST}._update_metadata", side_effect=RuntimeError("paddle_err"))

    df_in = pd.DataFrame(
        [
            {
                "metadata": {
                    "content_metadata": {"type": "structured", "subtype": "table"},
                    "table_metadata": {},
                    "content": "some_b64",
                }
            }
        ]
    )

    # We expect a re-raised RuntimeError from _update_metadata
    with pytest.raises(RuntimeError, match="paddle_err"):
        _extract_table_data(df_in, {}, validated_config)

    # Confirm we created a client
    mock_create_client.assert_called_once()
    # Ensure the paddle_client was closed in the finally block
    mock_client.close.assert_called_once()
    # Confirm _update_metadata was called once with our single row
    mock_update_metadata.assert_called_once()


def test_update_metadata_empty_list(paddle_mock):
    """
    If base64_images is empty, we should return an empty list
    and never call paddle_mock.infer.
    """
    with patch(f"{MODULE_UNDER_TEST}.base64_to_numpy") as mock_b64:
        result = _update_metadata([], paddle_mock)
    assert result == []
    mock_b64.assert_not_called()
    paddle_mock.infer.assert_not_called()


def test_update_metadata_all_valid(mocker, paddle_mock):
    imgs = ["b64imgA", "b64imgB"]
    # Patch base64_to_numpy so that both images are valid.
    mock_dim = mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy")
    mock_dim.side_effect = [
        np.zeros((100, 120, 3), dtype=np.uint8),  # b64imgA is valid
        np.zeros((80, 80, 3), dtype=np.uint8),  # b64imgB is valid
    ]

    # Set minimum dimensions so that both images pass.
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_WIDTH", 50)
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_HEIGHT", 50)

    # The paddle client returns a result for each valid image.
    paddle_mock.infer.return_value = [
        ("tableA", "fmtA"),
        ("tableB", "fmtB"),
    ]

    res = _update_metadata(imgs, paddle_mock, worker_pool_size=1)
    assert len(res) == 2
    assert res[0] == ("b64imgA", ("tableA", "fmtA"))
    assert res[1] == ("b64imgB", ("tableB", "fmtB"))

    # Expect one call to infer with all valid images.
    paddle_mock.infer.assert_called_once_with(
        data={"base64_images": ["b64imgA", "b64imgB"]},
        model_name="paddle",
        stage_name="table_data_extraction",
        max_batch_size=2,
        trace_info=None,
    )


def test_update_metadata_skip_small(mocker, paddle_mock):
    """
    Some images are below the min dimension => they skip inference
    and get ("", "") as results.
    """
    imgs = ["imgSmall", "imgBig"]
    mock_dim = mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy")
    # Return NumPy arrays of certain shape to emulate dimension checks.
    mock_dim.side_effect = [
        np.zeros((40, 40, 3), dtype=np.uint8),  # too small
        np.zeros((60, 70, 3), dtype=np.uint8),  # big enough
    ]
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_WIDTH", 50)
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_HEIGHT", 50)

    paddle_mock.infer.return_value = [("valid_table", "valid_fmt")]

    res = _update_metadata(imgs, paddle_mock)
    assert len(res) == 2
    # The first image is too small and is skipped.
    assert res[0] == ("imgSmall", ("", ""))
    # The second image is valid and processed.
    assert res[1] == ("imgBig", ("valid_table", "valid_fmt"))

    paddle_mock.infer.assert_called_once_with(
        data={"base64_images": ["imgBig"]},
        model_name="paddle",
        stage_name="table_data_extraction",
        max_batch_size=2,
        trace_info=None,
    )


def test_update_metadata_multiple_batches(mocker, paddle_mock):
    imgs = ["img1", "img2", "img3"]
    # Patch base64_to_numpy so that all images are valid.
    mock_dim = mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy")
    mock_dim.side_effect = [
        np.zeros((80, 80, 3), dtype=np.uint8),  # img1
        np.zeros((100, 50, 3), dtype=np.uint8),  # img2
        np.zeros((64, 64, 3), dtype=np.uint8),  # img3
    ]
    # Set minimum dimensions such that all images are considered valid.
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_WIDTH", 40)
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_HEIGHT", 40)

    # Since all images are valid, infer is called once with the full list.
    paddle_mock.infer.return_value = [
        ("table1", "fmt1"),
        ("table2", "fmt2"),
        ("table3", "fmt3"),
    ]

    res = _update_metadata(imgs, paddle_mock, worker_pool_size=2)
    assert len(res) == 3
    assert res[0] == ("img1", ("table1", "fmt1"))
    assert res[1] == ("img2", ("table2", "fmt2"))
    assert res[2] == ("img3", ("table3", "fmt3"))

    # Verify that infer is called only once with all valid images.
    paddle_mock.infer.assert_called_once_with(
        data={"base64_images": ["img1", "img2", "img3"]},
        model_name="paddle",
        stage_name="table_data_extraction",
        max_batch_size=2,
        trace_info=None,
    )


def test_update_metadata_inference_error(mocker, paddle_mock):
    """
    If paddle.infer fails for a batch, all valid images in that batch get ("",""),
    then we re-raise the exception.
    """
    imgs = ["imgA", "imgB"]
    mock_dim = mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy", return_value=np.zeros((60, 60, 3), dtype=np.uint8))
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_WIDTH", 20)
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_HEIGHT", 20)

    # Suppose the infer call fails
    paddle_mock.infer.side_effect = RuntimeError("paddle error")

    with pytest.raises(RuntimeError, match="paddle error"):
        _update_metadata(imgs, paddle_mock)

    # The code sets them to ("", "") before re-raising
    # We can’t see final 'res', but that’s the logic.


def test_update_metadata_mismatch_length(mocker, paddle_mock):
    """
    If paddle.infer returns fewer or more results than the valid_images => ValueError
    """
    imgs = ["img1", "img2"]
    mock_dim = mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy", return_value=np.zeros((80, 80, 3), dtype=np.uint8))
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_WIDTH", 20)
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_HEIGHT", 20)

    # We expect 2 results, but get only 1
    paddle_mock.infer.return_value = [("tableOnly", "fmtOnly")]

    with pytest.raises(ValueError, match="Expected 2 results"):
        _update_metadata(imgs, paddle_mock)


def test_update_metadata_non_list_return(mocker, paddle_mock):
    """
    If inference returns something that's not a list => ValueError
    """
    imgs = ["imgX"]
    mock_dim = mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy", return_value=np.zeros((70, 70, 3), dtype=np.uint8))
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_WIDTH", 50)
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_HEIGHT", 50)

    paddle_mock.infer.return_value = "some_string"

    with pytest.raises(ValueError, match="Expected a list of tuples"):
        _update_metadata(imgs, paddle_mock)


def test_update_metadata_all_small(mocker, paddle_mock):
    """
    If all images are too small, we skip inference entirely and each gets ("","").
    """
    imgs = ["imgA", "imgB"]
    mock_dim = mocker.patch(f"{MODULE_UNDER_TEST}.base64_to_numpy")
    mock_dim.side_effect = [np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((5, 20, 3), dtype=np.uint8)]
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_WIDTH", 30)
    mocker.patch(f"{MODULE_UNDER_TEST}.PADDLE_MIN_HEIGHT", 30)

    res = _update_metadata(imgs, paddle_mock)
    assert res[0] == ("imgA", ("", ""))
    assert res[1] == ("imgB", ("", ""))

    # No calls to infer
    paddle_mock.infer.assert_not_called()
