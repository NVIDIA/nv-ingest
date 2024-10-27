from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
from nv_ingest_client.client.client import NvIngestClient
from nv_ingest_client.client.pipeline import NvIngestPipeline
from nv_ingest_client.primitives import BatchJobSpec
from nv_ingest_client.primitives.jobs import JobStateEnum
from nv_ingest_client.primitives.tasks import DedupTask
from nv_ingest_client.primitives.tasks import EmbedTask
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import FilterTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import StoreTask
from nv_ingest_client.primitives.tasks import VdbUploadTask


@pytest.fixture
def mock_client():
    client = MagicMock(spec=NvIngestClient)
    return client


@pytest.fixture
def pipeline(mock_client):
    documents = ["data/multimodal_test.pdf"]
    return NvIngestPipeline(documents, client=mock_client)


def test_dedup_task_no_args(pipeline):
    pipeline.dedup()

    task = pipeline._job_specs.job_specs["pdf"][0]._tasks[0], DedupTask
    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[0], DedupTask)


def test_dedup_task_some_args(pipeline):
    pipeline.dedup(content_type="foo", filter=True)

    task = pipeline._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, DedupTask)
    assert task._content_type == "foo"
    assert task._filter is True


def test_embed_task_no_args(pipeline):
    pipeline.embed()

    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[0], EmbedTask)


def test_embed_task_some_args(pipeline):
    pipeline.embed(text=False, tables=False)

    task = pipeline._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, EmbedTask)
    assert task._text is False
    assert task._tables is False


def test_extract_task_no_args(pipeline):
    pipeline.extract()

    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[0], ExtractTask)


def test_extract_task_some_args(pipeline):
    pipeline.extract(extract_tables=True, extract_charts=True, extract_images=True)

    task = pipeline._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, ExtractTask)
    assert task._extract_tables is True
    assert task._extract_charts is True
    assert task._extract_images is True


def test_filter_task_no_args(pipeline):
    pipeline.filter()

    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[0], FilterTask)


def test_filter_task_some_args(pipeline):
    pipeline.filter(content_type="foo", min_size=42)

    task = pipeline._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, FilterTask)
    assert task._content_type == "foo"
    assert task._min_size == 42


def test_split_task_no_args(pipeline):
    pipeline.split()

    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[0], SplitTask)


def test_split_task_some_args(pipeline):
    pipeline.split(split_by="word", split_length=42)

    task = pipeline._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, SplitTask)
    assert task._split_by == "word"
    assert task._split_length == 42


def test_store_task_no_args(pipeline):
    pipeline.store()

    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[0], StoreTask)


def test_store_task_some_args(pipeline):
    pipeline.store(store_method="s3")

    task = pipeline._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, StoreTask)
    assert task._store_method == "s3"


def test_vdb_upload_task_no_args(pipeline):
    pipeline.vdb_upload()

    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[0], VdbUploadTask)


def test_vdb_upload_task_some_args(pipeline):
    pipeline.vdb_upload(filter_errors=True)

    task = pipeline._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, VdbUploadTask)
    assert task._filter_errors is True


def test_chain(pipeline):
    pipeline.dedup().embed().extract().filter().split().store().vdb_upload()
    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[0], DedupTask)
    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[1], EmbedTask)
    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[2], ExtractTask)
    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[3], FilterTask)
    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[4], SplitTask)
    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[5], StoreTask)
    assert isinstance(pipeline._job_specs.job_specs["pdf"][0]._tasks[6], VdbUploadTask)
    assert len(pipeline._job_specs.job_specs["pdf"][0]._tasks) == 7


def test_run(pipeline, mock_client):
    mock_client.add_job.return_value = ["job_id_1", "job_id_2"]
    mock_client.submit_job.return_value = ["job_state_1", "job_state_2"]
    mock_client.fetch_job_result.return_value = [{"result": "success"}]

    result = pipeline.run(timeout=30)

    mock_client.add_job.assert_called_once_with(pipeline._job_specs)
    mock_client.submit_job.assert_called_once_with(mock_client.add_job.return_value, pipeline._job_queue_id)

    mock_client.fetch_job_result.assert_called_once_with(mock_client.add_job.return_value)
    assert result == [{"result": "success"}]


def test_run_async(pipeline, mock_client):
    mock_client.add_job.return_value = ["job_id_1", "job_id_2"]

    future1 = Future()
    future2 = Future()
    future1.set_result("result_1")
    future2.set_result("result_2")
    mock_client.submit_job_async.return_value = {future1: "job_id_1", future2: "job_id_2"}

    pipeline._job_states = {}
    pipeline._job_states["job_id_1"] = MagicMock(state=JobStateEnum.COMPLETED)
    pipeline._job_states["job_id_2"] = MagicMock(state=JobStateEnum.FAILED)

    mock_client.fetch_job_result.side_effect = (
        lambda job_id, *args, **kwargs: "result_1" if job_id == "job_id_1" else "result_2"
    )

    combined_future = pipeline.run_async(timeout=15)
    combined_result = combined_future.result()

    assert combined_result == ["result_1", "result_2"]


def test_create_client(pipeline):
    pipeline._client = None
    pipeline._create_client()
    assert isinstance(pipeline._client, NvIngestClient)

    with pytest.raises(ValueError, match="self._client already exists"):
        pipeline._create_client()


def test_job_state_counting(pipeline):
    pipeline._job_states = {
        "job_1": MagicMock(state=JobStateEnum.COMPLETED),
        "job_2": MagicMock(state=JobStateEnum.FAILED),
        "job_3": MagicMock(state=JobStateEnum.CANCELLED),
        "job_4": MagicMock(state=JobStateEnum.COMPLETED),
    }

    assert pipeline.completed_jobs() == 2
    assert pipeline.failed_jobs() == 1
    assert pipeline.cancelled_jobs() == 1
    assert pipeline.remaining_jobs() == 0  # All jobs are in terminal states
