# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import tempfile
from concurrent.futures import Future
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives import BatchJobSpec
from nv_ingest_client.primitives.jobs import JobStateEnum
from nv_ingest_client.primitives.tasks import ChartExtractionTask
from nv_ingest_client.primitives.tasks import DedupTask
from nv_ingest_client.primitives.tasks import EmbedTask
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import FilterTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import StoreTask
from nv_ingest_client.primitives.tasks import TableExtractionTask
from nv_ingest_client.primitives.tasks import VdbUploadTask

MODULE_UNDER_TEST = "nv_ingest_client.client.interface"


@pytest.fixture
def mock_client():
    client = MagicMock(spec=NvIngestClient)
    return client


@pytest.fixture
def documents():
    return ["data/multimodal_test.pdf"]


@pytest.fixture
def ingestor(mock_client, documents):
    return Ingestor(documents, client=mock_client)


@pytest.fixture
def ingestor_without_doc(mock_client):
    return Ingestor(client=mock_client)


def test_dedup_task_no_args(ingestor):
    ingestor.dedup()

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0], DedupTask
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], DedupTask)


def test_dedup_task_some_args(ingestor):
    ingestor.dedup(content_type="foo", filter=True)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, DedupTask)
    assert task._content_type == "foo"
    assert task._filter is True


def test_embed_task_no_args(ingestor):
    ingestor.embed()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], EmbedTask)


def test_embed_task_some_args(ingestor):
    ingestor.embed(text=False, tables=False)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, EmbedTask)
    assert task._text is False
    assert task._tables is False


def test_extract_task_no_args(ingestor):
    ingestor.extract()

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, ExtractTask)
    assert task._extract_tables is True
    assert task._extract_charts is True

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[1], TableExtractionTask)
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[2], ChartExtractionTask)


def test_extract_task_args_tables_false(ingestor):
    ingestor.extract(extract_tables=False)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, ExtractTask)
    assert task._extract_tables is False
    assert task._extract_charts is True


def test_extract_task_args_charts_false(ingestor):
    ingestor.extract(extract_charts=False)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, ExtractTask)
    assert task._extract_tables is True
    assert task._extract_charts is False

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[1], TableExtractionTask)


def test_extract_task_args_tables_and_charts_false(ingestor):
    ingestor.extract(extract_tables=False, extract_charts=False)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, ExtractTask)
    assert task._extract_tables is False
    assert task._extract_charts is False


def test_extract_task_some_args(ingestor):
    ingestor.extract(extract_tables=True, extract_charts=True, extract_images=True)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, ExtractTask)
    assert task._extract_tables is True
    assert task._extract_charts is True
    assert task._extract_images is True


def test_filter_task_no_args(ingestor):
    ingestor.filter()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], FilterTask)


def test_filter_task_some_args(ingestor):
    ingestor.filter(content_type="foo", min_size=42)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, FilterTask)
    assert task._content_type == "foo"
    assert task._min_size == 42


def test_split_task_no_args(ingestor):
    ingestor.split()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], SplitTask)


def test_split_task_some_args(ingestor):
    ingestor.split(split_by="word", split_length=42)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, SplitTask)
    assert task._split_by == "word"
    assert task._split_length == 42


def test_store_task_no_args(ingestor):
    ingestor.store()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], StoreTask)


def test_store_task_some_args(ingestor):
    ingestor.store(store_method="s3")

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, StoreTask)
    assert task._store_method == "s3"


def test_vdb_upload_task_no_args(ingestor):
    ingestor.vdb_upload()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], VdbUploadTask)


def test_vdb_upload_task_some_args(ingestor):
    ingestor.vdb_upload(filter_errors=True)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, VdbUploadTask)
    assert task._filter_errors is True


def test_chain(ingestor):
    ingestor.dedup().embed().extract().filter().split().store().vdb_upload()
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], DedupTask)
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[1], EmbedTask)
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[2], ExtractTask)
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[3], TableExtractionTask)
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[4], ChartExtractionTask)
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[5], FilterTask)
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[6], SplitTask)
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[7], StoreTask)
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[8], VdbUploadTask)
    assert len(ingestor._job_specs.job_specs["pdf"][0]._tasks) == 9


def test_ingest(ingestor, mock_client):
    mock_client.add_job.return_value = ["job_id_1", "job_id_2"]
    mock_client.submit_job.return_value = ["job_state_1", "job_state_2"]
    mock_client.fetch_job_result.return_value = [{"result": "success"}]

    result = ingestor.ingest(timeout=30)

    mock_client.add_job.assert_called_once_with(ingestor._job_specs)
    mock_client.submit_job.assert_called_once_with(mock_client.add_job.return_value, ingestor._job_queue_id)

    mock_client.fetch_job_result.assert_called_once_with(mock_client.add_job.return_value)
    assert result == [{"result": "success"}]


def test_ingest_async(ingestor, mock_client):
    mock_client.add_job.return_value = ["job_id_1", "job_id_2"]

    future1 = Future()
    future2 = Future()
    future1.set_result("result_1")
    future2.set_result("result_2")
    mock_client.submit_job_async.return_value = {future1: "job_id_1", future2: "job_id_2"}

    ingestor._job_states = {}
    ingestor._job_states["job_id_1"] = MagicMock(state=JobStateEnum.COMPLETED)
    ingestor._job_states["job_id_2"] = MagicMock(state=JobStateEnum.FAILED)

    mock_client.fetch_job_result.side_effect = lambda job_id, *args, **kwargs: (
        "result_1" if job_id == "job_id_1" else "result_2"
    )

    combined_future = ingestor.ingest_async(timeout=15)
    combined_result = combined_future.result()

    assert combined_result == ["result_1", "result_2"]


def test_create_client(ingestor):
    ingestor._client = None
    ingestor._create_client()
    assert isinstance(ingestor._client, NvIngestClient)

    with pytest.raises(ValueError, match="self._client already exists"):
        ingestor._create_client()


def test_client_initialization_with_kwargs(documents):
    client_kwargs = {
        "message_client_hostname": "custom-hostname",
        "message_client_port": 8080,
        "extra_arg": "should_be_ignored",
    }

    manager = Ingestor(documents, **client_kwargs)

    assert manager._client._message_client_hostname == "custom-hostname"
    assert manager._client._message_client_port == 8080


def test_job_state_counting(ingestor):
    ingestor._job_states = {
        "job_1": MagicMock(state=JobStateEnum.COMPLETED),
        "job_2": MagicMock(state=JobStateEnum.FAILED),
        "job_3": MagicMock(state=JobStateEnum.CANCELLED),
        "job_4": MagicMock(state=JobStateEnum.COMPLETED),
    }

    assert ingestor.completed_jobs() == 2
    assert ingestor.failed_jobs() == 1
    assert ingestor.cancelled_jobs() == 1
    assert ingestor.remaining_jobs() == 0  # All jobs are in terminal states


@patch("glob.glob")
@patch("os.path.exists")
def test_check_files_local_all_local(mock_exists, mock_glob, ingestor_without_doc):
    ingestor = ingestor_without_doc
    ingestor.files(["/local/path/doc1.pdf", "/local/path/doc2.pdf"])
    mock_glob.side_effect = lambda x, recursive: [x]
    mock_exists.return_value = True
    assert ingestor._check_files_local() is True


@patch("glob.glob")
@patch("os.path.exists")
def test_check_files_local_some_missing(mock_exists, mock_glob, ingestor_without_doc):
    ingestor = ingestor_without_doc
    ingestor.files(["/local/path/doc1.pdf", "/local/path/missing_doc.pdf"])
    mock_glob.side_effect = lambda x, recursive: [x]
    mock_exists.side_effect = lambda x: x != "/local/path/missing_doc.pdf"
    assert ingestor._check_files_local() is False


def test_files_with_remote_files(ingestor_without_doc):
    with tempfile.TemporaryDirectory() as temp_dir:
        ingestor_without_doc.files(["s3://bucket/path/to/doc1.pdf", "s3://bucket/path/to/doc2.pdf"])

        assert ingestor_without_doc._all_local is False
        assert ingestor_without_doc._job_specs is None

        ingestor_without_doc._documents = [f"{temp_dir}/doc1.pdf", f"{temp_dir}/doc2.pdf"]
        ingestor_without_doc._all_local = True
        ingestor_without_doc._job_specs = BatchJobSpec(ingestor_without_doc._documents)

        expected_paths = [f"{temp_dir}/doc1.pdf", f"{temp_dir}/doc2.pdf"]
        assert ingestor_without_doc._documents == expected_paths
        assert ingestor_without_doc._all_local is True
        assert isinstance(ingestor_without_doc._job_specs, BatchJobSpec)


def test_all_tasks_adds_default_tasks(ingestor):
    ingestor.all_tasks()

    task_classes = {ExtractTask, DedupTask, FilterTask, SplitTask, EmbedTask}
    added_tasks = {
        type(task) for job_specs in ingestor._job_specs._file_type_to_job_spec.values() for task in job_specs[0]._tasks
    }

    assert task_classes.issubset(added_tasks), "Not all default tasks were added"
