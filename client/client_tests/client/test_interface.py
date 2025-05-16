# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import logging
import os
import tempfile
from concurrent.futures import Future
from unittest.mock import MagicMock, ANY
from unittest.mock import patch

import pytest
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives import BatchJobSpec
from nv_ingest_client.primitives.jobs import JobStateEnum
from nv_ingest_client.primitives.tasks import CaptionTask
from nv_ingest_client.primitives.tasks import ChartExtractionTask
from nv_ingest_client.primitives.tasks import DedupTask
from nv_ingest_client.primitives.tasks import EmbedTask
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import FilterTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import StoreEmbedTask
from nv_ingest_client.primitives.tasks import StoreTask
from nv_ingest_client.primitives.tasks import TableExtractionTask
from nv_ingest_client.util.vdb.milvus import Milvus

import nv_ingest_client.client.interface as module_under_test

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@pytest.fixture
def mock_client():
    client = MagicMock(spec=NvIngestClient)
    return client


@pytest.fixture
def documents():
    return ["data/multimodal_test.pdf"]


@pytest.fixture
def text_documents():
    return ["data/test.txt", "data/test.html", "data/test.json", "data/test.md", "data/test.sh"]


@pytest.fixture
def ingestor(mock_client, documents):
    return Ingestor(documents, client=mock_client)


@pytest.fixture
def ingestor_without_doc(mock_client):
    return Ingestor(client=mock_client)


def test_dedup_task_no_args(ingestor):
    ingestor.dedup()

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0], DedupTask
    _ = task
    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], DedupTask)


def test_dedup_task_some_args(ingestor):
    ingestor.dedup(content_type="image", filter=True)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, DedupTask)
    assert task._content_type == "image"
    assert task._filter is True


def test_embed_task_no_args(ingestor):
    ingestor.embed()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], EmbedTask)


def test_embed_task_some_args(ingestor, caplog):
    # `text` and `table` arguments were deprecated before GA.
    with caplog.at_level(logging.WARNING):
        ingestor.embed(text=False, tables=False)

    assert "'text' parameter is deprecated" in caplog.records[0].message
    assert "'tables' parameter is deprecated" in caplog.records[1].message


def test_extract_task_no_args(ingestor):
    ingestor.extract()

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, ExtractTask)
    assert task._extract_tables is True
    assert task._extract_charts is True
    assert task._extract_infographics is False

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
    ingestor.extract(extract_tables=True, extract_charts=True, extract_images=True, extract_infographics=True)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, ExtractTask)
    assert task._extract_tables is True
    assert task._extract_charts is True
    assert task._extract_images is True
    assert task._extract_infographics is True


def test_extract_task_text_filetypes(text_documents):
    for doc in text_documents:
        Ingestor(client=mock_client).files([doc]).extract(
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_images=False,
            extract_infographics=False,
            document_type=doc.split(".")[1],
        )


def test_filter_task_no_args(ingestor):
    ingestor.filter()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], FilterTask)


def test_filter_task_some_args(ingestor):
    ingestor.filter(content_type="image", min_size=42)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, FilterTask)
    assert task._content_type == "image"
    assert task._min_size == 42


def test_split_task_no_args(ingestor):
    ingestor.split()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], SplitTask)


def test_split_task_some_args(ingestor):
    ingestor.split(tokenizer="intfloat/e5-large-unsupervised", chunk_size=42)

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, SplitTask)
    assert task._tokenizer == "intfloat/e5-large-unsupervised"
    assert task._chunk_size == 42


def test_store_task_no_args(ingestor):
    ingestor.store()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], StoreTask)


def test_store_task_some_args(ingestor):
    ingestor.store(store_method="s3")

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, StoreTask)
    assert task._store_method == "s3"


def test_store_embed_task_no_args(ingestor):
    ingestor.store_embed()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], StoreEmbedTask)


def test_store_task_some_args_extra_param(ingestor):
    ingestor.store_embed(params={"extra_param": "extra"})

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, StoreEmbedTask)
    assert task._params["extra_param"] == "extra"


def test_vdb_upload_task_no_args(ingestor):
    ingestor.vdb_upload()

    assert isinstance(ingestor._vdb_bulk_upload, Milvus)


def test_vdb_upload_task_some_args(ingestor):
    ingestor.vdb_upload(filter_errors=True)

    assert isinstance(ingestor._vdb_bulk_upload, Milvus)


def test_caption_task_no_args(ingestor):
    ingestor.caption()

    assert isinstance(ingestor._job_specs.job_specs["pdf"][0]._tasks[0], CaptionTask)


def test_caption_task_some_args(ingestor):
    ingestor.caption(model_name="foo")

    task = ingestor._job_specs.job_specs["pdf"][0]._tasks[0]
    assert isinstance(task, CaptionTask)
    assert task._model_name == "foo"


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
    assert isinstance(ingestor._vdb_bulk_upload, Milvus)
    assert len(ingestor._job_specs.job_specs["pdf"][0]._tasks) == 8


def test_ingest(ingestor, mock_client):
    """
    Test the ingest method for the successful processing path.

    Verifies that the ingestor correctly adds jobs (if applicable) and
    calls the client's concurrent processing method, returning the
    successful results.
    """
    # Arrange
    # Mock the methods expected to be called by ingestor.ingest
    job_indices = ["job_id_1", "job_id_2"]
    # Assume ingestor might call add_job first
    # If ingestor receives indices directly, this mock might not be needed.
    mock_client.add_job.return_value = job_indices
    # Mock the main processing method to return successful results
    expected_results = [{"result": "success_1"}, {"result": "success_2"}]
    mock_client.process_jobs_concurrently.return_value = expected_results

    # Store expected arguments used in process_jobs_concurrently
    # Replace with actual values if available from ingestor instance
    expected_job_queue_id = getattr(ingestor, "_job_queue_id", "default_queue")
    expected_max_retries = getattr(ingestor, "_max_retries", None)
    expected_verbose = getattr(ingestor, "_verbose", False)
    # Assume ingestor stores the indices after calling add_job, or gets them passed in
    ingestor._job_indices = job_indices  # Simulate ingestor storing indices

    # Act
    result = ingestor.ingest(timeout=30)  # timeout=30 is passed to process_jobs_concurrently

    # Assert
    # Verify add_job was called (if ingestor is responsible for it)
    if hasattr(ingestor, "_job_specs"):
        mock_client.add_job.assert_called_once_with(ingestor._job_specs)

    # Verify the main concurrent processing method was called correctly
    mock_client.process_jobs_concurrently.assert_called_once_with(
        job_indices=job_indices,
        job_queue_id=expected_job_queue_id,
        timeout=30,  # Check if the passed timeout is used directly
        max_job_retries=expected_max_retries,
        completion_callback=ANY,  # Usually None or an internal callback
        return_failures=False,  # Specific to this test case
        verbose=expected_verbose,
    )
    # Verify the result returned by ingestor matches the mocked result
    assert result == expected_results
    # Verify submit_job_async was NOT called directly by ingestor
    mock_client.submit_job_async.assert_not_called()
    # Verify fetch_job_result was NOT called directly by ingestor
    if hasattr(mock_client, "fetch_job_result"):  # Check if attr exists before asserting not called
        mock_client.fetch_job_result.assert_not_called()


def test_ingest_return_failures(ingestor, mock_client):
    """
    Test the ingest method when return_failures=True.

    Verifies that the ingestor calls the client's concurrent processing
    method with return_failures=True and correctly returns both results
    and failures.
    """
    # Arrange
    job_indices = ["job_id_1", "job_id_2", "job_id_3"]
    # Assume ingestor might call add_job first
    mock_client.add_job.return_value = job_indices
    # Mock the main processing method to return both results and failures
    expected_results = [{"result": "success_1"}]
    expected_failures = [("job_id_2", "TimeoutError"), ("job_id_3", "Processing Error")]
    mock_client.process_jobs_concurrently.return_value = (expected_results, expected_failures)

    # Store expected arguments used in process_jobs_concurrently
    expected_job_queue_id = getattr(ingestor, "_job_queue_id", "default_queue")
    expected_max_retries = getattr(ingestor, "_max_retries", None)
    expected_verbose = getattr(ingestor, "_verbose", False)
    ingestor._job_indices = job_indices  # Simulate ingestor storing indices

    # Act
    results, failures = ingestor.ingest(timeout=30, return_failures=True)  # Pass return_failures=True

    # Assert
    # Verify add_job was called (if applicable)
    if hasattr(ingestor, "_job_specs"):
        mock_client.add_job.assert_called_once_with(ingestor._job_specs)

    # Verify the main concurrent processing method was called correctly
    mock_client.process_jobs_concurrently.assert_called_once_with(
        job_indices=job_indices,
        job_queue_id=expected_job_queue_id,
        timeout=30,
        max_job_retries=expected_max_retries,
        completion_callback=ANY,
        return_failures=True,  # Specific to this test case
        # data_only=False, # Removed
        verbose=expected_verbose,
    )
    # Verify the results and failures returned match the mocked tuple
    assert results == expected_results
    assert failures == expected_failures
    # Verify submit_job was NOT called directly by ingestor (also fixing original inconsistency)
    mock_client.submit_job.assert_not_called()
    # Verify fetch_job_result was NOT called directly by ingestor
    if hasattr(mock_client, "fetch_job_result"):
        mock_client.fetch_job_result.assert_not_called()


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
        ["result_1"] if job_id == "job_id_1" else ["result_2"]
    )

    combined_result = ingestor.ingest_async(timeout=15).result()
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

    task_classes = {ExtractTask, DedupTask, FilterTask, SplitTask, EmbedTask, StoreEmbedTask}
    added_tasks = {
        type(task) for job_specs in ingestor._job_specs._file_type_to_job_spec.values() for task in job_specs[0]._tasks
    }

    assert task_classes.issubset(added_tasks), "Not all default tasks were added"


def test_load_with_existing_local_files(tmp_path):
    doc1_path = tmp_path / "doc1.txt"
    doc1_path.write_text("content1")
    doc2_path = tmp_path / "subdir" / "doc2.pdf"
    doc2_path.parent.mkdir()
    doc2_path.write_text("content2")

    initial_doc_paths = [str(doc1_path), str(doc2_path)]
    ingestor = Ingestor().files(initial_doc_paths)

    _ = ingestor.load()

    assert ingestor._all_local is True
    assert ingestor._documents == initial_doc_paths


@patch("fsspec.open")
def test_load_downloads_remote_file(mock_fsspec_open):
    remote_file = "https://aws-s3-presigned-url/remote.txt" + "?" + "x" * 1708

    fake_file_content = b"fake data from remote"
    mock_file_obj = io.BytesIO(fake_file_content)
    mock_file_obj.path = remote_file

    mock_fsspec_open.return_value.__enter__.return_value = mock_file_obj

    ingestor = Ingestor().files(remote_file)
    assert ingestor._all_local is False

    _ = ingestor.load()

    assert ingestor._all_local is True

    assert os.path.exists(ingestor._documents[0])
    assert "remote.txt" in ingestor._documents[0]
    with open(ingestor._documents[0], "rb") as f:
        assert f.read() == fake_file_content


@patch("fsspec.open")
def test_load_mixed_local_and_remote(mock_fsspec_open, tmp_path):
    local_file = tmp_path / "local.txt"
    local_file.write_text("Local content.")

    remote_file = "https://aws-s3-presigned-url/remote.txt" + "?" + "x" * 1708
    ingestor = Ingestor().files([str(local_file), remote_file])
    assert ingestor._all_local is False

    fake_file_content = b"fake data from remote"

    mock_file_obj = io.BytesIO(fake_file_content)
    mock_file_obj.path = remote_file

    mock_fsspec_open.return_value.__enter__.return_value = mock_file_obj

    _ = ingestor.load()

    assert ingestor._all_local is True
    assert any("local.txt" in path for path in ingestor._documents)
    assert any("remote.txt" in path for path in ingestor._documents)
