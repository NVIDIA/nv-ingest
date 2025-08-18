# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import json
import logging
import os
import tempfile
from concurrent.futures import Future
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import patch

import nv_ingest_client.client.interface as module_under_test
import pytest

from client.client_tests.utilities_for_test import (
    cleanup_test_workspace,
    create_test_workspace,
    get_git_root,
    find_root_by_pattern,
)
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import LazyLoadedList
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
from nv_ingest_client.util.vdb import VDB


MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@pytest.fixture
def mock_client():
    client = MagicMock(spec=NvIngestClient)
    return client


@pytest.fixture
def workspace():
    test_workspace = create_test_workspace("ingestor_pytest_")
    doc1_path = os.path.join(test_workspace, "doc1.txt")
    with open(doc1_path, "w") as f:
        f.write("This is a test document.")

    yield test_workspace, doc1_path

    cleanup_test_workspace(test_workspace)


@pytest.fixture
def documents():
    # Use utilities to find the actual data directory
    git_root = get_git_root(__file__)
    if git_root:
        data_dir = os.path.join(git_root, "data")
        if os.path.exists(data_dir):
            pdf_path = os.path.join(data_dir, "multimodal_test.pdf")
            if os.path.exists(pdf_path):
                return [pdf_path]

    # Fallback: search for data directory pattern
    root_dir = find_root_by_pattern("data/multimodal_test.pdf", start_dir=os.path.dirname(__file__))
    pdf_path = os.path.join(root_dir, "data", "multimodal_test.pdf")
    if os.path.exists(pdf_path):
        return [pdf_path]

    # If no actual test file found, fall back to the original path
    return ["data/multimodal_test.pdf"]


@pytest.fixture
def text_documents():
    # Use utilities to find the actual data directory
    git_root = get_git_root(__file__)
    if git_root:
        data_dir = os.path.join(git_root, "data")
        if os.path.exists(data_dir):
            files = [
                os.path.join(data_dir, "test.txt"),
                os.path.join(data_dir, "test.html"),
                os.path.join(data_dir, "test.json"),
                os.path.join(data_dir, "test.md"),
                os.path.join(data_dir, "test.sh"),
            ]
            # Return only files that actually exist
            existing_files = [f for f in files if os.path.exists(f)]
            if existing_files:
                return existing_files

    # Fallback: search for data directory pattern
    root_dir = find_root_by_pattern("data", start_dir=os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, "data")
    if os.path.exists(data_dir):
        files = [
            os.path.join(data_dir, "test.txt"),
            os.path.join(data_dir, "test.html"),
            os.path.join(data_dir, "test.json"),
            os.path.join(data_dir, "test.md"),
            os.path.join(data_dir, "test.sh"),
        ]
        # Return only files that actually exist
        existing_files = [f for f in files if os.path.exists(f)]
        if existing_files:
            return existing_files

    # If no actual test files found, fall back to original paths
    return [
        "data/test.txt",
        "data/test.html",
        "data/test.json",
        "data/test.md",
        "data/test.sh",
    ]


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
    ingestor.extract(
        extract_tables=True,
        extract_charts=True,
        extract_images=True,
        extract_infographics=True,
    )

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
    ingestor.split(tokenizer="intfloat/e5-large-unsupervised", chunk_size=42, chunk_overlap=20)

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
    mock_client.process_jobs_concurrently.return_value = (expected_results, [])

    # Store expected arguments used in process_jobs_concurrently
    # Replace with actual values if available from ingestor instance
    expected_job_queue_id = getattr(ingestor, "_job_queue_id", "default_queue")
    expected_max_retries = getattr(ingestor, "_max_retries", None)
    expected_verbose = getattr(ingestor, "_verbose", False)
    # Assume ingestor stores the indices after calling add_job, or gets them passed in
    ingestor._job_indices = job_indices  # Simulate ingestor storing indices
    ingestor._output_conig = None

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
        return_failures=True,
        stream_to_callback_only=False,
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
    mock_client.process_jobs_concurrently.return_value = (
        expected_results,
        expected_failures,
    )

    # Store expected arguments used in process_jobs_concurrently
    expected_job_queue_id = getattr(ingestor, "_job_queue_id", "default_queue")
    expected_max_retries = getattr(ingestor, "_max_retries", None)
    expected_verbose = getattr(ingestor, "_verbose", False)
    ingestor._job_indices = job_indices  # Simulate ingestor storing indices
    ingestor._output_conig = None

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
        return_failures=True,
        # data_only=False, # Removed
        stream_to_callback_only=False,
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
    mock_client.submit_job_async.return_value = {
        future1: "job_id_1",
        future2: "job_id_2",
    }

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

        ingestor_without_doc._documents = [
            f"{temp_dir}/doc1.pdf",
            f"{temp_dir}/doc2.pdf",
        ]
        ingestor_without_doc._all_local = True
        ingestor_without_doc._job_specs = BatchJobSpec(ingestor_without_doc._documents)

        expected_paths = [f"{temp_dir}/doc1.pdf", f"{temp_dir}/doc2.pdf"]
        assert ingestor_without_doc._documents == expected_paths
        assert ingestor_without_doc._all_local is True
        assert isinstance(ingestor_without_doc._job_specs, BatchJobSpec)


def test_all_tasks_adds_default_tasks(ingestor):
    ingestor.all_tasks()

    task_classes = {
        ExtractTask,
        DedupTask,
        FilterTask,
        SplitTask,
        EmbedTask,
        StoreEmbedTask,
    }
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


ENSURE_DIR_PATH = "nv_ingest_client.client.interface.ensure_directory_with_permissions"


def test_save_to_disk_sets_config_and_calls_ensure_dir(ingestor, tmp_path):
    output_dir_str = str(tmp_path / "test_output")

    with patch(ENSURE_DIR_PATH) as mock_ensure_dir:
        returned_ingestor = ingestor.save_to_disk(output_directory=output_dir_str)

        assert ingestor._output_config is not None
        assert ingestor._output_config["output_directory"] == output_dir_str
        mock_ensure_dir.assert_called_once_with(output_dir_str)
        assert returned_ingestor is ingestor


def test_save_to_disk_config_structure(ingestor, tmp_path):
    output_dir_str = str(tmp_path / "specific_config")

    with patch(ENSURE_DIR_PATH):
        ingestor.save_to_disk(output_directory=output_dir_str)

    expected_config = {
        "output_directory": output_dir_str,
        "cleanup": True,
    }
    assert ingestor._output_config == expected_config


def test_save_to_disk_propagates_oserror_from_ensure_dir(ingestor, tmp_path):
    output_dir_str = str(tmp_path / "restricted_output")

    with patch(ENSURE_DIR_PATH, side_effect=OSError("Test Permission Denied")) as mock_ensure_dir:
        with pytest.raises(OSError, match="Test Permission Denied"):
            ingestor.save_to_disk(output_directory=output_dir_str)
        mock_ensure_dir.assert_called_once_with(output_dir_str)


@pytest.fixture
def create_jsonl_file(tmp_path):
    def _creator(filename="test_data.jsonl", data=None, content_override=None):
        filepath = tmp_path / filename
        if content_override is not None:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content_override)
        elif data is not None:
            with open(filepath, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        else:
            default_data = [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Charlie"},
            ]
            with open(filepath, "w", encoding="utf-8") as f:
                for item in default_data:
                    f.write(json.dumps(item) + "\n")
        return str(filepath)

    return _creator


def test_lazy_list_core_functionality(create_jsonl_file):
    default_data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ]
    filepath = create_jsonl_file(data=default_data)

    lazy_list_prelen = LazyLoadedList(filepath, expected_len=3)
    assert len(lazy_list_prelen) == 3
    assert list(lazy_list_prelen) == default_data
    assert lazy_list_prelen[1] == default_data[1]
    assert lazy_list_prelen[-1] == default_data[-1]
    assert "len=3" in repr(lazy_list_prelen)

    lazy_list_ondemand = LazyLoadedList(filepath)
    assert lazy_list_ondemand._len is None
    assert len(lazy_list_ondemand) == 3
    assert lazy_list_ondemand._len == 3  # Check caching
    assert list(lazy_list_ondemand) == default_data  # Iteration after len calculation
    assert lazy_list_ondemand[0] == default_data[0]
    assert "len=3" in repr(lazy_list_ondemand)

    assert lazy_list_ondemand.get_all_items() == default_data
    assert isinstance(lazy_list_ondemand.get_all_items(), list)


def test_save_to_disk_with_default_cleanup(workspace, monkeypatch):
    monkeypatch.setattr("nv_ingest_client.client.interface.NvIngestClient", MagicMock())
    _, doc1_path = workspace
    ingestor_instance = Ingestor(documents=[doc1_path])

    dir_to_be_cleaned = None
    with ingestor_instance as ingestor:
        ingestor.save_to_disk()
        dir_to_be_cleaned = ingestor._output_config["output_directory"]
        assert dir_to_be_cleaned is not None, "Directory for cleanup should be set"
        assert os.path.exists(dir_to_be_cleaned), "Temp directory should exist inside context"

    assert not os.path.exists(dir_to_be_cleaned), "Temp directory should be removed after exiting context"


def test_save_to_disk_with_explicit_cleanup_true(workspace, monkeypatch):
    monkeypatch.setattr("nv_ingest_client.client.interface.NvIngestClient", MagicMock())
    test_workspace, doc1_path = workspace
    user_dir = os.path.join(test_workspace, "user_results")
    os.makedirs(user_dir, exist_ok=True)

    ingestor_instance = Ingestor(documents=[doc1_path])

    with ingestor_instance as ingestor:
        ingestor.save_to_disk(output_directory=user_dir, cleanup=True)
        assert ingestor._output_config["output_directory"] == user_dir
        assert os.path.exists(user_dir), "User-provided directory should exist inside context"

    assert not os.path.exists(user_dir), "User-provided directory should be removed when cleanup=True"


def test_vdb_upload_with_purge_removes_result_files(workspace, monkeypatch):
    mock_client = MagicMock(spec=NvIngestClient)
    mock_vdb_op = MagicMock(spec=VDB)
    monkeypatch.setattr(
        "nv_ingest_client.client.interface.NvIngestClient",
        lambda *args, **kwargs: mock_client,
    )

    test_workspace, doc1_path = workspace
    results_dir = os.path.join(test_workspace, "vdb_purge_test")
    os.makedirs(results_dir)

    dummy_result_filepath = os.path.join(results_dir, "doc1.txt.results.jsonl")

    def fake_processor(completion_callback=None, **kwargs):
        if completion_callback:
            with open(dummy_result_filepath, "w") as f:
                f.write('{"data": "some_embedding"}\n')
            completion_callback(results_data=[{"data": "some_embedding"}], job_id="0")
        return ([], [])

    mock_client.process_jobs_concurrently.side_effect = fake_processor
    mock_client._job_index_to_job_spec = {"0": MagicMock(source_name=doc1_path)}

    with Ingestor(documents=[doc1_path]) as ingestor:
        ingestor.save_to_disk(output_directory=results_dir, cleanup=False)
        ingestor.vdb_upload(vdb_op=mock_vdb_op, purge_results_after_upload=True)
        ingestor.ingest(show_progress=False)

        mock_vdb_op.run.assert_called_once()

    assert not os.path.exists(dummy_result_filepath), "Result file should be purged after VDB upload"
    assert os.path.exists(results_dir)


def test_vdb_upload_without_purge_preserves_result_files(workspace, monkeypatch):
    mock_client = MagicMock(spec=NvIngestClient)
    mock_vdb_op = MagicMock(spec=VDB)
    monkeypatch.setattr(
        "nv_ingest_client.client.interface.NvIngestClient",
        lambda *args, **kwargs: mock_client,
    )

    test_workspace, doc1_path = workspace
    results_dir = os.path.join(test_workspace, "vdb_preserve_test")
    os.makedirs(results_dir)
    dummy_result_filepath = os.path.join(results_dir, "doc1.txt.results.jsonl")

    def fake_processor(completion_callback=None, **kwargs):
        if completion_callback:
            with open(dummy_result_filepath, "w") as f:
                f.write('{"data": "some_embedding"}\n')
            completion_callback(results_data=[{"data": "some_embedding"}], job_id="0")
        return ([], [])

    mock_client.process_jobs_concurrently.side_effect = fake_processor
    mock_client._job_index_to_job_spec = {"0": MagicMock(source_name=doc1_path)}

    with Ingestor(documents=[doc1_path]) as ingestor:
        ingestor.save_to_disk(output_directory=results_dir, cleanup=False)
        ingestor.vdb_upload(vdb_op=mock_vdb_op, purge_results_after_upload=False)
        ingestor.ingest(show_progress=False)

        mock_vdb_op.run.assert_called_once()

    assert os.path.exists(dummy_result_filepath), "Result file should be preserved"


def test_vdb_upload_with_failures_return_failures_true(workspace, monkeypatch, caplog):
    """Test VDB upload with failures when return_failures=True.

    Should upload successful results, emit warning message, and return failures.
    """
    mock_client = MagicMock(spec=NvIngestClient)
    mock_vdb_op = MagicMock(spec=VDB)
    monkeypatch.setattr(
        "nv_ingest_client.client.interface.NvIngestClient",
        lambda *args, **kwargs: mock_client,
    )

    test_workspace, doc1_path = workspace
    results_dir = os.path.join(test_workspace, "vdb_failure_test")
    os.makedirs(results_dir)

    # Mock successful results for 2 jobs and failures for 1 job
    successful_results = [[{"data": "embedding1", "source": "doc1"}], [{"data": "embedding2", "source": "doc2"}]]
    failures = [("job_3", "Processing failed")]

    def fake_processor(completion_callback=None, **kwargs):
        if completion_callback:
            with open(os.path.join(results_dir, "doc1.txt.results.jsonl"), "w") as f:
                f.write('{"data": "embedding1"}\n')
                f.write('{"data": "embedding2"}\n')
            completion_callback(results_data=successful_results[0], job_id="0")
            completion_callback(results_data=successful_results[1], job_id="1")
        return (successful_results, failures)

    mock_client.process_jobs_concurrently.side_effect = fake_processor
    mock_client._job_index_to_job_spec = {"0": MagicMock(source_name=doc1_path), "1": MagicMock(source_name="doc2.txt")}

    with Ingestor(documents=[doc1_path]) as ingestor:
        ingestor.save_to_disk(output_directory=results_dir, cleanup=False)
        ingestor.vdb_upload(vdb_op=mock_vdb_op, purge_results_after_upload=True)

        # Should return results and failures when return_failures=True
        results, returned_failures = ingestor.ingest(show_progress=False, return_failures=True)

        # Verify VDB upload was called with successful results only
        mock_vdb_op.run.assert_called_once()
        called_args = mock_vdb_op.run.call_args[0][0]
        assert len(called_args) == 2, "Should have 2 LazyLoadedList objects"
        assert all(isinstance(item, LazyLoadedList) for item in called_args), "Results should be LazyLoadedList objects"

        # Verify warning message was logged
        assert "Job was not completely successful" in caplog.text
        assert "2 out of 3 records completed successfully" in caplog.text
        assert "Uploading successful results to vector database" in caplog.text

        # Verify return values
        assert len(results) == 2
        assert all(isinstance(item, LazyLoadedList) for item in results)
        assert returned_failures == failures

        # Verify purge happened after successful upload
        assert not os.path.exists(os.path.join(results_dir, "doc1.txt.results.jsonl"))


def test_vdb_upload_with_failures_return_failures_false(workspace, monkeypatch):
    """Test VDB upload with failures when return_failures=False.

    Should raise RuntimeError without uploading anything.
    """
    mock_client = MagicMock(spec=NvIngestClient)
    mock_vdb_op = MagicMock(spec=VDB)
    monkeypatch.setattr(
        "nv_ingest_client.client.interface.NvIngestClient",
        lambda *args, **kwargs: mock_client,
    )

    test_workspace, doc1_path = workspace
    results_dir = os.path.join(test_workspace, "vdb_failure_strict_test")
    os.makedirs(results_dir)

    # Mock some successful results and some failures
    successful_results = [[{"data": "embedding1"}]]
    failures = [("job_2", "Processing failed")]

    def fake_processor(completion_callback=None, **kwargs):
        return (successful_results, failures)

    mock_client.process_jobs_concurrently.side_effect = fake_processor

    with Ingestor(documents=[doc1_path]) as ingestor:
        ingestor.save_to_disk(output_directory=results_dir, cleanup=False)
        ingestor.vdb_upload(vdb_op=mock_vdb_op, purge_results_after_upload=True)

        # Should raise RuntimeError when return_failures=False and failures exist
        with pytest.raises(RuntimeError) as exc_info:
            ingestor.ingest(show_progress=False, return_failures=False)

        # Verify error message content
        error_msg = str(exc_info.value)
        assert "Failed to ingest documents, unable to complete vdb bulk upload" in error_msg
        assert "no successful results" in error_msg
        assert "1 out of" in error_msg and "records failed" in error_msg

        # Verify VDB upload was NOT called
        mock_vdb_op.run.assert_not_called()


def test_vdb_upload_with_no_failures(workspace, monkeypatch):
    """Test VDB upload with no failures.

    Should work normally regardless of return_failures setting.
    """
    mock_client = MagicMock(spec=NvIngestClient)
    mock_vdb_op = MagicMock(spec=VDB)
    monkeypatch.setattr(
        "nv_ingest_client.client.interface.NvIngestClient",
        lambda *args, **kwargs: mock_client,
    )

    test_workspace, doc1_path = workspace
    results_dir = os.path.join(test_workspace, "vdb_success_test")
    os.makedirs(results_dir)
    dummy_result_filepath = os.path.join(results_dir, "doc1.txt.results.jsonl")

    # Mock only successful results, no failures
    successful_results = [[{"data": "embedding1"}], [{"data": "embedding2"}]]
    failures = []

    def fake_processor(completion_callback=None, **kwargs):
        if completion_callback:
            with open(dummy_result_filepath, "w") as f:
                f.write('{"data": "embedding1"}\n')
                f.write('{"data": "embedding2"}\n')
            completion_callback(results_data=successful_results[0], job_id="0")
            completion_callback(results_data=successful_results[1], job_id="1")
        return (successful_results, failures)

    mock_client.process_jobs_concurrently.side_effect = fake_processor
    mock_client._job_index_to_job_spec = {"0": MagicMock(source_name=doc1_path), "1": MagicMock(source_name="doc2.txt")}

    with Ingestor(documents=[doc1_path]) as ingestor:
        ingestor.save_to_disk(output_directory=results_dir, cleanup=False)
        ingestor.vdb_upload(vdb_op=mock_vdb_op, purge_results_after_upload=True)

        # Test with return_failures=False (should return only results)
        results = ingestor.ingest(show_progress=False, return_failures=False)

        # Verify VDB upload was called with all results
        mock_vdb_op.run.assert_called_once()
        called_args = mock_vdb_op.run.call_args[0][0]
        assert len(called_args) == 2, "Should have 2 LazyLoadedList objects"
        assert all(isinstance(item, LazyLoadedList) for item in called_args), "Results should be LazyLoadedList objects"

        # Verify return value (only results, no failures tuple)
        assert len(results) == 2
        assert all(isinstance(item, LazyLoadedList) for item in results)

        # Verify purge happened
        assert not os.path.exists(dummy_result_filepath)


def test_vdb_upload_with_all_failures_return_failures_true(workspace, monkeypatch, caplog):
    """Test VDB upload when all jobs fail and return_failures=True.

    Should not upload anything but should emit warning and return failures.
    """
    mock_client = MagicMock(spec=NvIngestClient)
    mock_vdb_op = MagicMock(spec=VDB)
    monkeypatch.setattr(
        "nv_ingest_client.client.interface.NvIngestClient",
        lambda *args, **kwargs: mock_client,
    )

    test_workspace, doc1_path = workspace
    results_dir = os.path.join(test_workspace, "vdb_all_failures_test")
    os.makedirs(results_dir)

    # Mock no successful results, only failures
    successful_results = []
    failures = [("job_1", "Processing failed"), ("job_2", "Processing failed"), ("job_3", "Processing failed")]

    def fake_processor(completion_callback=None, **kwargs):
        return (successful_results, failures)

    mock_client.process_jobs_concurrently.side_effect = fake_processor

    with Ingestor(documents=[doc1_path]) as ingestor:
        ingestor.save_to_disk(output_directory=results_dir, cleanup=False)
        ingestor.vdb_upload(vdb_op=mock_vdb_op, purge_results_after_upload=True)

        # Should return empty results and all failures when return_failures=True
        results, returned_failures = ingestor.ingest(show_progress=False, return_failures=True)

        # Verify VDB upload was NOT called (no successful results to upload)
        mock_vdb_op.run.assert_not_called()

        # Verify warning message was logged
        assert "Job was not completely successful" in caplog.text
        assert "0 out of 3 records completed successfully" in caplog.text

        # Verify return values
        assert len(results) == 0
        assert results == []  # Should be empty list when no successful results
        assert returned_failures == failures


def test_vdb_upload_return_failures_true_with_tuple_return(workspace, monkeypatch):
    """Test that VDB upload with return_failures=True returns tuple format.

    Verifies the return format is (results, failures) when return_failures=True.
    """
    mock_client = MagicMock(spec=NvIngestClient)
    mock_vdb_op = MagicMock(spec=VDB)
    monkeypatch.setattr(
        "nv_ingest_client.client.interface.NvIngestClient",
        lambda *args, **kwargs: mock_client,
    )

    test_workspace, doc1_path = workspace

    # Mock successful results with no failures
    successful_results = [[{"data": "embedding1"}]]
    failures = []

    def fake_processor(completion_callback=None, **kwargs):
        return (successful_results, failures)

    mock_client.process_jobs_concurrently.side_effect = fake_processor

    with Ingestor(documents=[doc1_path]) as ingestor:
        ingestor.vdb_upload(vdb_op=mock_vdb_op)

        # Should return tuple when return_failures=True
        result = ingestor.ingest(show_progress=False, return_failures=True)

        # Verify it's a tuple with results and failures
        assert isinstance(result, tuple)
        assert len(result) == 2
        results, returned_failures = result
        assert len(results) == 1
        assert results == successful_results  # Should be raw data when save_to_disk() is not used
        assert returned_failures == failures

        # Verify VDB upload was called
        mock_vdb_op.run.assert_called_once()
        called_args = mock_vdb_op.run.call_args[0][0]
        assert called_args == successful_results  # Should be raw data when save_to_disk() is not used
