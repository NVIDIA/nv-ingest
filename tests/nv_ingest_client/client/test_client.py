# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random
import re
import uuid
from concurrent.futures import Future
from concurrent.futures import as_completed
from unittest.mock import MagicMock

import pytest
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives.jobs import JobSpec
from nv_ingest_client.primitives.jobs import JobState
from nv_ingest_client.primitives.jobs import JobStateEnum
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import TaskType


class MockClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.counter = 0

    def incr(self, key):
        self.counter += 1
        return self.counter

    def get_client(self):
        return self


class ExtendedMockClient(MockClient):
    def __init__(self, host, port):
        super().__init__(host, port)
        self.submitted_messages = []

    def submit_message(self, job_queue_id, job_spec_str):
        # Simulate message submission by storing it
        self.submitted_messages.append((job_queue_id, job_spec_str))


class ExtendedMockClientWithFailure(ExtendedMockClient):
    def submit_message(self, job_queue_id, job_spec_str):
        if "fail_queue" in job_queue_id:
            raise Exception("Simulated submission failure")
        super().submit_message(job_queue_id, job_spec_str)


class ExtendedMockClientWithFetch(ExtendedMockClientWithFailure):
    def __init__(self, host, port):
        super().__init__(host, port)
        self.messages = {}  # Simulate a Redis key-value store for response channels
        self.deletions = 0

    def fetch_message(self, response_channel, timeout):
        # Simulate fetching a message with a timeout. For simplicity, ignore the timeout in the mock.
        return self.messages.get(response_channel)

    def delete(self, response_channel):
        # Simulate deleting a response channel
        self.deletions += 1


@pytest.fixture
def mock_client_allocator():
    return MagicMock(return_value=MockClient("localhost", 6379))


@pytest.fixture
def extended_mock_client_allocator():
    return MagicMock(return_value=ExtendedMockClientWithFetch("localhost", 6379))


@pytest.fixture
def nv_ingest_client(mock_client_allocator):
    return NvIngestClient(
        message_client_allocator=mock_client_allocator,
        message_client_hostname="localhost",
        message_client_port=6379,
        msg_counter_id="nv-ingest-message-id",
        worker_pool_size=1,
    )


@pytest.fixture
def nv_ingest_client_with_jobs(extended_mock_client_allocator):
    client = NvIngestClient(
        message_client_allocator=extended_mock_client_allocator,
        message_client_hostname="localhost",
        message_client_port=6379,
        msg_counter_id="nv-ingest-message-id",
        worker_pool_size=1,
    )

    job_id = "12345678-1234-5678-1234-567812345678"
    client._job_states = {
        "job1": JobState(JobSpec(job_id="job1"), state=JobStateEnum.PENDING),
        "job_completed": JobState(JobSpec(job_id="job_completed"), state=JobStateEnum.COMPLETED),
        "job2": JobState(JobSpec(job_id="job3"), state=JobStateEnum.PENDING),
        "job3": JobState(JobSpec(job_id="job4"), state=JobStateEnum.PENDING),
        "async_job": JobState(JobSpec(job_id="async_job"), state=JobStateEnum.SUBMITTED),
        "no_submit": JobState(JobSpec(job_id="no_submit"), state=JobStateEnum.CANCELLED),
        job_id: JobState(
            job_spec=JobSpec(
                payload={},
                job_id=job_id,
                tasks=[],
                source_id="source",
                extended_options={},
            ),
            state=JobStateEnum.PENDING,
        ),
    }
    return client


@pytest.fixture
def job_state_submitted_async():
    job_state = JobState(JobSpec(job_id="async_job"), state=JobStateEnum.SUBMITTED)
    job_state.future = MagicMock()
    return job_state


@pytest.fixture
def job_state_processing():
    return JobState(JobSpec(job_id="processing_job"), state=JobStateEnum.PROCESSING)


@pytest.fixture
def job_state_invalid():
    return JobState(JobSpec(job_id="invalid_job"), state=JobStateEnum.COMPLETED)


def test_init(nv_ingest_client, mock_client_allocator):
    assert nv_ingest_client._message_client_hostname == "localhost"
    assert nv_ingest_client._message_client_port == 6379
    assert nv_ingest_client._message_counter_id == "nv-ingest-message-id"
    mock_client_allocator.assert_called_once_with(host="localhost", port=6379)


# _pop_job_state
def test_pop_job_state(nv_ingest_client):
    state = JobState(job_spec=JobSpec("test_job_state"))
    nv_ingest_client._job_states["test_job_id"] = state
    popped_state = nv_ingest_client._pop_job_state("test_job_id")
    assert popped_state == state
    assert "test_job_id" not in nv_ingest_client._job_states


# _generate_job_id
def test_generate_job_id_format(nv_ingest_client):
    job_id = nv_ingest_client._generate_job_id()
    # Regex to match the expected format
    pattern = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_\d+$")
    assert pattern.match(job_id) is not None, "Job ID does not match the expected format."


def test_generate_job_id_uniqueness(nv_ingest_client):
    job_id1 = nv_ingest_client._generate_job_id()
    job_id2 = nv_ingest_client._generate_job_id()
    assert job_id1 != job_id2, "Generated job IDs must be unique."


# _get_and_check_job_state
def test_get_existing_job_state(nv_ingest_client_with_jobs):
    job_state = nv_ingest_client_with_jobs._get_and_check_job_state("job1")
    assert job_state.job_id == "job1" and job_state.state == JobStateEnum.PENDING


# Test handling non-existent jobs
def test_get_non_existent_job_state(nv_ingest_client_with_jobs):
    with pytest.raises(ValueError) as exc_info:
        nv_ingest_client_with_jobs._get_and_check_job_state("non_existent_job")
    assert "does not exist" in str(exc_info.value)


# Test validating job state against a single required state
def test_validate_job_state_single(nv_ingest_client_with_jobs):
    job_state = nv_ingest_client_with_jobs._get_and_check_job_state("job1", required_state=JobStateEnum.PENDING)
    assert job_state.state == JobStateEnum.PENDING


# Test validating job state against multiple acceptable states
def test_validate_job_state_multiple(nv_ingest_client_with_jobs):
    job_state = nv_ingest_client_with_jobs._get_and_check_job_state(
        "job_completed", required_state=[JobStateEnum.CANCELLED, JobStateEnum.COMPLETED]
    )
    assert job_state.state == JobStateEnum.COMPLETED


# Test handling invalid required states
def test_invalid_required_state(nv_ingest_client_with_jobs):
    with pytest.raises(ValueError) as exc_info:
        nv_ingest_client_with_jobs._get_and_check_job_state("job1", required_state=JobStateEnum.SUBMITTED)
    assert "has invalid state" in str(exc_info.value)


# job_count
def test_job_count_with_multiple_jobs(nv_ingest_client_with_jobs):
    """Test that job_count accurately reflects the number of jobs."""
    expected_count = 7  # Adjust based on the number of jobs added in the fixture
    assert nv_ingest_client_with_jobs.job_count() == expected_count, f"Job count should be {expected_count}."


# create_job
def test_successful_job_creation(nv_ingest_client):
    job_id = uuid.uuid4()
    result_id = nv_ingest_client.create_job(
        job_id=job_id,
        payload="value",
        source_id="source_123",
        source_name="source_name",
        document_type="txt",
    )
    assert str(job_id) == result_id, "The created job ID should match the input."


def test_fail_on_duplicate_job_creation(nv_ingest_client):
    job_id = uuid.uuid4()
    result_id = nv_ingest_client.create_job(
        job_id=job_id,
        payload="value",
        source_id="source_123",
        source_name="source_name",
        document_type="txt",
    )
    assert str(job_id) == result_id, "The created job ID should match the input."

    with pytest.raises(ValueError):
        nv_ingest_client.create_job(
            job_id=job_id,
            payload="value",
            source_id="source_123",
            source_name="source_name",
            document_type="txt",
        )


def test_job_creation_with_all_parameters(nv_ingest_client):
    job_id = str(uuid.uuid4())
    payload = {"data": "value"}
    tasks = ["task1", "task2"]
    source_id = "source_123"
    source_name = "source_name.pdf"
    extended_options = {"option1": "value1"}

    result_id = nv_ingest_client.create_job(
        job_id=job_id,
        payload=payload,
        tasks=tasks,
        source_id=source_id,
        source_name=source_name,
        extended_options=extended_options,
    )

    assert job_id == result_id, "The created job ID should match the input."
    assert nv_ingest_client._job_states[job_id].job_spec.payload == payload


def test_duplicate_job_id_handling(nv_ingest_client):
    job_id = str(uuid.uuid4())
    payload = {"data": "value"}
    tasks = ["task1", "task2"]
    source_id = "source_123"
    source_name = "source_name.pdf"
    extended_options = {"option1": "value1"}

    nv_ingest_client.create_job(
        job_id=job_id,
        payload=payload,
        tasks=tasks,
        source_id=source_id,
        source_name=source_name,
        extended_options=extended_options,
    )

    with pytest.raises(ValueError):
        # Second creation attempt with the same job ID should fail
        nv_ingest_client.create_job(
            job_id=job_id,
            payload=payload,
            tasks=tasks,
            source_id=source_id,
            source_name=source_name,
            extended_options=extended_options,
        )


def test_automatic_job_id_generation(nv_ingest_client):
    job_id = str(uuid.uuid4())
    payload = {"data": "value"}
    tasks = ["task1", "task2"]
    source_id = "source_123"
    source_name = "source_name.pdf"
    extended_options = {"option1": "value1"}

    result_id = nv_ingest_client.create_job(
        job_id=job_id,
        payload=payload,
        tasks=tasks,
        source_id=source_id,
        source_name=source_name,
        extended_options=extended_options,
    )

    assert result_id in nv_ingest_client._job_states, "A job ID should be generated and used for tracking."


def test_correct_storage_of_job_details(nv_ingest_client):
    job_id = str(uuid.uuid4())
    payload = {"data": "new_value"}
    tasks = ["task1", "task2"]
    source_id = "source_123"
    source_name = "source_name.pdf"
    extended_options = {"option1": "value1"}

    result_id = nv_ingest_client.create_job(
        job_id=job_id,
        payload=payload,
        tasks=tasks,
        source_id=source_id,
        source_name=source_name,
        extended_options=extended_options,
    )

    stored_job = nv_ingest_client._job_states[result_id]
    assert stored_job.job_spec.payload == payload, "The job's payload should match what was provided."


def test_successful_task_creation(nv_ingest_client_with_jobs):
    job_id = "12345678-1234-5678-1234-567812345678"
    task_type = TaskType.SPLIT
    task_params = {"split_by": "sentence"}

    # Assuming task_factory and task creation are implemented
    nv_ingest_client_with_jobs.create_task(job_id, task_type, task_params)

    # Verify the task was added correctly
    job_state = nv_ingest_client_with_jobs._job_states[job_id]
    assert len(job_state.job_spec._tasks) == 1, "Task was not added to the job"


def test_non_existent_job(nv_ingest_client):
    with pytest.raises(ValueError):
        nv_ingest_client.create_task("nonexistent_job_id", TaskType.SPLIT, {"split_by": "sentence"})


def test_add_task_post_submission(nv_ingest_client_with_jobs):
    # Change job state to simulate post-submission status
    job_id = "12345678-1234-5678-1234-567812345678"
    nv_ingest_client_with_jobs._job_states[job_id].state = JobStateEnum.PROCESSING

    with pytest.raises(ValueError):
        nv_ingest_client_with_jobs.create_task(job_id, TaskType.SPLIT, {"split_by": "sentence"})


def test_parameter_validation(nv_ingest_client_with_jobs):
    job_id = "12345678-1234-5678-1234-567812345678"
    task_type = TaskType.SPLIT
    task_params = {"split_by": "sentence", "split_length": 128}

    nv_ingest_client_with_jobs.create_task(job_id, task_type, task_params)
    job_state = nv_ingest_client_with_jobs._job_states[job_id]
    created_task = job_state.job_spec._tasks[0]

    # Assuming tasks have a way to expose their type and parameters for assertion
    assert isinstance(created_task, SplitTask), "Task type mismatch"


# submit_job
def test_successful_job_submission(nv_ingest_client_with_jobs):
    job_id = "12345678-1234-5678-1234-567812345678"
    job_queue_id = "test_queue"

    nv_ingest_client_with_jobs.submit_job(job_id, job_queue_id)

    # Verify the job was submitted
    mock_client = nv_ingest_client_with_jobs._message_client
    assert len(mock_client.submitted_messages) == 1
    submitted_job_queue_id, _ = mock_client.submitted_messages[0]
    assert submitted_job_queue_id == job_queue_id
    assert nv_ingest_client_with_jobs._job_states[job_id].state == JobStateEnum.SUBMITTED


def test_submit_job_nonexistent_id_raises(nv_ingest_client_with_jobs):
    with pytest.raises(ValueError):
        nv_ingest_client_with_jobs.submit_job("nonexistent_job", "test_queue")


def test_submit_job_invalid_state_raises(nv_ingest_client_with_jobs):
    job_id = "no_submit"

    with pytest.raises(ValueError):
        nv_ingest_client_with_jobs.submit_job(job_id, "test_queue")


def test_submission_failure_sets_job_to_failed(nv_ingest_client_with_jobs):
    job_id = "12345678-1234-5678-1234-567812345678"
    job_queue_id = "fail_queue"

    with pytest.raises(Exception, match="Simulated submission failure"):
        nv_ingest_client_with_jobs.submit_job(job_id, job_queue_id)

    assert (
        nv_ingest_client_with_jobs._job_states[job_id].state == JobStateEnum.FAILED
    ), "Job state should be set to FAILED after a submission failure"


def test_successful_submissions(nv_ingest_client_with_jobs):
    job_ids = ["12345678-1234-5678-1234-567812345678", "job1"]
    job_queue_id = "test_queue"

    responses = nv_ingest_client_with_jobs.submit_job(job_ids, job_queue_id)

    assert len(responses) == len(job_ids), "The number of responses should match the number of submitted jobs"


def test_mixed_submission_outcomes(nv_ingest_client_with_jobs):
    job_ids = ["12345678-1234-5678-1234-567812345678", "fail_submission"]
    job_queue_id = "test_queue"

    with pytest.raises(ValueError, match="does not exist"):
        nv_ingest_client_with_jobs.submit_job(job_ids, job_queue_id)


def test_empty_job_id_list(nv_ingest_client_with_jobs):
    responses = nv_ingest_client_with_jobs.submit_job([], "test_queue")
    assert responses == [], "Response list should be empty for an empty job ID list"


@pytest.mark.parametrize("job_id", ["invalid_job", ""])
def test_invalid_job_ids(nv_ingest_client_with_jobs, job_id):
    with pytest.raises(ValueError):
        nv_ingest_client_with_jobs.submit_job([job_id], "test_queue")


def test_successful_async_submission(nv_ingest_client_with_jobs):
    job_id = "12345678-1234-5678-1234-567812345678"
    job_queue_id = "test_queue"

    nv_ingest_client_with_jobs.submit_job_async(job_id, job_queue_id)

    job_state = nv_ingest_client_with_jobs._job_states[job_id]
    assert job_state.state in [
        JobStateEnum.SUBMITTED_ASYNC,
        JobStateEnum.SUBMITTED,
    ], "The job state should be updated to SUBMITTED_ASYNC"
    assert job_state.future is not None, "A Future should be associated with the job"


def test_submit_async_nonexistent_job_raises(nv_ingest_client_with_jobs):
    with pytest.raises(ValueError):
        nv_ingest_client_with_jobs.submit_job_async("nonexistent_job", "test_queue")


def test_submit_async_invalid_state_raises(nv_ingest_client_with_jobs):
    # Manually set a job to a non-PENDING state for this test
    job_id = "12345678-1234-5678-1234-567812345678"
    nv_ingest_client_with_jobs._job_states[job_id].state = JobStateEnum.COMPLETED

    with pytest.raises(ValueError):
        nv_ingest_client_with_jobs.submit_job_async(job_id, "test_queue")


def test_job_future_result_on_success(nv_ingest_client_with_jobs):
    job_id = "12345678-1234-5678-1234-567812345678"
    job_queue_id = "test_queue"

    nv_ingest_client_with_jobs.submit_job_async(job_id, job_queue_id)

    # Simulate job completion
    future = nv_ingest_client_with_jobs._job_states[job_id].future

    result = future.result(timeout=5)
    assert result == [None], "The future's result should reflect the job's success"


def test_job_future_result_on_failure(nv_ingest_client_with_jobs):
    job_id = "12345678-1234-5678-1234-567812345678"
    job_queue_id = "fail_queue"  # Assume this queue simulates a failure scenario

    future_map = nv_ingest_client_with_jobs.submit_job_async(job_id, job_queue_id)

    with pytest.raises(Exception, match="Simulated submission failure"):
        for future in future_map:
            future.result(timeout=5)


def test_successful_multiple_job_submissions(nv_ingest_client_with_jobs):
    job_ids = ["job1", "job2", "job3"]
    job_queue_id = "test_queue"

    futures = nv_ingest_client_with_jobs.submit_job_async(job_ids, job_queue_id)

    assert len(futures) == len(job_ids), "Should return the same number of futures as job IDs"
    assert all(
        isinstance(future, Future) for future in futures
    ), "Each item in the returned list should be a Future object"


def test_successful_multiple_job_submissions_failure(nv_ingest_client_with_jobs):
    job_ids = ["job1", "job2", "job_completed"]
    job_queue_id = "test_queue"

    random.shuffle(job_ids)
    with pytest.raises(ValueError):
        nv_ingest_client_with_jobs.submit_job(job_ids, job_queue_id)


def test_successful_multiple_job_submissions_async(nv_ingest_client_with_jobs):
    job_ids = ["job1", "job2", "job3"]
    job_queue_id = "test_queue"

    futures = nv_ingest_client_with_jobs.submit_job_async(job_ids, job_queue_id)

    assert len(futures) == len(job_ids), "Should return the same number of futures as job IDs"
    assert all(
        isinstance(future, Future) for future in futures
    ), "Each item in the returned list should be a Future object"


def test_empty_job_id_list_async(nv_ingest_client_with_jobs):
    job_queue_id = "test_queue"
    futures = nv_ingest_client_with_jobs.submit_job_async([], job_queue_id)

    assert futures == {}, "Submitting an empty job ID list should return an empty list of futures"


@pytest.mark.parametrize("job_id", ["job1", "job2", "job3"])
def test_futures_reflect_submission_outcome(nv_ingest_client_with_jobs, job_id):
    job_queue_id = "test_queue"

    future_dict = nv_ingest_client_with_jobs.submit_job_async([job_id], job_queue_id)

    for future in future_dict:
        assert isinstance(future, Future), "The method should return a Future object"


def test_fetch_job_result_after_successful_submission(nv_ingest_client_with_jobs):
    job_ids = ["job1", "job2"]
    job_queue_id = "test_queue"

    # Simulate successful job submissions and retrieve futures
    _ = nv_ingest_client_with_jobs.submit_job(job_ids, job_queue_id)

    # Assume ExtendedMockClient simulates responses for submitted jobs
    for job_id in job_ids:
        response_channel = f"response_{job_id}"
        # Double-encode the dictionary
        double_encoded_json = json.dumps({"result": "success"})
        nv_ingest_client_with_jobs._message_client.get_client().messages[
            response_channel
        ] = f'{{"data": {double_encoded_json}}}'

    # Fetch job results
    for job_id in job_ids:
        result = nv_ingest_client_with_jobs.fetch_job_result(job_id, 5)[0]
        assert result[0] == {"result": "success"}, f"The fetched job result for {job_id} should be successful"


def test_fetch_job_result_async_after_successful_submission(nv_ingest_client_with_jobs):
    job_ids = ["job1", "job2"]
    job_queue_id = "test_queue"

    # Simulate successful job submissions and retrieve futures
    futures = nv_ingest_client_with_jobs.submit_job_async(job_ids, job_queue_id)
    for _ in as_completed(futures):
        pass

    # Assume ExtendedMockClient simulates responses for submitted jobs
    for job_id in job_ids:
        response_channel = f"response_{job_id}"
        # Double-encode the dictionary
        double_encoded_json = json.dumps({"result": "success"})
        nv_ingest_client_with_jobs._message_client.get_client().messages[
            response_channel
        ] = f'{{"data": {double_encoded_json}}}'

    # Fetch job results
    for job_id, future in zip(job_ids, futures):
        futures_dict = nv_ingest_client_with_jobs.fetch_job_result_async(job_id, 5)

        for future in futures_dict.keys():
            result = future.result()[0]
            assert result[0] == {"result": "success"}, f"The fetched job result for {job_id} should be successful"


def test_fetch_job_results_async_after_successful_submission(
    nv_ingest_client_with_jobs,
):
    job_ids = ["job1", "job2"]
    job_queue_id = "test_queue"

    # Simulate successful job submissions and retrieve futures
    futures = nv_ingest_client_with_jobs.submit_job_async(job_ids, job_queue_id)
    for _ in as_completed(futures):
        pass

    # Assume ExtendedMockClient simulates responses for submitted jobs
    for job_id in job_ids:
        response_channel = f"response_{job_id}"
        # Double-encode the dictionary
        double_encoded_json = json.dumps({"result": "success"})
        nv_ingest_client_with_jobs._message_client.get_client().messages[
            response_channel
        ] = f'{{"data": {double_encoded_json}}}'

    # Fetch job results
    for job_id, future in zip(job_ids, futures):
        futures = nv_ingest_client_with_jobs.fetch_job_result_async([job_id], 5)

        for future in as_completed(futures.keys()):
            result = future.result()[0]
            assert result[0] == {"result": "success"}, f"The fetched job result for {job_id} should be successful"
