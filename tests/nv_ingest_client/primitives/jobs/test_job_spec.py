# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Dict
from unittest.mock import Mock

import pytest
from nv_ingest_client.primitives.jobs.job_spec import BatchJobSpec
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from nv_ingest_client.primitives.tasks import Task


# Assuming the Task class has a to_dict method
class MockTask(Task):
    def to_dict(self) -> Dict:
        return {"document_type": "pdf", "task": "mocktask"}


# Fixture to create a JobSpec instance
@pytest.fixture
def job_spec_fixture() -> JobSpec:
    return JobSpec(
        document_type="pdf",
        payload={"key": "value"},
        tasks=[MockTask()],
        source_id="source123",
        source_name="source123.pdf",
        extended_options={"tracing_options": {"option1": "value1"}},
    )


# Test initialization
def test_job_spec_initialization():
    job_spec = JobSpec(
        payload={"key": "value"},
        tasks=[MockTask()],
        source_id="source123",
        extended_options={"option1": "value1"},
    )

    assert job_spec.payload == {"key": "value"}
    assert len(job_spec._tasks) == 1
    assert job_spec.source_id == "source123"
    assert job_spec._extended_options == {"option1": "value1"}


# Test to_dict method
def test_to_dict(job_spec_fixture):
    job_dict = job_spec_fixture.to_dict()
    assert job_dict["job_payload"]["content"] == [{"key": "value"}]
    assert len(job_dict["tasks"]) == 1
    assert job_dict["tracing_options"] == {"option1": "value1"}


# Test add_task method
def test_add_task(job_spec_fixture):
    new_task = MockTask()
    job_spec_fixture.add_task(new_task)
    assert len(job_spec_fixture._tasks) == 2


# Test add_task method with invalid task
def test_add_invalid_task(job_spec_fixture):
    with pytest.raises(ValueError):
        job_spec_fixture.add_task("not_a_task")


# Test payload property getter and setter
def test_payload_getter_setter(job_spec_fixture):
    job_spec_fixture.payload = {"new_key": "new_value"}
    assert job_spec_fixture.payload == {"new_key": "new_value"}


# Test job_id property getter and setter
def test_job_id_getter_setter(job_spec_fixture):
    new_job_id = uuid.uuid4()
    job_spec_fixture.job_id = new_job_id
    assert job_spec_fixture.job_id == new_job_id


# Test __str__ method
def test_str_method(job_spec_fixture):
    job_spec_str = str(job_spec_fixture)
    assert "source-id: source123" in job_spec_str
    assert "task count: 1" in job_spec_str


def test_set_properties():
    job_spec = JobSpec()

    job_spec.source_id = "source456"
    assert job_spec.source_id == "source456"

    job_spec.source_name = "source456.pdf"
    assert job_spec.source_name == "source456.pdf"


@pytest.fixture
def batch_job_spec_fixture(job_spec_fixture) -> BatchJobSpec:
    batch_job_spec = BatchJobSpec()
    batch_job_spec.add_job_spec(job_spec_fixture)
    return batch_job_spec


@pytest.fixture
def batch_job_spec_fixture(job_spec_fixture) -> BatchJobSpec:
    batch_job_spec = BatchJobSpec()
    batch_job_spec.add_job_spec(job_spec_fixture)
    return batch_job_spec


def test_init_with_job_specs(job_spec_fixture):
    batch_job_spec = BatchJobSpec([job_spec_fixture])

    assert "pdf" in batch_job_spec._file_type_to_job_spec
    assert job_spec_fixture in batch_job_spec._file_type_to_job_spec["pdf"]


def test_init_with_files(mocker, job_spec_fixture):
    mocker.patch("nv_ingest_client.util.util.generate_matching_files", return_value=["file1.pdf"])
    mocker.patch("nv_ingest_client.util.util.create_job_specs_for_batch", return_value=[job_spec_fixture])

    batch_job_spec = BatchJobSpec(["file1.pdf"])

    # Verify that the files were processed and job specs were created
    assert "pdf" in batch_job_spec._file_type_to_job_spec
    assert len(batch_job_spec._file_type_to_job_spec["pdf"]) > 0


def test_add_task_to_specific_document_type(batch_job_spec_fixture):
    task = MockTask()

    # Add task to jobs with document_type 'pdf'
    batch_job_spec_fixture.add_task(task, document_type="pdf")

    # Assert that the task was added to the JobSpec with document_type 'pdf'
    for job_spec in batch_job_spec_fixture._file_type_to_job_spec["pdf"]:
        assert task in job_spec._tasks


def test_add_task_to_inferred_document_type(batch_job_spec_fixture):
    task = MockTask()

    # Add task without specifying document_type, should infer from task's to_dict
    batch_job_spec_fixture.add_task(task)

    # Assert that the task was added to the JobSpec with the inferred document_type 'pdf'
    for job_spec in batch_job_spec_fixture._file_type_to_job_spec["pdf"]:
        assert task in job_spec._tasks


def test_add_task_to_all_job_specs(batch_job_spec_fixture):
    # Mock a task without a document_type
    task = MockTask()
    task.to_dict = Mock(return_value={"task": "mocktask"})  # No document_type returned

    # Add task without document_type, it should add to all job specs
    batch_job_spec_fixture.add_task(task)

    # Assert that the task was added to all job specs in the batch
    for job_specs in batch_job_spec_fixture._file_type_to_job_spec.values():
        for job_spec in job_specs:
            assert task in job_spec._tasks


def test_add_task_raises_value_error_for_invalid_task(batch_job_spec_fixture):
    # Create an invalid task that doesn't derive from Task
    invalid_task = object()

    # Expect a ValueError when adding an invalid task
    with pytest.raises(ValueError, match="Task must derive from nv_ingest_client.primitives.Task class"):
        batch_job_spec_fixture.add_task(invalid_task)


def test_to_dict(batch_job_spec_fixture):
    result = batch_job_spec_fixture.to_dict()

    assert isinstance(result, dict)
    assert "pdf" in result
    assert len(result["pdf"]) > 0


def test_str_method(batch_job_spec_fixture):
    result = str(batch_job_spec_fixture)

    assert "pdf" in result
    assert "source123" in result
