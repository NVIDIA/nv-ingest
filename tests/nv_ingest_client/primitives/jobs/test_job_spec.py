# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Dict

import pytest
from nv_ingest_client.primitives.jobs.job_spec import BatchJobSpec
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from nv_ingest_client.primitives.tasks import Task


# Assuming the Task class has a to_dict method
class MockTask(Task):
    def to_dict(self) -> Dict:
        return {"task": "mocktask"}


# Fixture to create a JobSpec instance
@pytest.fixture
def job_spec_fixture() -> JobSpec:
    return JobSpec(
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


# Test initializing BatchJobSpec with JobSpec objects
def test_initialization_with_job_specs(job_spec_fixture):
    batch = BatchJobSpec([job_spec_fixture, job_spec_fixture])
    assert len(batch.job_specs) == 2
    assert batch.job_specs[0] == job_spec_fixture
    assert batch.job_specs[1] == job_spec_fixture


# Test adding a JobSpec to BatchJobSpec
def test_add_job_spec(job_spec_fixture):
    batch = BatchJobSpec()
    assert len(batch.job_specs) == 0

    batch.add_job_spec(job_spec_fixture)
    assert len(batch.job_specs) == 1
    assert batch.job_specs[0] == job_spec_fixture


# Test adding a Task to JobSpecs in BatchJobSpec
def test_add_task(job_spec_fixture):
    mock_task = MockTask()

    batch = BatchJobSpec([job_spec_fixture])
    batch.add_task(mock_task)

    # Ensure the task has been added to the job spec
    for task in batch.job_specs[0]._tasks:
        assert task.to_dict() == {"task": "mocktask"}


# Test converting BatchJobSpec to dictionary
def test_to_dict(job_spec_fixture):
    batch = BatchJobSpec([job_spec_fixture])
    result = batch.to_dict()

    assert len(result) == 1
    assert result[0]["job_payload"]["content"] == [{"key": "value"}]
    assert len(result[0]["tasks"]) == 1
    assert result[0]["tracing_options"] == {"option1": "value1"}


# Test string representation of BatchJobSpec
def test_str_representation(job_spec_fixture):
    batch = BatchJobSpec([job_spec_fixture])
    result = str(batch)

    assert "source-id: source123" in result
    assert "task count: 1" in result
