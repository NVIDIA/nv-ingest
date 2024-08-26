# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import Future
from uuid import uuid4

import pytest
from nv_ingest_client.primitives.jobs import JobSpec
from nv_ingest_client.primitives.jobs import JobState
from nv_ingest_client.primitives.jobs import JobStateEnum


# Helper function to create a JobState instance with default parameters
def create_job_state(
    job_id=str(uuid4()),
    state=JobStateEnum.PENDING,
    future=None,
    response=None,
    response_channel=None,
):
    return JobState(
        job_spec=JobSpec(job_id=job_id),
        state=state,
        future=future,
        response=response,
        response_channel=response_channel,
    )


# Test initialization and property getters
def test_job_state_initialization():
    job_id = str(uuid4())
    state = JobStateEnum.SUBMITTED
    future = Future()
    response = {"result": "success"}
    response_channel = "channel1"

    job_state = JobState(JobSpec(job_id=job_id), state, future, response, response_channel)

    assert job_state.job_id == job_id
    assert job_state.state == state
    assert job_state.future is future
    assert job_state.response == response
    assert job_state.response_channel == response_channel


# Test state transition rules
@pytest.mark.parametrize(
    "initial_state, next_state",
    [
        (JobStateEnum.PENDING, JobStateEnum.SUBMITTED),
        (JobStateEnum.SUBMITTED, JobStateEnum.PROCESSING),
        (JobStateEnum.PROCESSING, JobStateEnum.COMPLETED),
    ],
)
def test_valid_state_transitions(initial_state, next_state):
    job_state = create_job_state(state=initial_state)
    job_state.state = next_state
    assert job_state.state == next_state


# Test invalid state transitions
@pytest.mark.parametrize(
    "initial_state, invalid_next_state",
    [
        (JobStateEnum.COMPLETED, JobStateEnum.PENDING),
        (JobStateEnum.FAILED, JobStateEnum.PROCESSING),
        (JobStateEnum.CANCELLED, JobStateEnum.SUBMITTED),
    ],
)
def test_invalid_state_transitions(initial_state, invalid_next_state):
    job_state = create_job_state(state=initial_state)
    with pytest.raises(ValueError):
        job_state.state = invalid_next_state


# Test setting job_id and response_channel in non-PENDING states
@pytest.mark.parametrize(
    "attribute, value",
    [
        ("job_id", str(uuid4())),
        ("response_channel", "new_channel"),
    ],
)
def test_setting_job_id_and_response_channel_in_non_pending_state(attribute, value):
    job_state = create_job_state(state=JobStateEnum.PROCESSING)
    with pytest.raises(ValueError):
        setattr(job_state, attribute, value)


# Test setting future and response in terminal states
@pytest.mark.parametrize(
    "attribute, value",
    [
        ("future", Future()),
        ("response", {"result": "error"}),
    ],
)
@pytest.mark.parametrize("state", [JobStateEnum.COMPLETED, JobStateEnum.FAILED, JobStateEnum.CANCELLED])
def test_setting_future_and_response_in_terminal_states(attribute, value, state):
    job_state = create_job_state(state=state)
    setattr(job_state, attribute, value)


# Test valid setting of job_id and response_channel in PENDING state
@pytest.mark.parametrize(
    "attribute, value",
    [
        ("job_id", str(uuid4())),
        ("response_channel", "channel2"),
    ],
)
def test_valid_setting_of_job_id_and_response_channel(attribute, value):
    job_state = create_job_state()
    setattr(job_state, attribute, value)
    assert getattr(job_state, attribute) == value


def test_job_try_set_state_after_preflight():
    job_state = create_job_state(state=JobStateEnum.COMPLETED)

    with pytest.raises(ValueError):
        job_state.state = JobStateEnum.PROCESSING


def test_job_try_set_jobspec():
    job_state = create_job_state(state=JobStateEnum.PENDING)
    job_spec = JobSpec()
    job_state.job_spec = job_spec


def test_job_try_set_jobspec_after_preflight():
    job_state = create_job_state(state=JobStateEnum.COMPLETED)
    job_spec = JobSpec()

    with pytest.raises(ValueError):
        job_state.job_spec = job_spec
