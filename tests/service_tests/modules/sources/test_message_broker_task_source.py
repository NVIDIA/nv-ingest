# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from datetime import datetime
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from nv_ingest.framework.orchestration.morpheus.modules.sources.message_broker_task_source import (
    process_message,
    fetch_and_process_messages,
)

import nv_ingest.framework.orchestration.morpheus.modules.sources.message_broker_task_source as module_under_test

# Define the module under test.
MODULE_UNDER_TEST = f"{module_under_test.__name__}"


# -----------------------------------------------------------------------------
# Dummy Classes for Testing (for client and BaseModel response simulation)
# -----------------------------------------------------------------------------


class DummyValidatedConfig:
    def __init__(self, task_queue):
        self.task_queue = task_queue


class DummyResponse(BaseModel):
    response_code: int
    response: str  # JSON string


class DummyClient:
    """
    A dummy client whose fetch_message method returns values from a given list.
    Instead of raising KeyboardInterrupt when responses are exhausted,
    it returns None (which in production causes the loop to continue).
    """

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def fetch_message(self, task_queue, count, override_fetch_mode=None):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        else:
            return None


# -----------------------------------------------------------------------------
# Tests for process_message using mocks
# -----------------------------------------------------------------------------


@patch(f"{MODULE_UNDER_TEST}.MODULE_NAME", new="dummy_module")
@patch(f"{MODULE_UNDER_TEST}.format_trace_id", return_value="trace-98765")
@patch(f"{MODULE_UNDER_TEST}.annotate_cm")
@patch(f"{MODULE_UNDER_TEST}.validate_ingest_job")
@patch(f"{MODULE_UNDER_TEST}.ControlMessageTask")
@patch(f"{MODULE_UNDER_TEST}.IngestControlMessage")
def test_process_message_normal(
    mock_IngestControlMessage,
    mock_ControlMessageTask,
    mock_validate_ingest_job,
    mock_annotate_cm,
    mock_format_trace_id,
):
    """
    Test that process_message correctly processes a valid job.
    The job dict contains:
      - "job_id": identifier
      - "job_payload": a list of dicts to be converted to a DataFrame
      - "tasks": a list of task dicts (each with optional id, type, task_properties)
      - "tracing_options": options that trigger trace tagging.
    Expect that the returned control message:
      - receives a payload DataFrame built from job_payload,
      - has metadata "job_id" and "response_channel" set to the job_id,
      - is annotated with message "Created",
      - has tasks added (one per task in the job),
      - and has trace-related metadata/timestamps set.
    """
    # Create a fake control message instance.
    fake_cm = MagicMock()
    mock_IngestControlMessage.return_value = fake_cm

    # Simulate ControlMessageTask instances.
    fake_task_instance = MagicMock()
    fake_task_instance.model_dump.return_value = {"id": "task1", "type": "process", "properties": {"p": 1}}
    fake_task_instance2 = MagicMock()
    fake_task_instance2.model_dump.return_value = {"id": "auto", "type": "unknown", "properties": {"p": 2}}
    mock_ControlMessageTask.side_effect = [fake_task_instance, fake_task_instance2]

    # Build a valid job dictionary.
    job = {
        "job_id": "job123",
        "job_payload": [{"field": "value1"}, {"field": "value2"}],
        "tasks": [
            {"id": "task1", "type": "process", "task_properties": {"p": 1}},
            {"task_properties": {"p": 2}},
        ],
        "tracing_options": {
            "trace": True,
            "ts_send": datetime.now().timestamp() * 1e9,  # nanoseconds
            "trace_id": 98765,
        },
    }
    # Make a copy because process_message pops keys.
    job_copy = copy.deepcopy(job)
    ts_fetched = datetime.now()

    result_cm = process_message(job_copy, ts_fetched)

    # Verify that payload() was called with a DataFrame built from job_payload.
    fake_cm.payload.assert_called_once()
    df_arg = fake_cm.payload.call_args[0][0]
    pd.testing.assert_frame_equal(df_arg, pd.DataFrame(job["job_payload"]))

    # Verify metadata calls.
    fake_cm.set_metadata.assert_any_call("job_id", "job123")
    fake_cm.set_metadata.assert_any_call("response_channel", "job123")
    mock_annotate_cm.assert_called_with(fake_cm, message="Created")
    # Verify that tasks were added.
    assert fake_cm.add_task.call_count == 2
    # Check trace-related metadata/timestamps.
    trace_meta_calls = [call for call in fake_cm.set_metadata.call_args_list if "trace" in call[0][0]]
    trace_timestamp_calls = [call for call in fake_cm.set_timestamp.call_args_list if "trace" in call[0][0]]
    assert trace_meta_calls or trace_timestamp_calls
    fake_cm.set_metadata.assert_any_call("trace_id", "trace-98765")


@patch(f"{MODULE_UNDER_TEST}.MODULE_NAME", new="dummy_module")
@patch(f"{MODULE_UNDER_TEST}.annotate_cm")
@patch(f"{MODULE_UNDER_TEST}.validate_ingest_job", side_effect=ValueError("Invalid job"))
@patch(f"{MODULE_UNDER_TEST}.IngestControlMessage")
def test_process_message_validation_failure_with_job_id(
    mock_IngestControlMessage, mock_validate_ingest_job, mock_annotate_cm
):
    """
    Test that when validate_ingest_job fails—even if the job dict contains a 'job_id'—
    process_message re‑raises the exception.
    """
    fake_cm = MagicMock()
    mock_IngestControlMessage.return_value = fake_cm

    job = {
        "job_id": "job_fail",
        "job_payload": [{"a": "b"}],
        "tasks": [],
        "tracing_options": {},
        "invalid": True,  # Triggers failure.
    }
    job_copy = copy.deepcopy(job)
    ts_fetched = datetime.now()

    with pytest.raises(ValueError, match="Invalid job"):
        process_message(job_copy, ts_fetched)


@patch(f"{MODULE_UNDER_TEST}.validate_ingest_job", side_effect=ValueError("Invalid job"))
@patch(f"{MODULE_UNDER_TEST}.IngestControlMessage")
def test_process_message_validation_failure_no_job_id(mock_IngestControlJob, mock_validate_ingest_job):
    """
    Test that if validate_ingest_job fails and there is no 'job_id' in the job dict,
    process_message re‑raises the exception.
    """
    fake_cm = MagicMock()
    mock_IngestControlJob.return_value = fake_cm

    job = {
        "job_payload": [{"a": "b"}],
        "tasks": [],
        "tracing_options": {},
        "invalid": True,
    }
    ts_fetched = datetime.now()

    with pytest.raises(ValueError, match="Invalid job"):
        process_message(copy.deepcopy(job), ts_fetched)


# -----------------------------------------------------------------------------
# Tests for fetch_and_process_messages using mocks
# -----------------------------------------------------------------------------


@patch(f"{MODULE_UNDER_TEST}.process_message", return_value="processed")
def test_fetch_and_process_messages_dict_job(mock_process_message):
    """
    Test that when the client returns a job as a dictionary,
    fetch_and_process_messages yields a processed control message.
    """
    job = {
        "job_id": "job_dict",
        "job_payload": [{"x": 1}, {"x": 2}],
        "tasks": [],
        "tracing_options": {},
    }
    client = DummyClient([job])
    config = DummyValidatedConfig(task_queue="queue1")

    gen = fetch_and_process_messages(client, config)
    result = next(gen)
    gen.close()
    assert result == "processed"
    mock_process_message.assert_called_once()


@patch(f"{MODULE_UNDER_TEST}.process_message", return_value="processed")
def test_fetch_and_process_messages_base_model_job(mock_process_message):
    """
    Test that when the client returns a job as a BaseModel with response_code 0,
    the job.response is JSON-decoded and processed.
    """
    job_dict = {
        "job_id": "job_bm",
        "job_payload": [{"y": "a"}],
        "tasks": [],
        "tracing_options": {},
    }
    response_obj = DummyResponse(response_code=0, response=json.dumps(job_dict))
    client = DummyClient([response_obj])
    config = DummyValidatedConfig(task_queue="queue1")

    gen = fetch_and_process_messages(client, config)
    result = next(gen)
    gen.close()
    assert result == "processed"
    mock_process_message.assert_called_once()


@patch(f"{MODULE_UNDER_TEST}.process_message", return_value="processed")
def test_fetch_and_process_messages_skip_on_nonzero_response_code(mock_process_message):
    """
    Test that when the client returns a BaseModel job with a nonzero response_code,
    the job is skipped and not processed.
    Then a subsequent valid dictionary job is processed.
    """
    response_bad = DummyResponse(response_code=1, response="{}")
    job = {
        "job_id": "job_valid",
        "job_payload": [{"z": 100}],
        "tasks": [],
        "tracing_options": {},
    }
    client = DummyClient([response_bad, job])
    config = DummyValidatedConfig(task_queue="queue1")

    gen = fetch_and_process_messages(client, config)
    result = next(gen)
    gen.close()
    assert result == "processed"
    assert mock_process_message.call_count == 1


def test_fetch_and_process_messages_timeout_error():
    """
    Test that if client.fetch_message raises a TimeoutError,
    fetch_and_process_messages catches it and continues to the next message.
    """
    call = [0]

    class TimeoutThenValidClient:
        def fetch_message(self, task_queue, count, override_fetch_mode=None):
            if call[0] == 0:
                call[0] += 1
                raise TimeoutError("Timeout")
            else:
                return DummyResponse(
                    response_code=0,
                    response=json.dumps(
                        {
                            "job_id": "job_after_timeout",
                            "job_payload": [{"a": "b"}],
                            "tasks": [],
                            "tracing_options": {},
                        }
                    ),
                )

    config = DummyValidatedConfig(task_queue="queue1")

    with patch(f"{MODULE_UNDER_TEST}.process_message", return_value="processed") as mock_process_message:
        gen = fetch_and_process_messages(TimeoutThenValidClient(), config)
        result = next(gen)
        gen.close()
        assert result == "processed"
        mock_process_message.assert_called_once()


def test_fetch_and_process_messages_exception_handling():
    """
    Test that if client.fetch_message raises a generic Exception,
    fetch_and_process_messages logs the error and continues fetching.
    """
    call = [0]

    class ExceptionThenValidClient:
        def fetch_message(self, task_queue, count, override_fetch_mode=None):
            if call[0] == 0:
                call[0] += 1
                raise Exception("Generic error")
            else:
                return DummyResponse(
                    response_code=0,
                    response=json.dumps(
                        {
                            "job_id": "job_after_exception",
                            "job_payload": [{"c": "d"}],
                            "tasks": [],
                            "tracing_options": {},
                        }
                    ),
                )

    config = DummyValidatedConfig(task_queue="queue1")

    # Patch process_message and the logger so we can assert they're called appropriately.
    with patch(f"{MODULE_UNDER_TEST}.process_message", return_value="processed") as mock_process_message, patch(
        f"{MODULE_UNDER_TEST}.logger"
    ) as mock_logger:
        # Create the generator.
        gen = fetch_and_process_messages(ExceptionThenValidClient(), config)

        # The generator should swallow the Exception, log it, then yield the processed message.
        result = next(gen)
        gen.close()

        # Assert that the processed result is returned.
        assert result == "processed"

        # Assert that the process_message function was called once.
        mock_process_message.assert_called_once()

        # Assert that an exception was logged. Adjust the number as needed.
        assert mock_logger.exception.call_count > 0
