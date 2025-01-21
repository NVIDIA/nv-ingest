# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime

import pydantic
import pytest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from ....import_checks import CUDA_DRIVER_OK
from ....import_checks import MORPHEUS_IMPORT_OK

if MORPHEUS_IMPORT_OK and CUDA_DRIVER_OK:
    import cudf
    from morpheus.messages import ControlMessage
    from morpheus.messages import MessageMeta

    from nv_ingest.modules.sources.message_broker_task_source import process_message

MODULE_UNDER_TEST = "nv_ingest.modules.sources.message_broker_task_source"


@pytest.fixture
def job_payload():
    return json.dumps(
        {
            "job_payload": {
                "content": ["sample content"],
                "source_name": ["source1"],
                "source_id": ["id1"],
                "document_type": ["pdf"],
            },
            "job_id": "12345",
            "tasks": [
                {
                    "type": "split",
                    "task_properties": {
                        "split_by": "word",
                        "split_length": 100,
                        "split_overlap": 0,
                    },
                },
                {
                    "type": "extract",
                    "task_properties": {
                        "document_type": "pdf",
                        "method": "OCR",
                        "params": {},
                    },
                },
                {"type": "embed", "task_properties": {}},
            ],
        }
    )


# Test Case 1: Valid job with all required fields
@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_process_message_valid_job(job_payload):
    """
    Test that process_message processes a valid job correctly.
    """
    job = json.loads(job_payload)
    ts_fetched = datetime.now()

    # Mock validate_ingest_job to prevent actual validation logic if needed
    result = process_message(job, ts_fetched)

    # Check that result is an instance of ControlMessage
    assert isinstance(result, ControlMessage)

    # Check that the metadata is set correctly
    print(result)
    print(job)
    assert result.get_metadata("job_id") == "12345"
    assert result.get_metadata("response_channel") == "12345"

    # Check that tasks are added
    expected_tasks = job["tasks"]
    tasks_in_message = result.get_tasks()
    assert len(tasks_in_message) == len(expected_tasks)

    # Check that the payload is set correctly
    message_meta = result.payload()
    assert isinstance(message_meta, MessageMeta)

    # Check that the DataFrame contains the job payload
    df = message_meta.copy_dataframe()
    assert isinstance(df, cudf.DataFrame)
    for column in job["job_payload"]:
        assert column in df.columns
        # Convert cudf Series to list for comparison
        assert df[column].to_arrow().to_pylist() == job["job_payload"][column]

    # Since do_trace_tagging is False by default
    assert result.get_metadata("config::add_trace_tagging") is None


# Test Case 2: Job missing 'job_id'
@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_process_message_missing_job_id(job_payload):
    """
    Test that process_message raises an exception when 'job_id' is missing.
    """
    job = json.loads(job_payload)
    job.pop("job_id")
    ts_fetched = datetime.now()

    # We expect validate_ingest_job to raise an exception due to missing 'job_id'
    with patch(f"{MODULE_UNDER_TEST}.validate_ingest_job") as mock_validate_ingest_job:
        mock_validate_ingest_job.side_effect = KeyError("job_id")

        with pytest.raises(KeyError) as exc_info:
            process_message(job, ts_fetched)

        assert "job_id" in str(exc_info.value)
        mock_validate_ingest_job.assert_called_once_with(job)


# Test Case 3: Job missing 'job_payload'
@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_process_message_missing_job_payload(job_payload):
    """
    Test that process_message handles a job missing 'job_payload'.
    """
    job = json.loads(job_payload)
    job.pop("job_payload")
    ts_fetched = datetime.now()

    # We need to allow validate_ingest_job to pass
    with pytest.raises(pydantic.ValidationError) as exc_info:
        process_message(job, ts_fetched)


# Test Case 5: Job with invalid tasks (missing 'type' in a task)
@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_process_message_invalid_tasks(job_payload):
    """
    Test that process_message raises an exception when a task is invalid.
    """
    job = json.loads(job_payload)
    # Remove 'type' from one of the tasks to make it invalid
    job["tasks"][0].pop("type")
    ts_fetched = datetime.now()

    # Since we're not mocking validate_ingest_job, it should raise an exception during validation
    with pytest.raises(Exception) as exc_info:
        process_message(job, ts_fetched)

    # Check that the exception message indicates a validation error
    assert 'task must have a "type"' in str(exc_info.value).lower() or "validation" in str(exc_info.value).lower()


# Test Case 6: Job with tracing options enabled
@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_process_message_with_tracing(job_payload):
    """
    Test that process_message adds tracing metadata when tracing options are enabled.
    """
    job = json.loads(job_payload)
    job["tracing_options"] = {
        "trace": True,
        "ts_send": int(datetime.now().timestamp() * 1e9),  # ts_send in nanoseconds
        "trace_id": "trace-123",
    }
    ts_fetched = datetime.now()

    # Adjust MODULE_NAME based on your actual module name
    MODULE_NAME = "message_broker_task_source"

    # Call the function
    result = process_message(job, ts_fetched)

    # Assertions
    assert isinstance(result, ControlMessage)

    # Check that tracing metadata were added
    assert result.get_metadata("config::add_trace_tagging") is True
    assert result.get_metadata("trace_id") == "trace-123"

    # Check timestamps
    assert result.get_timestamp(f"trace::entry::{MODULE_NAME}") is not None
    assert result.get_timestamp(f"trace::exit::{MODULE_NAME}") is not None
    assert result.get_timestamp("trace::entry::broker_source_network_in") is not None
    assert result.get_timestamp("trace::exit::broker_source_network_in") == ts_fetched
    assert result.get_timestamp("latency::ts_send") is not None


# Test Case 7: Exception occurs during processing and 'job_id' is present
@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_process_message_exception_with_job_id(job_payload):
    """
    Test that process_message handles exceptions and sets metadata when 'job_id' is present.
    """
    job = json.loads(job_payload)
    ts_fetched = datetime.now()

    # Modify job_payload to cause an exception during DataFrame creation
    job["job_payload"] = None  # This should cause an exception when creating DataFrame

    # Call the function
    with pytest.raises(ValidationError):
        _ = process_message(job, ts_fetched)


# Test Case 8: Exception occurs during processing and 'job_id' is missing
@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_process_message_exception_without_job_id(job_payload):
    """
    Test that process_message raises an exception when 'job_id' is missing and an exception occurs.
    """
    job = json.loads(job_payload)
    job.pop("job_id")  # Remove 'job_id' to simulate missing job ID
    ts_fetched = datetime.now()

    # Modify job_payload to cause an exception during DataFrame creation
    job["job_payload"] = None

    with pytest.raises(Exception) as exc_info:
        process_message(job, ts_fetched)
