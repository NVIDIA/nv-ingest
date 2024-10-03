import cudf
from datetime import datetime
from morpheus.messages import ControlMessage

from nv_ingest.modules.telemetry.otel_tracer import extract_annotated_task_results
from nv_ingest.modules.telemetry.otel_tracer import extract_timestamps_from_message


def test_extract_timestamps_single_task():
    msg = ControlMessage()
    msg.set_timestamp("trace::entry::foo", datetime.fromtimestamp(1))
    msg.set_timestamp("trace::exit::foo", datetime.fromtimestamp(2))

    expected_output = {"foo": (int(1e9), int(2e9))}  # Convert seconds to nanoseconds

    result = extract_timestamps_from_message(msg)

    assert result == expected_output


def test_extract_timestamps_no_tasks():
    msg = ControlMessage()

    expected_output = {}

    result = extract_timestamps_from_message(msg)

    assert result == expected_output


def test_extract_annotated_task_results_invalid_metadata():
    msg = ControlMessage()

    # Simulate setting non-annotation metadata and valid annotation metadata
    msg.set_metadata("random::metadata", {"random_key": "value"})  # Should be ignored
    msg.set_metadata("annotation::task1", {"task_id": "task1", "task_result": "success"})

    expected_output = {"task1": "success"}

    result = extract_annotated_task_results(msg)

    assert result == expected_output


def test_extract_annotated_task_results_missing_fields():
    msg = ControlMessage()

    # Simulate setting metadata with missing task_id and task_result
    msg.set_metadata("annotation::task1", {"task_result": "success"})  # Missing task_id (should be skipped)
    msg.set_metadata("annotation::task2", {"task_id": "task2"})  # Missing task_result (should be skipped)

    expected_output = {}

    result = extract_annotated_task_results(msg)

    assert result == expected_output


def test_extract_annotated_task_results_no_annotation_keys():
    msg = ControlMessage()

    # Simulate setting metadata with no annotation keys
    msg.set_metadata("random::metadata", {"random_key": "value"})

    expected_output = {}

    result = extract_annotated_task_results(msg)

    assert result == expected_output
