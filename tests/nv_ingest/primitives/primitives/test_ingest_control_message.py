from datetime import datetime

import pandas as pd

from nv_ingest.primitives.ingest_control_message import ControlMessageTask, IngestControlMessage

import pytest
from pydantic import ValidationError


def test_valid_task():
    """
    Validate that a ControlMessageTask can be successfully created with valid input data.
    """
    data = {
        "name": "Example Task",
        "id": "task-123",
        "properties": {"param1": "value1", "param2": 42},
    }
    task = ControlMessageTask(**data)
    assert task.name == "Example Task"
    assert task.id == "task-123"
    assert task.properties == {"param1": "value1", "param2": 42}


def test_valid_task_without_properties():
    """
    Validate that a ControlMessageTask defaults properties to an empty dictionary when not provided.
    """
    data = {
        "name": "Minimal Task",
        "id": "task-456",
    }
    task = ControlMessageTask(**data)
    assert task.name == "Minimal Task"
    assert task.id == "task-456"
    assert task.properties == {}


def test_missing_required_field_name():
    """
    Validate that creating a ControlMessageTask without the 'name' field raises a ValidationError.
    """
    data = {"id": "task-no-name", "properties": {"some_property": "some_value"}}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("name",)
    assert errors[0]["type"] == "missing"


def test_missing_required_field_id():
    """
    Validate that creating a ControlMessageTask without the 'id' field raises a ValidationError.
    """
    data = {"name": "Task With No ID", "properties": {"some_property": "some_value"}}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("id",)
    assert errors[0]["type"] == "missing"


def test_extra_fields_forbidden():
    """
    Validate that providing extra fields to ControlMessageTask raises a ValidationError.
    """
    data = {"name": "Task With Extras", "id": "task-extra", "properties": {}, "unexpected_field": "foo"}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("unexpected_field",)


def test_properties_accepts_various_types():
    """
    Validate that the 'properties' field accepts various types of values.
    """
    data = {
        "name": "Complex Properties Task",
        "id": "task-complex",
        "properties": {
            "string_prop": "string value",
            "int_prop": 123,
            "list_prop": [1, 2, 3],
            "dict_prop": {"nested": True},
        },
    }
    task = ControlMessageTask(**data)
    assert task.properties["string_prop"] == "string value"
    assert task.properties["int_prop"] == 123
    assert task.properties["list_prop"] == [1, 2, 3]
    assert task.properties["dict_prop"] == {"nested": True}


def test_properties_with_invalid_type():
    """
    Validate that providing an invalid type for 'properties' (e.g., a list) raises a ValidationError.
    """
    data = {"name": "Invalid Properties Task", "id": "task-invalid-props", "properties": ["this", "should", "fail"]}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("properties",)


def test_set_and_get_metadata():
    """
    Validate that set_metadata correctly stores metadata and get_metadata returns the correct value.
    """
    cm = IngestControlMessage()
    cm.set_metadata("key1", "value1")
    assert cm.get_metadata("key1") == "value1"


def test_get_all_metadata():
    """
    Validate that get_metadata returns a copy of all metadata when no key is provided.
    """
    cm = IngestControlMessage()
    cm.set_metadata("key1", "value1")
    cm.set_metadata("key2", "value2")
    all_metadata = cm.get_metadata()
    assert isinstance(all_metadata, dict)
    assert all_metadata == {"key1": "value1", "key2": "value2"}
    all_metadata["key1"] = "modified"
    assert cm.get_metadata("key1") == "value1"


def test_has_metadata():
    """
    Validate that has_metadata returns True if the metadata key exists and False otherwise.
    """
    cm = IngestControlMessage()
    cm.set_metadata("present", 123)
    assert cm.has_metadata("present")
    assert not cm.has_metadata("absent")


def test_list_metadata():
    """
    Validate that list_metadata returns all metadata keys as a list.
    """
    cm = IngestControlMessage()
    keys = ["alpha", "beta", "gamma"]
    for key in keys:
        cm.set_metadata(key, key.upper())
    metadata_keys = cm.list_metadata()
    assert sorted(metadata_keys) == sorted(keys)


def test_set_timestamp_with_datetime():
    """
    Validate that set_timestamp accepts a datetime object and get_timestamp returns the correct value.
    """
    cm = IngestControlMessage()
    dt = datetime(2025, 1, 1, 12, 0, 0)
    cm.set_timestamp("start", dt)
    retrieved = cm.get_timestamp("start")
    assert retrieved == dt


def test_set_timestamp_with_string():
    """
    Validate that set_timestamp accepts an ISO format string and get_timestamp returns the correct datetime object.
    """
    cm = IngestControlMessage()
    iso_str = "2025-01-01T12:00:00"
    dt = datetime.fromisoformat(iso_str)
    cm.set_timestamp("start", iso_str)
    retrieved = cm.get_timestamp("start")
    assert retrieved == dt


def test_set_timestamp_invalid_input():
    """
    Validate that set_timestamp raises a ValueError when provided with an invalid timestamp format or type.
    """
    cm = IngestControlMessage()
    with pytest.raises(ValueError):
        cm.set_timestamp("bad", 123)
    with pytest.raises(ValueError):
        cm.set_timestamp("bad", "not-a-timestamp")


def test_get_timestamp_nonexistent():
    """
    Validate that get_timestamp returns None for a non-existent key when fail_if_nonexist is False.
    """
    cm = IngestControlMessage()
    assert cm.get_timestamp("missing") is None


def test_get_timestamp_nonexistent_fail():
    """
    Validate that get_timestamp raises a KeyError for a non-existent key when fail_if_nonexist is True.
    """
    cm = IngestControlMessage()
    with pytest.raises(KeyError):
        cm.get_timestamp("missing", fail_if_nonexist=True)


def test_get_timestamps():
    """
    Validate that get_timestamps returns a dictionary of all set timestamps.
    """
    cm = IngestControlMessage()
    dt1 = datetime(2025, 1, 1, 12, 0, 0)
    dt2 = datetime(2025, 1, 2, 12, 0, 0)
    cm.set_timestamp("start", dt1)
    cm.set_timestamp("end", dt2)
    timestamps = cm.get_timestamps()
    assert timestamps == {"start": dt1, "end": dt2}
    timestamps["start"] = datetime(2025, 1, 1, 0, 0, 0)
    assert cm.get_timestamp("start") == dt1


def test_filter_timestamp():
    """
    Validate that filter_timestamp returns only those timestamps whose keys match the given regex pattern.
    """
    cm = IngestControlMessage()
    dt1 = datetime(2025, 1, 1, 12, 0, 0)
    dt2 = datetime(2025, 1, 2, 12, 0, 0)
    dt3 = datetime(2025, 1, 3, 12, 0, 0)
    cm.set_timestamp("start", dt1)
    cm.set_timestamp("end", dt2)
    cm.set_timestamp("middle", dt3)
    filtered = cm.filter_timestamp("nothing")
    assert set(filtered.keys()) == set()
    filtered = cm.filter_timestamp("^(s|m)")
    expected_keys = {"start", "middle"}
    assert set(filtered.keys()) == expected_keys
    filtered_e = cm.filter_timestamp("^e")
    assert set(filtered_e.keys()) == {"end"}


def test_remove_existing_task():
    """
    Validate that remove_task successfully removes an existing task.
    """
    cm = IngestControlMessage()
    task = ControlMessageTask(name="Test Task", id="task1", properties={"param": "value"})
    cm.add_task(task)
    assert cm.has_task("task1")
    cm.remove_task("task1")
    assert not cm.has_task("task1")
    tasks = list(cm.get_tasks())
    assert all(t.id != "task1" for t in tasks)


def test_remove_nonexistent_task():
    """
    Validate that calling remove_task on a non-existent task id does not raise an error and does not affect
    existing tasks.
    """
    cm = IngestControlMessage()
    task = ControlMessageTask(name="Test Task", id="task1", properties={"param": "value"})
    cm.add_task(task)
    cm.remove_task("nonexistent")
    assert cm.has_task("task1")
    tasks = list(cm.get_tasks())
    assert any(t.id == "task1" for t in tasks)


def test_payload_get_default():
    """
    Validate that the payload getter returns an empty DataFrame by default.
    """
    cm = IngestControlMessage()
    payload = cm.payload()
    assert isinstance(payload, pd.DataFrame)
    assert payload.empty


def test_payload_set_valid():
    """
    Validate that setting a valid pandas DataFrame as payload updates the payload correctly.
    """
    cm = IngestControlMessage()
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    returned_payload = cm.payload(df)
    pd.testing.assert_frame_equal(returned_payload, df)
    pd.testing.assert_frame_equal(cm.payload(), df)


def test_payload_set_invalid():
    """
    Validate that setting an invalid payload (non-DataFrame) raises a ValueError.
    """
    cm = IngestControlMessage()
    with pytest.raises(ValueError):
        cm.payload("not a dataframe")
