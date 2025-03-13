import re

from nv_ingest_api.primitives.control_message_task import ControlMessageTask
from nv_ingest_api.primitives.ingest_control_message import IngestControlMessage

import pytest
import pandas as pd
from datetime import datetime
from pydantic import ValidationError


def test_valid_task():
    data = {
        "type": "Example Task",
        "id": "task-123",
        "properties": {"param1": "value1", "param2": 42},
    }
    task = ControlMessageTask(**data)
    assert task.type == "Example Task"
    assert task.id == "task-123"
    assert task.properties == {"param1": "value1", "param2": 42}


def test_valid_task_without_properties():
    data = {"type": "Minimal Task", "id": "task-456"}
    task = ControlMessageTask(**data)
    assert task.type == "Minimal Task"
    assert task.id == "task-456"
    assert task.properties == {}


def test_missing_required_field_name():
    data = {"id": "task-no-name", "properties": {"some_property": "some_value"}}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("type",)
    assert errors[0]["type"] == "missing"


def test_missing_required_field_id():
    data = {"type": "Task With No ID", "properties": {"some_property": "some_value"}}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("id",)
    assert errors[0]["type"] == "missing"


def test_extra_fields_forbidden():
    data = {"type": "Task With Extras", "id": "task-extra", "properties": {}, "unexpected_field": "foo"}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("unexpected_field",)


def test_properties_accepts_various_types():
    data = {
        "type": "Complex Properties Task",
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
    data = {"type": "Invalid Properties Task", "id": "task-invalid-props", "properties": ["this", "should", "fail"]}
    with pytest.raises(ValidationError) as exc_info:
        ControlMessageTask(**data)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("properties",)


def test_set_and_get_metadata():
    cm = IngestControlMessage()
    cm.set_metadata("key1", "value1")
    # Test string lookup remains unchanged.
    assert cm.get_metadata("key1") == "value1"


def test_get_all_metadata():
    cm = IngestControlMessage()
    cm.set_metadata("key1", "value1")
    cm.set_metadata("key2", "value2")
    all_metadata = cm.get_metadata()
    assert isinstance(all_metadata, dict)
    assert all_metadata == {"key1": "value1", "key2": "value2"}
    # Ensure a copy is returned.
    all_metadata["key1"] = "modified"
    assert cm.get_metadata("key1") == "value1"


def test_has_metadata():
    cm = IngestControlMessage()
    cm.set_metadata("present", 123)
    # Test string lookup remains unchanged.
    assert cm.has_metadata("present")
    assert not cm.has_metadata("absent")


def test_list_metadata():
    cm = IngestControlMessage()
    keys = ["alpha", "beta", "gamma"]
    for key in keys:
        cm.set_metadata(key, key.upper())
    metadata_keys = cm.list_metadata()
    assert sorted(metadata_keys) == sorted(keys)


def test_get_metadata_regex_match():
    """
    Validate that get_metadata returns a dict of all matching metadata entries when a regex is provided.
    """
    cm = IngestControlMessage()
    cm.set_metadata("alpha", 1)
    cm.set_metadata("beta", 2)
    cm.set_metadata("gamma", 3)
    # Use a regex to match keys that start with "a" or "g".
    pattern = re.compile("^(a|g)")
    result = cm.get_metadata(pattern)
    expected = {"alpha": 1, "gamma": 3}
    assert result == expected


def test_get_metadata_regex_no_match():
    """
    Validate that get_metadata returns the default value when a regex is provided but no keys match.
    """
    cm = IngestControlMessage()
    cm.set_metadata("alpha", 1)
    cm.set_metadata("beta", 2)
    pattern = re.compile("z")
    # Return default as an empty dict when no match is found.
    result = cm.get_metadata(pattern, default_value={})
    assert result == {}


def test_has_metadata_regex_match():
    """
    Validate that has_metadata returns True if any metadata key matches the regex.
    """
    cm = IngestControlMessage()
    cm.set_metadata("key1", "value1")
    cm.set_metadata("other", "value2")
    assert cm.has_metadata(re.compile("^key"))
    assert not cm.has_metadata(re.compile("nonexistent"))


def test_set_timestamp_with_datetime():
    cm = IngestControlMessage()
    dt = datetime(2025, 1, 1, 12, 0, 0)
    cm.set_timestamp("start", dt)
    retrieved = cm.get_timestamp("start")
    assert retrieved == dt


def test_set_timestamp_with_string():
    cm = IngestControlMessage()
    iso_str = "2025-01-01T12:00:00"
    dt = datetime.fromisoformat(iso_str)
    cm.set_timestamp("start", iso_str)
    retrieved = cm.get_timestamp("start")
    assert retrieved == dt


def test_set_timestamp_invalid_input():
    cm = IngestControlMessage()
    with pytest.raises(ValueError):
        cm.set_timestamp("bad", 123)
    with pytest.raises(ValueError):
        cm.set_timestamp("bad", "not-a-timestamp")


def test_get_timestamp_nonexistent():
    cm = IngestControlMessage()
    assert cm.get_timestamp("missing") is None


def test_get_timestamp_nonexistent_fail():
    cm = IngestControlMessage()
    with pytest.raises(KeyError):
        cm.get_timestamp("missing", fail_if_nonexist=True)


def test_get_timestamps():
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
    cm = IngestControlMessage()
    task = ControlMessageTask(type="Test Task", id="task1", properties={"param": "value"})
    cm.add_task(task)
    assert cm.has_task("task1")
    cm.remove_task("task1")
    assert not cm.has_task("task1")
    tasks = list(cm.get_tasks())
    assert all(t.id != "task1" for t in tasks)


@pytest.mark.xfail
def test_remove_nonexistent_task():
    cm = IngestControlMessage()
    task = ControlMessageTask(type="Test Task", id="task1", properties={"param": "value"})
    cm.add_task(task)
    cm.remove_task("nonexistent")
    assert cm.has_task("task1")
    tasks = list(cm.get_tasks())
    assert any(t.id == "task1" for t in tasks)


def test_payload_get_default():
    cm = IngestControlMessage()
    payload = cm.payload()
    assert isinstance(payload, pd.DataFrame)
    assert payload.empty


def test_payload_set_valid():
    cm = IngestControlMessage()
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    returned_payload = cm.payload(df)
    pd.testing.assert_frame_equal(returned_payload, df)
    pd.testing.assert_frame_equal(cm.payload(), df)


def test_payload_set_invalid():
    cm = IngestControlMessage()
    with pytest.raises(ValueError):
        cm.payload("not a dataframe")


def test_config_get_default():
    cm = IngestControlMessage()
    default_config = cm.config()
    assert isinstance(default_config, dict)
    assert default_config == {}


def test_config_update_valid():
    cm = IngestControlMessage()
    new_config = {"setting": True, "threshold": 10}
    updated_config = cm.config(new_config)
    assert updated_config == new_config
    additional_config = {"another_setting": "value"}
    updated_config = cm.config(additional_config)
    assert updated_config == {"setting": True, "threshold": 10, "another_setting": "value"}


def test_config_update_invalid():
    cm = IngestControlMessage()
    with pytest.raises(ValueError):
        cm.config("not a dict")


def test_copy_creates_deep_copy():
    cm = IngestControlMessage()
    task = ControlMessageTask(type="Test Task", id="task1", properties={"param": "value"})
    cm.add_task(task)
    cm.set_metadata("meta", "data")
    dt = datetime(2025, 1, 1, 12, 0, 0)
    cm.set_timestamp("start", dt)
    df = pd.DataFrame({"col": [1, 2]})
    cm.payload(df)
    cm.config({"config_key": "config_value"})

    copy_cm = cm.copy()
    assert copy_cm is not cm
    assert list(copy_cm.get_tasks()) == list(cm.get_tasks())
    assert copy_cm.get_metadata() == cm.get_metadata()
    assert copy_cm.get_timestamps() == cm.get_timestamps()
    pd.testing.assert_frame_equal(copy_cm.payload(), cm.payload())
    assert copy_cm.config() == cm.config()

    copy_cm.remove_task("task1")
    copy_cm.set_metadata("meta", "new_data")
    copy_cm.set_timestamp("start", "2025-01-02T12:00:00")
    copy_cm.payload(pd.DataFrame({"col": [3, 4]}))
    copy_cm.config({"config_key": "new_config"})

    assert cm.has_task("task1")
    assert cm.get_metadata("meta") == "data"
    assert cm.get_timestamp("start") == dt
    pd.testing.assert_frame_equal(cm.payload(), df)
    assert cm.config()["config_key"] == "config_value"


@pytest.mark.xfail
def test_remove_nonexistent_task_logs_warning(caplog):
    cm = IngestControlMessage()
    with caplog.at_level("WARNING"):
        cm.remove_task("nonexistent")
        assert "Attempted to remove non-existent task" in caplog.text
