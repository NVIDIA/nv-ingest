# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask


def test_empty_control_message():
    """
    Validate that an IngestControlMessage with no tasks returns an empty list from get_tasks()
    and that has_task returns False for any task id.
    """
    cm = IngestControlMessage()
    assert list(cm.get_tasks()) == []
    assert not cm.has_task("nonexistent")


def test_add_single_task():
    """
    Validate that adding a single ControlMessageTask stores the task correctly, making it retrievable
    via has_task and get_tasks.
    """
    cm = IngestControlMessage()
    task = ControlMessageTask(type="Test Task", id="task1", properties={"key": "value"})
    cm.add_task(task)
    assert cm.has_task("task1")
    tasks = list(cm.get_tasks())
    assert len(tasks) == 1
    assert tasks[0] == task


def test_add_duplicate_task():
    """
    Validate that adding a duplicate task (same id) raises a ValueError indicating that tasks must be unique.
    """
    cm = IngestControlMessage()
    task = ControlMessageTask(type="Test Task", id="task1", properties={"key": "value"})
    cm.add_task(task)
    duplicate_task = ControlMessageTask(type="Another Task", id="task1", properties={"key": "other"})
    cm.add_task(duplicate_task)


def test_multiple_tasks():
    """
    Validate that multiple tasks added to IngestControlMessage are stored and retrievable.
    Ensures that has_task returns True for all added tasks and that get_tasks returns the correct set of tasks.
    """
    cm = IngestControlMessage()
    task_data = [
        {"type": "Task A", "id": "a", "properties": {}},
        {"type": "Task B", "id": "b", "properties": {"x": 10}},
        {"type": "Task C", "id": "c", "properties": {"y": 20}},
    ]
    tasks = [ControlMessageTask(**data) for data in task_data]
    for task in tasks:
        cm.add_task(task)
    for data in task_data:
        assert cm.has_task(data["id"])
    retrieved_tasks = list(cm.get_tasks())
    assert len(retrieved_tasks) == len(task_data)
    retrieved_ids = {t.id for t in retrieved_tasks}
    expected_ids = {data["id"] for data in task_data}
    assert retrieved_ids == expected_ids
