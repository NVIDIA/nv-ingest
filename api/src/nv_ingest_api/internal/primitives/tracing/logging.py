# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import inspect
import uuid
from datetime import datetime
from enum import Enum

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


class TaskResultStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


def annotate_cm(control_message: IngestControlMessage, source_id=None, **kwargs):
    """
    Annotate a IngestControlMessage object with arbitrary metadata, a source ID, and a timestamp.
    Each annotation will be uniquely identified by a UUID.

    Parameters:
    - control_message: The IngestControlMessage object to be annotated.
    - source_id: A unique identifier for the source of the annotation. If None, uses the caller's __name__.
    - **kwargs: Arbitrary key-value pairs to be included in the annotation.
    """
    if source_id is None:
        # Determine the __name__ of the parent caller's module
        frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(frame)[2]
        module = inspect.getmodule(caller_frame[0])
        source_id = module.__name__ if module is not None else "UnknownModule"

    # Ensure 'annotation_timestamp' is not overridden by kwargs
    if "annotation_timestamp" in kwargs:
        raise ValueError("'annotation_timestamp' is a reserved key and cannot be specified.")

    message = kwargs.get("message")
    annotation_key = f"annotation::{message}" if message else f"annotation::{uuid.uuid4()}"

    annotation_timestamp = datetime.now()
    try:
        control_message.set_timestamp(annotation_key, annotation_timestamp)
    except Exception as e:
        print(f"Failed to set annotation timestamp: {e}")

    # Construct the metadata key uniquely identified by a UUID.
    metadata_key = f"annotation::{uuid.uuid4()}"

    # Construct the metadata value with reserved 'annotation_timestamp', source_id, and any provided kwargs.
    metadata_value = {
        "source_id": source_id,
    }
    metadata_value.update(kwargs)

    try:
        # Attempt to set the annotated metadata on the IngestControlMessage object.
        control_message.set_metadata(metadata_key, metadata_value)
    except Exception as e:
        # Handle any exceptions that occur when setting metadata.
        print(f"Failed to annotate IngestControlMessage: {e}")


def annotate_task_result(control_message, result, task_id, source_id=None, **kwargs):
    """
    Annotate a IngestControlMessage object with the result of a task, identified by a task_id,
    and an arbitrary number of additional key-value pairs. The result can be a TaskResultStatus
    enum or a string that will be converted to the corresponding enum.

    Parameters:
    - control_message: The IngestControlMessage object to be annotated.
    - result: The result of the task, either SUCCESS or FAILURE, as an enum or string.
    - task_id: A unique identifier for the task.
    - **kwargs: Arbitrary additional key-value pairs to be included in the annotation.
    """
    # Convert result to TaskResultStatus enum if it's a string
    if isinstance(result, str):
        try:
            result = TaskResultStatus[result.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid result string: {result}. Must be one of {[status.name for status in TaskResultStatus]}."
            )
    elif not isinstance(result, TaskResultStatus):
        raise ValueError("result must be an instance of TaskResultStatus Enum or a valid result string.")

    # Annotate the control message with task-related information, including the result and task_id.
    annotate_cm(
        control_message,
        source_id=source_id,
        task_result=result.value,
        task_id=task_id,
        **kwargs,
    )
