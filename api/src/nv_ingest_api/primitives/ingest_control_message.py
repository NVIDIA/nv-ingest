import copy
import re
from datetime import datetime

import logging
import pandas as pd
from typing import Any, Dict, Generator, Union

from nv_ingest_api.primitives.control_message_task import ControlMessageTask


logger = logging.getLogger(__name__)


def remove_task_by_type(ctrl_msg, task: str):
    """
    Remove a task from the control message by matching its type.

    This function iterates over the tasks in the control message, and if it finds a task
    whose type matches the provided task string, it removes that task (using its unique id)
    and returns the task's properties.

    Parameters
    ----------
    ctrl_msg : IngestControlMessage
        The control message from which to remove the task.
    task : str
        The task type to remove.

    Returns
    -------
    dict
        The properties of the removed task.

    Raises
    ------
    ValueError
        If no task with the given type is found.
    """
    task_obj = None
    for t in ctrl_msg.get_tasks():
        if t.type == task:
            task_obj = t
            break

    if task_obj is None:
        err_msg = f"process_control_message: Task '{task}' not found in control message."
        logger.error(err_msg)
        raise ValueError(err_msg)

    removed_task = ctrl_msg.remove_task(task_obj.id)
    return removed_task.properties


class IngestControlMessage:
    """
    A control message class for ingesting tasks and managing associated metadata,
    timestamps, configuration, and payload.
    """

    def __init__(self):
        """
        Initialize a new IngestControlMessage instance.
        """
        self._tasks: Dict[str, ControlMessageTask] = {}
        self._metadata: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._payload: pd.DataFrame = pd.DataFrame()
        self._config: Dict[str, Any] = {}

    def add_task(self, task: ControlMessageTask):
        """
        Add a task to the control message, keyed by the task's unique 'id'.

        Raises
        ------
        ValueError
            If a task with the same 'id' already exists.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task with id '{task.id}' already exists. Tasks must be unique.")
        self._tasks[task.id] = task

    def get_tasks(self) -> Generator[ControlMessageTask, None, None]:
        """
        Return all tasks as a generator.
        """
        yield from self._tasks.values()

    def has_task(self, task_id: str) -> bool:
        """
        Check if a task with the given ID exists.
        """
        return task_id in self._tasks

    def remove_task(self, task_id: str) -> ControlMessageTask:
        """
        Remove a task from the control message. Logs a warning if the task does not exist.
        """
        if task_id in self._tasks:
            _task = self._tasks[task_id]

            del self._tasks[task_id]

            return _task
        else:
            raise RuntimeError(f"Attempted to remove non-existent task with id: {task_id}")

    def config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get or update the control message configuration.

        If 'config' is provided, it must be a dictionary. The configuration is updated with the
        provided values. If no argument is provided, returns a copy of the current configuration.

        Raises
        ------
        ValueError
            If the provided configuration is not a dictionary.
        """
        if config is None:
            return self._config.copy()

        if not isinstance(config, dict):
            raise ValueError("Configuration must be provided as a dictionary.")

        self._config.update(config)
        return self._config.copy()

    def copy(self) -> "IngestControlMessage":
        """
        Create a deep copy of this control message.
        """
        return copy.deepcopy(self)

    def get_metadata(self, key: Union[str, re.Pattern] = None, default_value: Any = None) -> Any:
        """
        Retrieve metadata. If 'key' is None, returns a copy of all metadata.

        Parameters
        ----------
        key : str or re.Pattern, optional
            If a string is provided, returns the value for that exact key.
            If a regex pattern is provided, returns a dictionary of all metadata key-value pairs
            where the key matches the regex. If no matches are found, returns default_value.
        default_value : Any, optional
            The value to return if the key is not found or no regex matches.

        Returns
        -------
        Any
            The metadata value for an exact string key, or a dict of matching metadata if a regex is provided.
        """
        if key is None:
            return self._metadata.copy()

        # If key is a regex pattern (i.e. has a search method), perform pattern matching.
        if hasattr(key, "search"):
            matches = {k: v for k, v in self._metadata.items() if key.search(k)}
            return matches if matches else default_value

        # Otherwise, perform an exact lookup.
        return self._metadata.get(key, default_value)

    def has_metadata(self, key: Union[str, re.Pattern]) -> bool:
        """
        Check if a metadata key exists.

        Parameters
        ----------
        key : str or re.Pattern
            If a string is provided, checks for the exact key.
            If a regex pattern is provided, returns True if any metadata key matches the regex.

        Returns
        -------
        bool
            True if the key (or any matching key, in case of a regex) exists, False otherwise.
        """
        if hasattr(key, "search"):
            return any(key.search(k) for k in self._metadata)
        return key in self._metadata

    def list_metadata(self) -> list:
        """
        List all metadata keys.
        """
        return list(self._metadata.keys())

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata key-value pair.
        """
        self._metadata[key] = value

    def filter_timestamp(self, regex_filter: str) -> Dict[str, datetime]:
        """
        Retrieve timestamps whose keys match the regex filter.
        """
        pattern = re.compile(regex_filter)
        timestamps_snapshot = self._timestamps.copy()
        return {key: ts for key, ts in timestamps_snapshot.items() if pattern.search(key)}

    def get_timestamp(self, key: str, fail_if_nonexist: bool = False) -> datetime:
        """
        Retrieve a timestamp for a given key.

        Raises
        ------
        KeyError
            If the key is not found and 'fail_if_nonexist' is True.
        """
        if key in self._timestamps:
            return self._timestamps[key]
        if fail_if_nonexist:
            raise KeyError(f"Timestamp for key '{key}' does not exist.")
        return None

    def get_timestamps(self) -> Dict[str, datetime]:
        """
        Retrieve all timestamps.
        """
        return self._timestamps.copy()

    def set_timestamp(self, key: str, timestamp: Any) -> None:
        """
        Set a timestamp for a given key. Accepts either a datetime object or an ISO format string.

        Raises
        ------
        ValueError
            If the provided timestamp is neither a datetime object nor a valid ISO format string.
        """
        if isinstance(timestamp, datetime):
            self._timestamps[key] = timestamp

        elif isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
                self._timestamps[key] = dt
            except ValueError as e:
                raise ValueError(f"Invalid timestamp format: {timestamp}") from e

        else:
            raise ValueError("timestamp must be a datetime object or ISO format string")

    def payload(self, payload: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get or set the payload DataFrame.

        Raises
        ------
        ValueError
            If the provided payload is not a pandas DataFrame.
        """
        if payload is None:
            return self._payload

        if not isinstance(payload, pd.DataFrame):
            raise ValueError("Payload must be a pandas DataFrame")

        self._payload = payload
        return self._payload
