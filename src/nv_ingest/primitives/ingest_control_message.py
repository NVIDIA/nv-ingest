import re
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Generator


class ControlMessageTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    id: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class IngestControlMessage:
    """
    A control message class for ingesting tasks and managing associated metadata, timestamps, and payload.
    """

    def __init__(self):
        """
        Initialize a new IngestControlMessage instance.

        Parameters
        ----------
        None
        """
        self._tasks: Dict[str, ControlMessageTask] = {}
        self._metadata: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._payload: pd.DataFrame = pd.DataFrame()

    def add_task(self, task: ControlMessageTask):
        """
        Add a task to the control message, keyed by the task's unique 'id'.

        Parameters
        ----------
        task : ControlMessageTask
            A validated Pydantic model containing task information.

        Raises
        ------
        ValueError
            If a task with the same 'id' already exists in this control message.
        """
        if task.id in self._tasks:
            raise ValueError(
                f"Task with id '{task.id}' already exists in this control message. "
                "Tasks must be unique per-control message."
            )
        self._tasks[task.id] = task

    def get_tasks(self) -> Generator[ControlMessageTask, None, None]:
        """
        Return all tasks as a generator.

        Yields
        ------
        ControlMessageTask
            Each task in the control message.
        """
        for task in self._tasks.values():
            yield task

    def has_task(self, task_id: str) -> bool:
        """
        Check if a task with the given ID exists in this control message.

        Parameters
        ----------
        task_id : str
            The ID of the task to check.

        Returns
        -------
        bool
            True if the task exists, False otherwise.
        """
        return task_id in self._tasks

    def remove_task(self, task_id: str) -> None:
        """
        Remove a task from the control message.

        Parameters
        ----------
        task_id : str
            The unique identifier of the task to remove.

        Returns
        -------
        None
        """
        if task_id in self._tasks:
            del self._tasks[task_id]

    def config(self, config=None):
        """
        Configure the control message or retrieve its current configuration.

        Parameters
        ----------
        config : dict, optional
            A dictionary-like object with configuration data.

        Returns
        -------
        dict
            If called without arguments, returns the current configuration.
        """
        pass

    def copy(self):
        """
        Create a copy of this control message.

        Returns
        -------
        IngestControlMessage
            A new copy of the current control message.
        """
        pass

    def get_metadata(self, key: str = None, default_value: Any = None) -> Any:
        """
        Retrieve metadata stored in the control message.

        Parameters
        ----------
        key : str, optional
            The metadata key to retrieve. If None, returns a copy of all metadata.
        default_value : Any, optional
            The value to return if the key is not found.

        Returns
        -------
        Any
            The metadata value corresponding to the key, or a copy of all metadata if key is None.
        """
        if key is None:
            return self._metadata.copy()
        return self._metadata.get(key, default_value)

    def has_metadata(self, key: str) -> bool:
        """
        Check if a metadata key exists in the control message.

        Parameters
        ----------
        key : str
            The metadata key to check.

        Returns
        -------
        bool
            True if the key exists, False otherwise.
        """
        return key in self._metadata

    def list_metadata(self) -> list:
        """
        List all metadata keys in the control message.

        Returns
        -------
        list of str
            A list containing all metadata keys.
        """
        return list(self._metadata.keys())

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata key-value pair in the control message.

        Parameters
        ----------
        key : str
            The metadata key.
        value : Any
            The metadata value.
        """
        self._metadata[key] = value

    def filter_timestamp(self, regex_filter: str) -> Dict[str, datetime]:
        """
        Retrieve timestamps matching a regex filter.

        Parameters
        ----------
        regex_filter : str
            The regex pattern to match against timestamp keys.

        Returns
        -------
        dict
            A dictionary of matching timestamp entries.
        """
        pattern = re.compile(regex_filter)
        return {key: ts for key, ts in self._timestamps.items() if pattern.search(key)}

    def get_timestamp(self, key: str, fail_if_nonexist: bool = False) -> datetime:
        """
        Retrieve a timestamp for a given key.

        Parameters
        ----------
        key : str
            The key associated with a timestamp.
        fail_if_nonexist : bool, optional
            If True, raise an error if the timestamp doesn't exist. Otherwise, return None.

        Returns
        -------
        datetime or None
            The timestamp if found; otherwise None (or raises an error if fail_if_nonexist is True).

        Raises
        ------
        KeyError
            If the key does not exist and fail_if_nonexist is True.
        """
        if key in self._timestamps:
            return self._timestamps[key]
        if fail_if_nonexist:
            raise KeyError(f"Timestamp for key '{key}' does not exist.")
        return None

    def get_timestamps(self) -> Dict[str, datetime]:
        """
        Retrieve all timestamps.

        Returns
        -------
        dict
            A dictionary of all timestamps stored in this control message.
        """
        return self._timestamps.copy()

    def set_timestamp(self, key: str, timestamp: Any) -> None:
        """
        Set a timestamp for a given key.

        Parameters
        ----------
        key : str
            The key to associate with the timestamp.
        timestamp : datetime or str
            The timestamp to store. If a string is provided, it must be in ISO format.

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
        Get or set the payload for this control message.

        Parameters
        ----------
        payload : pd.DataFrame, optional
            A pandas DataFrame to set as the payload. If None, the current payload is returned.

        Returns
        -------
        pd.DataFrame
            The current payload if payload is None; otherwise, returns the newly set payload.

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
