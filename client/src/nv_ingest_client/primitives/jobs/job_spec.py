# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

from nv_ingest_client.primitives.tasks import Task
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class JobSpec(BaseModel):
    """
    Specification for creating a job for submission to the nv-ingest microservice.

    Attributes
    ----------
    document_type: str
        Type of document that is being submitted.
    extended_options : Dict
        Storage for the additional options.
    job_id : UUID
        Storage for the job's unique identifier.
    payload : str
         Storage for the payload data.
    source_id : str
         Storage for the source identifier.
    source_name: str
        Storage for the source name.
    tasks : List
        Storage for the list of tasks.

    Methods
    -------
    add_task(task):
        Adds a task to the job specification.
    """

    # _document_type: Optional[str] = Field(default=None, alias="document_type")
    # _extended_options: Optional[Dict] = Field(default={}, alias="extended_options")
    # _job_id: Optional[typing.Union[UUID, str]] = Field(default=None, alias="job_id")
    # _payload: str = Field(default=None, alias="payload")
    # _source_id: Optional[str] = Field(default=None, alias="source_id")
    # _source_name: Optional[str] = Field(default=None, alias="source_name")
    # _tasks: Optional[List] = Field(default=[], alias="tasks")

    document_type: Optional[str]
    extended_options: Optional[Dict] = {}
    job_id: Optional[typing.Union[UUID, str]]
    payload: Optional[str]
    source_id: Optional[str]
    source_name: Optional[str]
    tasks: Optional[List] = []

    def __str__(self) -> str:
        task_info = "\n".join(str(task) for task in self.tasks)
        return (
            f"job-id: {self.job_id}\n"
            f"source-id: {self.source_id}\n"
            f"source-name: {self.source_name}\n"
            f"document-type: {self.document_type}\n"
            f"task count: {len(self.tasks)}\n"
            f"payload: {'<*** ' + str(len(self.payload)) + ' ***>' if self.payload else 'Empty'}\n"
            f"extended-options: {self.extended_options}\n"
            f"{task_info}"
        )

    # @property
    # def tasks(self) -> Any:
    #     """Gets the job specification associated with the state."""
    #     return self._tasks
    
    # @tasks.setter
    # def job_spec(self, value: JobSpec) -> None:
    #     """Sets the job specification associated with the state."""
    #     if self._state not in _PREFLIGHT_STATES:
    #         err_msg = f"Attempt to change job_spec after job submission: {self._state.name}"
    #         logger.error(err_msg)

    # @property
    # def job_id(self) -> UUID:
    #     return self._job_id

    # @job_id.setter
    # def job_id(self, job_id: UUID) -> None:
    #     self._job_id = job_id

    # @property
    # def source_id(self) -> str:
    #     return self._source_id

    # @source_id.setter
    # def source_id(self, source_id: str) -> None:
    #     self._source_id = source_id

    # @property
    # def source_name(self) -> str:
    #     return self._source_name

    # @source_name.setter
    # def source_name(self, source_name: str) -> None:
    #     self._source_name = source_name

    def add_task(self, task) -> None:
        """
        Adds a task to the job specification.

        Parameters
        ----------
        task
            The task to add to the job specification. Assumes the task has a to_dict method.

        Raises
        ------
        ValueError
            If the task does not have a to_dict method.
        """
        if not isinstance(task, Task):
            raise ValueError("Task must derive from nv_ingest_client.primitives.Task class")

        self.tasks.append(task)
