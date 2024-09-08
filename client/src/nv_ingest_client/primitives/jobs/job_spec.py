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

from nv_ingest_client.primitives.tasks.caption import CaptionTask
from nv_ingest_client.primitives.tasks.dedup import DedupTask
from nv_ingest_client.primitives.tasks.embed import EmbedTask
from nv_ingest_client.primitives.tasks.filter import FilterTask
from nv_ingest_client.primitives.tasks.split import SplitTask
from nv_ingest_client.primitives.tasks.store import StoreTask
from nv_ingest_client.primitives.tasks.vdb_upload import VdbUploadTask
from nv_ingest_client.primitives.tasks.extract import ExtractTask
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

    document_type: Optional[str]
    extended_options: Optional[Dict] = {}
    job_id: Optional[typing.Union[UUID, str]]
    payload: Optional[str]
    source_id: Optional[str]
    source_name: Optional[str]
    tasks: Optional[List] = []

    class Config:
        # Allow population by field name as well as alias
        allow_population_by_field_name = True
        # Allow arbitrary types
        arbitrary_types_allowed = True

        json_encoders = {
            CaptionTask: lambda v: v.to_dict(),
            DedupTask: lambda v: v.to_dict(),
            EmbedTask: lambda v: v.to_dict(),
            ExtractTask: lambda v: v.to_dict(),
            FilterTask: lambda v: v.to_dict(),
            SplitTask: lambda v: v.to_dict(),
            StoreTask: lambda v: v.to_dict(),
            VdbUploadTask: lambda v: v.to_dict()
        }

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
    
    def to_dict(self) -> Dict:
        """
        Converts the job specification instance into a dictionary suitable for JSON serialization.

        Returns
        -------
        Dict
            A dictionary representation of the job specification.
        """
        return {
            "job_payload": {
                "source_name": [self.source_name],
                "source_id": [self.source_id],
                "content": [self.payload],
                "document_type": [self.document_type],
            },
            "job_id": str(self.job_id),
            "tasks": [task if isinstance(task, dict) else task.to_dict() for task in self.tasks],
            "tracing_options": self.extended_options.get("tracing_options", {}),
        }


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
