# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import UUID

from nv_ingest_client.primitives.tasks import Task
from nv_ingest_client.util.dataset import get_dataset_files
from nv_ingest_client.util.dataset import get_dataset_statistics

logger = logging.getLogger(__name__)


class JobSpec:
    """
    Specification for creating a job for submission to the nv-ingest microservice.

    Parameters
    ----------
    payload : Dict
        The payload data for the job.
    tasks : Optional[List], optional
        A list of tasks to be added to the job, by default None.
    source_id : Optional[str], optional
        An identifier for the source of the job, by default None.
    job_id : Optional[UUID], optional
        A unique identifier for the job, by default a new UUID is generated.
    extended_options : Optional[Dict], optional
        Additional options for job processing, by default None.

    Attributes
    ----------
    _payload : Dict
        Storage for the payload data.
    _tasks : List
        Storage for the list of tasks.
    _source_id : str
        Storage for the source identifier.
    _job_id : UUID
        Storage for the job's unique identifier.
    _extended_options : Dict
        Storage for the additional options.

    Methods
    -------
    to_dict() -> Dict:
        Converts the job specification to a dictionary.
    add_task(task):
        Adds a task to the job specification.
    """

    def __init__(
        self,
        payload: str = None,
        tasks: Optional[List] = None,
        source_id: Optional[str] = None,
        source_name: Optional[str] = None,
        document_type: Optional[str] = None,
        extended_options: Optional[Dict] = None,
    ) -> None:
        self._document_type = document_type or "txt"
        self._extended_options = extended_options or {}
        self._job_id = None
        self._payload = payload
        self._source_id = source_id
        self._source_name = source_name
        self._tasks = tasks or []

    def __str__(self) -> str:
        task_info = "\n".join(str(task) for task in self._tasks)
        return (
            f"source-id: {self._source_id}\n"
            f"source-name: {self._source_name}\n"
            f"document-type: {self._document_type}\n"
            f"task count: {len(self._tasks)}\n"
            f"payload: {'<*** ' + str(len(self._payload)) + ' ***>' if self._payload else 'Empty'}\n"
            f"extended-options: {self._extended_options}\n"
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
                "source_name": [self._source_name],
                "source_id": [self._source_id],
                "content": [self._payload],
                "document_type": [self._document_type],
            },
            "job_id": str(self._job_id),
            "tasks": [task.to_dict() for task in self._tasks],
            "tracing_options": self._extended_options.get("tracing_options", {}),
        }

    @property
    def payload(self) -> Dict:
        return self._payload

    @payload.setter
    def payload(self, payload: Dict) -> None:
        self._payload = payload

    @property
    def job_id(self) -> UUID:
        return self._job_id

    @job_id.setter
    def job_id(self, job_id: UUID) -> None:
        self._job_id = job_id

    @property
    def source_id(self) -> str:
        return self._source_id

    @source_id.setter
    def source_id(self, source_id: str) -> None:
        self._source_id = source_id

    @property
    def source_name(self) -> str:
        return self._source_name

    @source_name.setter
    def source_name(self, source_name: str) -> None:
        self._source_name = source_name

    @property
    def document_type(self) -> str:
        return self._document_type

    @source_name.setter
    def document_type(self, document_type: str) -> None:
        self._document_type = document_type

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

        self._tasks.append(task)


class BatchJobSpec:
    """
    A class representing a batch of job specifications, allowing for batch processing of multiple jobs.

    Parameters
    ----------
    job_specs : Optional[Union[List[JobSpec], List[str]]], optional
        Either a list of JobSpec objects or a list of file paths. If file paths are provided, JobSpec
        instances will be created from the file paths, by default None.

    Attributes
    ----------
    _job_specs : List[JobSpec]
        A list of JobSpec objects that are part of this batch.

    Methods
    -------
    from_files(files: List[str]):
        Generates a list of JobSpec instances from the given list of file paths.
    add_job_spec(job_spec: JobSpec):
        Adds a single JobSpec to the batch.
    to_dict() -> List[Dict]:
        Converts all JobSpec objects in the batch to a list of dictionaries for serialization.
    """

    def __init__(self, job_specs_or_files: Optional[Union[List[JobSpec], List[str]]] = None) -> None:
        """
        Initialize the BatchJobSpec with either a list of JobSpec objects or a list of file paths.

        Parameters
        ----------
        job_specs : Optional[Union[List[JobSpec], List[str]]], optional
            Either a list of JobSpec objects or a list of file paths, by default None.
        """
        self._file_type_to_job_spec = defaultdict(list)

        if job_specs_or_files:
            if isinstance(job_specs_or_files[0], JobSpec):
                self.from_job_specs(job_specs_or_files)
            elif isinstance(job_specs_or_files[0], str):
                self.from_files(job_specs_or_files)
            else:
                raise ValueError("Invalid input type for job_specs. Must be a list of JobSpec or file paths.")

    def from_job_specs(self, job_specs: Union[JobSpec, List[JobSpec]]) -> None:
        if isinstance(job_specs, JobSpec):
            job_specs = [JobSpec]

        for job_spec in job_specs:
            self.add_job_spec(job_spec)

    def from_files(self, files: Union[str, List[str]]) -> None:
        """
        Generates JobSpec instances from a list of file paths.

        Parameters
        ----------
        files : List[str]
            A list of file paths to generate JobSpec instances from.
        """
        from nv_ingest_client.client.util import create_job_specs_for_batch
        from nv_ingest_client.util.util import generate_matching_files

        if isinstance(files, str):
            files = [files]

        matching_files = list(generate_matching_files(files))
        if not matching_files:
            logger.warning(f"No files found matching {files}.")
            return

        job_specs = create_job_specs_for_batch(matching_files)
        for job_spec in job_specs:
            self.add_job_spec(job_spec)

    def _from_dataset(self, dataset: str, shuffle_dataset: bool = True) -> None:
        with open(dataset, "rb") as file:
            dataset_bytes = BytesIO(file.read())

        logger.debug(get_dataset_statistics(dataset_bytes))
        dataset_files = get_dataset_files(dataset_bytes, shuffle_dataset)

        self.from_files(dataset_files)

    @classmethod
    def from_dataset(cls, dataset: str, shuffle_dataset: bool = True):
        batch_job_spec = cls()
        batch_job_spec._from_dataset(dataset, shuffle_dataset=shuffle_dataset)
        return batch_job_spec

    def add_job_spec(self, job_spec: JobSpec) -> None:
        """
        Adds a single JobSpec to the batch.

        Parameters
        ----------
        job_spec : JobSpec
            The JobSpec instance to add to the batch.
        """
        self._file_type_to_job_spec[job_spec.document_type].append(job_spec)

    def add_task(self, task):
        """
        Adds a task to the job specification.

        Parameters
        ----------
        file_type

        task
            The task to add to the job specification.

        Raises
        ------
        ValueError
            If the task does not have a to_dict method.
        """
        if not isinstance(task, Task):
            raise ValueError("Task must derive from nv_ingest_client.primitives.Task class")

        for job_spec in self._file_type_to_job_spec[task.document_type]:
            job_spec.add_task(task)

    def to_dict(self) -> List[Dict]:
        """
        Converts the batch of JobSpec instances into a list of dictionaries for serialization.

        Returns
        -------
        List[Dict]
            A list of dictionaries representing the JobSpec objects in the batch.
        """
        return [job_spec.to_dict() for job_spec in self._job_specs]

    def __str__(self) -> str:
        """
        Provides a string representation of the BatchJobSpec, listing all JobSpec instances.

        Returns
        -------
        str
            A string representation of the batch.
        """
        return "\n".join(str(job_spec) for job_spec in self._job_specs)

    @property
    def job_specs(self) -> Dict[str, List[str]]:
        return self._file_type_to_job_spec
