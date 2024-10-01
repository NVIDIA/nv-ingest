# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=broad-except

import json
import logging
import concurrent.futures
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from nv_ingest_client.message_clients.rest.rest_client import RestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.jobs import JobState
from nv_ingest_client.primitives.jobs import JobStateEnum
from nv_ingest_client.primitives.tasks import Task
from nv_ingest_client.primitives.tasks import TaskType
from nv_ingest_client.primitives.tasks import task_factory

logger = logging.getLogger(__name__)


class DataDecodeException(Exception):
    """
    Exception raised for errors in decoding data.

    Attributes:
        message -- explanation of the error
        data -- the data that failed to decode, optionally
    """

    def __init__(self, message="Data decoding error", data=None):
        self.message = message
        self.data = data
        super().__init__(f"{message}: {data}")

    def __str__(self):
        return f"{self.__class__.__name__}({self.message}, Data={self.data})"


class NvIngestClient:
    """
    A client class for interacting with the nv-ingest service, supporting custom client allocators.
    """

    def __init__(
            self,
            message_client_allocator: Callable[..., RestClient] = RestClient,
            message_client_hostname: Optional[str] = "localhost",
            message_client_port: Optional[int] = 7670,
            message_client_kwargs: Optional[Dict] = None,
            msg_counter_id: Optional[str] = "nv-ingest-message-id",
            worker_pool_size: int = 1,
    ) -> None:
        """
        Initializes the NvIngestClient with a client allocator, REST configuration, a message counter ID,
        and a worker pool for parallel processing.

        Parameters
        ----------
        message_client_allocator : Callable[..., RestClient]
            A callable that when called returns an instance of the client used for communication.
        message_client_hostname : str, optional
            The hostname of the REST server. Defaults to "localhost".
        message_client_port : int, optional
            The port number of the REST server. Defaults to 7670.
        msg_counter_id : str, optional
            The key for tracking message counts. Defaults to "nv-ingest-message-id".
        worker_pool_size : int, optional
            The number of worker processes in the pool. Defaults to 1.
        """

        self._current_message_id = 0
        self._job_states = {}
        self._message_client_hostname = message_client_hostname or "localhost"
        self._message_client_port = message_client_port or 7670
        self._message_counter_id = msg_counter_id or "nv-ingest-message-id"

        logger.debug("Instantiate NvIngestClient:\n%s", str(self))
        self._message_client = message_client_allocator(
            host=self._message_client_hostname, port=self._message_client_port
        )

        # Initialize the worker pool with the specified size
        self._worker_pool = ThreadPoolExecutor(max_workers=worker_pool_size)

        self._telemetry = {}

    def __str__(self) -> str:
        """
        Returns a string representation of the NvIngestClient configuration and runtime state.

        Returns
        -------
        str
            A string representation of the client showing the Redis configuration.
        """
        info = "NvIngestClient:\n"
        info += f" message_client_host: {self._message_client_hostname}\n"
        info += f" message_client_port: {self._message_client_port}\n"

        return info

    def _generate_job_index(self) -> str:
        """
        Generates a unique job ID by combining a UUID with an incremented value from Redis.

        Returns
        -------
        str
            A unique job ID in the format of "<UUID>_<Redis incremented value>".  IF the client
            is a RedisClient. In the case of a RestClient it is simply the UUID.
        """

        job_index = str(self._current_message_id)
        self._current_message_id += 1

        return job_index

    def _pop_job_state(self, job_index: str) -> JobState:
        """
        Deletes the job with the specified ID from the job tracking dictionary.

        Parameters
        ----------
        job_index : str
            The ID of the job to delete.
        """

        job_state = self._get_and_check_job_state(job_index)
        self._job_states.pop(job_index)

        return job_state

    def _get_and_check_job_state(
            self,
            job_index: str,
            required_state: Union[JobStateEnum, List[JobStateEnum]] = None,
    ) -> JobState:
        if required_state and not isinstance(required_state, list):
            required_state = [required_state]

        job_state = self._job_states.get(job_index)

        if not job_state:
            raise ValueError(f"Job with ID {job_index} does not exist in JobStates: {self._job_states}")
        if required_state and (job_state.state not in required_state):
            raise ValueError(
                f"Job with ID {job_state.job_spec.job_id} has invalid state {job_state.state}, expected {required_state}"
            )

        return job_state

    def job_count(self):
        return len(self._job_states)

    def add_job(self, job_spec: JobSpec = None) -> str:
        job_index = self._generate_job_index()

        self._job_states[job_index] = JobState(job_spec=job_spec)

        return job_index

    def create_job(
            self,
            payload: str,
            source_id: str,
            source_name: str,
            document_type: str = None,
            tasks: Optional[list] = None,
            extended_options: Optional[dict] = None,
    ) -> str:
        """
        Creates a new job with the specified parameters and adds it to the job tracking dictionary.

        Parameters
        ----------
        job_id : uuid.UUID, optional
            The unique identifier for the job. If not provided, a new UUID will be generated.
        payload : dict
            The payload associated with the job. Defaults to an empty dictionary if not provided.
        tasks : list, optional
            A list of tasks to be associated with the job.
        document_type : str
            The type of document to be processed.
        source_id : str
            The source identifier for the job.
        source_name : str
            The unique name of the job's source data.
        extended_options : dict, optional
            Additional options for job creation.

        Returns
        -------
        str
            The job ID as a string.

        Raises
        ------
        ValueError
            If a job with the specified `job_id` already exists.
        """

        document_type = document_type or source_name.split(".")[-1]
        job_spec = JobSpec(
            payload=payload or {},
            tasks=tasks,
            document_type=document_type,
            source_id=source_id,
            source_name=source_name,
            extended_options=extended_options,
        )

        return self.add_job(job_spec)

    def add_task(self, job_index: str, task: Task) -> None:
        job_state = self._get_and_check_job_state(job_index, required_state=JobStateEnum.PENDING)

        job_state.job_spec.add_task(task)

    def create_task(
            self,
            job_index: Union[str, int],
            task_type: TaskType,
            task_params: dict = None,
    ) -> None:
        """
        Creates a task of the specified type with given parameters and associates it with the existing job.

        Parameters
        ----------
        job_index: Union[str, int]
            The unique identifier of the job to which the task will be added. This can be either a string or an integer.
        task_type : TaskType
            The type of the task to be created, defined as an enum value.
        task_params : dict
            A dictionary containing parameters for the task.

        Raises
        ------
        ValueError
            If the job with the specified `job_id` does not exist or if an attempt is made to modify a job after its
            submission.
        """
        task_params = task_params or {}

        return self.add_task(job_index, task_factory(task_type, **task_params))

    def _fetch_job_result(self, job_index: str, timeout: float = 100, data_only: bool = True) -> Tuple[Dict, str]:
        """
        Fetches the job result from a message client, handling potential errors and state changes.

        Args:
            job_index (str): The identifier of the job.
            timeout (float): Timeout for the fetch operation in seconds.
            data_only (bool): If True, only returns the data part of the job result.

        Returns:
            Tuple[Dict, str]: The job result and the job ID.

        Raises:
            ValueError: If there is an error in decoding the job result.
            TimeoutError: If the fetch operation times out.
            Exception: For all other unexpected issues.
        """

        try:
            job_state = self._get_and_check_job_state(job_index, required_state=[JobStateEnum.SUBMITTED])
            response = self._message_client.fetch_message(job_state.job_id, timeout)

            if response is not None:
                try:
                    job_state.state = JobStateEnum.PROCESSING
                    response_json = json.loads(response)
                    if data_only:
                        response_json = response_json["data"]

                    return response_json, job_index
                except json.JSONDecodeError as err:
                    logger.error(f"Error decoding job result for job ID {job_index}: {err}")
                    raise ValueError(f"Error decoding job result: {err}") from err
                finally:
                    # Only pop once we know we've successfully decoded the response or errored out
                    _ = self._pop_job_state(job_index)
            else:
                raise TimeoutError(f"Timeout: No response within {timeout} seconds for job ID {job_index}")

        except TimeoutError:
            raise
        except RuntimeError as err:
            logger.error(f"Unexpected error while fetching job result for job ID {job_index}: {err}")
            raise
        except Exception as err:
            logger.error(f"Unexpected error while fetching job result for job ID {job_index}: {err}")
            raise

    # The Pythonic invocation and the CLI invocation approach currently have different approaches to timeouts
    # This distinction is made obvious by provided two separate functions. One for "_cli" and one for
    # direct Python use. This is the "_cli" approach
    def fetch_job_result_cli(self, job_ids: Union[str, List[str]], timeout: float = 100, data_only: bool = True):
        if isinstance(job_ids, str):
            job_ids = [job_ids]

        return [self._fetch_job_result(job_id, timeout, data_only) for job_id in job_ids]
    
    
    # Nv-Ingest jobs are often "long running". Therefore after
    # submission we intermittently check if the job is completed.
    def _fetch_job_result_wait(self, job_id: str, timeout: float = 60, data_only: bool = True):
        while True:
            try:
                return [self._fetch_job_result(job_id, timeout, data_only)]
            except TimeoutError:
                logger.debug("Job still processing ... aka HTTP 202 received")
    
    # This is the direct Python approach function for retrieving jobs which handles the timeouts directly
    # in the function itself instead of expecting the user to handle it themselves
    # Note this method only supports fetching a single job result synchronously
    def fetch_job_result(self, job_id: str, timeout: float = 100, data_only: bool = True):        
        # A thread pool executor is a simple approach to performing an action with a timeout
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self._fetch_job_result_wait, job_id, timeout, data_only)
            try:
                # Wait for the result within the specified timeout
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                # Raise a standard Python TimeoutError which the client will be expecting
                raise TimeoutError(f"Job processing did not complete within the specified {timeout} seconds")
        

    def _ensure_submitted(self, job_ids: List[str]):
        if isinstance(job_ids, str):
            job_ids = [job_ids]  # Ensure job_ids is always a list

        submission_futures = {}
        for job_id in job_ids:
            job_state = self._get_and_check_job_state(
                job_id,
                required_state=[JobStateEnum.SUBMITTED, JobStateEnum.SUBMITTED_ASYNC],
            )
            if job_state.state == JobStateEnum.SUBMITTED_ASYNC:
                submission_futures[job_state.future] = job_state

        for future in as_completed(submission_futures.keys()):
            job_state = submission_futures[future]
            job_state.state = JobStateEnum.SUBMITTED
            job_state.future = None

    def fetch_job_result_async(
            self, job_ids: Union[str, List[str]], timeout: float = 10, data_only: bool = True
    ) -> Dict[Future, str]:
        """
        Fetches job results for a list or a single job ID asynchronously and returns a mapping of futures to job IDs.

        Parameters:
            job_ids (Union[str, List[str]]): A single job ID or a list of job IDs.
            timeout (float): Timeout for fetching each job result, in seconds.
            data_only (bool): Whether to return only the data part of the job result.

        Returns:
            Dict[Future, str]: A dictionary mapping each future to its corresponding job ID.
        """
        if isinstance(job_ids, str):
            job_ids = [job_ids]  # Ensure job_ids is always a list

        # Make sure all jobs have actually been submitted before launching fetches.
        self._ensure_submitted(job_ids)

        future_to_job_id = {}
        for job_id in job_ids:
            job_state = self._get_and_check_job_state(job_id)

            future = self._worker_pool.submit(self.fetch_job_result_cli, job_id, timeout, data_only)
            job_state.future = future
            future_to_job_id[future] = job_id

        return future_to_job_id

    def _submit_job(
            self,
            job_index: str,
            job_queue_id: str,
    ) -> Optional[Dict]:
        """
        Submits a job to a specified job queue and optionally waits for a response if blocking is True.

        Parameters
        ----------
        job_index : str
            The unique identifier of the job to be submitted.
        job_queue_id : str
            The ID of the job queue where the job will be submitted.

        Returns
        -------
        Optional[Dict]
            The job result if blocking is True and a result is available before the timeout, otherwise None.

        Raises
        ------
        Exception
            If submitting the job fails.
        """

        job_state = self._get_and_check_job_state(
            job_index, required_state=[JobStateEnum.PENDING, JobStateEnum.SUBMITTED_ASYNC]
        )

        try:
            message = json.dumps(job_state.job_spec.to_dict())

            x_trace_id, job_id = self._message_client.submit_message(job_queue_id, message)

            job_state.state = JobStateEnum.SUBMITTED
            job_state.job_id = job_id

            # Free up memory -- payload should never be used again, and we don't want to keep it around.
            job_state.job_spec.payload = None
            
            return x_trace_id
        except Exception as err:
            logger.error(f"Failed to submit job {job_index} to queue {job_queue_id}: {err}")
            job_state.state = JobStateEnum.FAILED
            raise

    def submit_job(self, job_indices: Union[str, List[str]], job_queue_id: str) -> List[Union[Dict, None]]:
        if isinstance(job_indices, str):
            job_indices = [job_indices]

        return [self._submit_job(job_id, job_queue_id) for job_id in job_indices]

    def submit_job_async(self, job_indices: Union[str, List[str]], job_queue_id: str) -> Dict[Future, str]:
        """
        Asynchronously submits one or more jobs to a specified job queue using a thread pool.
        This method handles both single job ID or a list of job IDs.

        Parameters
        ----------
        job_indices : Union[str, List[str]]
            A single job ID or a list of job IDs to be submitted.
        job_queue_id : str
            The ID of the job queue where the jobs will be submitted.

        Returns
        -------
        Dict[Future, str]
            A dictionary mapping futures to their respective job IDs for later retrieval of outcomes.

        Notes
        -----
        - This method queues the jobs for asynchronous submission and returns a mapping of futures to job IDs.
        - It does not wait for any of the jobs to complete.
        - Ensure that each job is in the proper state before submission.
        """

        if isinstance(job_indices, str):
            job_indices = [job_indices]  # Convert single job_id to a list

        future_to_job_index = {}
        for job_index in job_indices:
            job_state = self._get_and_check_job_state(job_index, JobStateEnum.PENDING)
            job_state.state = JobStateEnum.SUBMITTED_ASYNC

            future = self._worker_pool.submit(self.submit_job, job_index, job_queue_id)
            job_state.future = future
            future_to_job_index[future] = job_index

        return future_to_job_index
