# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=broad-except

import json
import logging
import math
import time
from collections import defaultdict
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures import wait
from concurrent.futures import FIRST_COMPLETED
from typing import Any, Type, Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from nv_ingest_api.util.service_clients.client_base import MessageBrokerClientBase
from nv_ingest_api.util.service_clients.rest.rest_client import RestClient
from nv_ingest_client.primitives import BatchJobSpec
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.jobs import JobState
from nv_ingest_client.primitives.jobs import JobStateEnum
from nv_ingest_client.primitives.tasks import Task
from nv_ingest_client.primitives.tasks import TaskType
from nv_ingest_client.primitives.tasks import is_valid_task_type
from nv_ingest_client.primitives.tasks import task_factory
from nv_ingest_client.util.util import create_job_specs_for_batch

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
        message_client_allocator: Type[MessageBrokerClientBase] = RestClient,
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
        self._job_index_to_job_spec = {}
        self._message_client_hostname = message_client_hostname or "localhost"
        self._message_client_port = message_client_port or 7670
        self._message_counter_id = msg_counter_id or "nv-ingest-message-id"
        self._message_client_kwargs = message_client_kwargs or {}

        logger.debug("Instantiate NvIngestClient:\n%s", str(self))
        self._message_client = message_client_allocator(
            host=self._message_client_hostname,
            port=self._message_client_port,
            **self._message_client_kwargs,
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
                f"Job with ID {job_state.job_spec.job_id} has invalid state "
                f"{job_state.state}, expected {required_state}"
            )

        return job_state

    def job_count(self):
        return len(self._job_states)

    def _add_single_job(self, job_spec: JobSpec) -> str:
        job_index = self._generate_job_index()

        self._job_states[job_index] = JobState(job_spec=job_spec)

        return job_index

    def add_job(self, job_spec: Union[BatchJobSpec, JobSpec]) -> Union[str, List[str]]:
        if isinstance(job_spec, JobSpec):
            job_index = self._add_single_job(job_spec)
            self._job_index_to_job_spec[job_index] = job_spec

            return job_index
        elif isinstance(job_spec, BatchJobSpec):
            job_indexes = []
            for _, job_specs in job_spec.job_specs.items():
                for job in job_specs:
                    job_index = self._add_single_job(job)
                    job_indexes.append(job_index)
                    self._job_index_to_job_spec[job_index] = job
            return job_indexes
        else:
            raise ValueError(f"Unexpected type: {type(job_spec)}")

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

        job_id = self.add_job(job_spec)
        return job_id

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

    def _fetch_job_result(
        self, job_index: str, timeout: Tuple[int, Union[float, None]] = (100, None), data_only: bool = False
    ) -> Tuple[Any, str, Optional[str]]:
        """
        Internal method to fetch a job result using its client-side index.
        Handles different response codes from the message client's ResponseSchema.

        Args:
            job_index (str): The client-side identifier of the job.
            timeout (float): Timeout for the fetch operation in seconds.
            data_only (bool): If True, attempts to extract 'data' field from the JSON response.

        Returns:
            Tuple[Any, str, Optional[str]]: The job result (parsed JSON or specific part), client job index,
            and trace_id.

        Raises:
            ValueError: If the job state is invalid, JSON decoding fails, or server_job_id is missing.
            TimeoutError: If the underlying fetch indicates the job is not ready (RestClient response_code == 2).
            RuntimeError: If the underlying fetch returns a terminal error (RestClient response_code == 1, e.g.,
                HTTP 404/400/500).
            Exception: For other unexpected issues during processing.
        """
        try:
            # Get job state using the client-side index
            job_state = self._get_and_check_job_state(
                job_index, required_state=[JobStateEnum.SUBMITTED, JobStateEnum.SUBMITTED_ASYNC]
            )

            # Validate server_job_id before making the call
            server_job_id = job_state.job_id
            if not server_job_id:
                error_msg = (
                    f"Cannot fetch job index {job_index}: Server Job ID is missing or invalid in state"
                    f" {job_state.state}."
                )
                logger.error(error_msg)
                job_state.state = JobStateEnum.FAILED
                raise ValueError(error_msg)

            # Fetch using the *server-assigned* job ID
            response = self._message_client.fetch_message(server_job_id, timeout)
            job_state.trace_id = response.trace_id  # Store trace ID from this fetch attempt

            # --- Handle ResponseSchema Code ---
            if response.response_code == 0:  # Success (e.g., HTTP 200)
                try:
                    # Don't change state here yet, only after successful processing
                    logger.debug(
                        f"Received successful response for job index {job_index} (Server ID: {server_job_id}). "
                        f"Decoding JSON."
                    )

                    response_json = json.loads(response.response)
                    result_data = response_json.get("data") if data_only else response_json

                    # Mark state as PROCESSING *after* successful decode, just before returning
                    job_state.state = JobStateEnum.PROCESSING
                    # Pop state *only* after successful processing is complete
                    self._pop_job_state(job_index)
                    logger.debug(
                        f"Successfully processed and removed job index {job_index} (Server ID: {server_job_id})"
                    )
                    return result_data, job_index, job_state.trace_id

                except json.JSONDecodeError as err:
                    logger.error(
                        f"Failed to decode JSON response for job index {job_index} (Server ID: {server_job_id}):"
                        f" {err}. Response text: {response.response[:500]}"
                    )
                    job_state.state = JobStateEnum.FAILED  # Mark as failed due to decode error
                    raise ValueError(f"Error decoding job result JSON: {err}") from err
                except Exception as e:
                    # Catch other potential errors during processing of successful response
                    logger.exception(
                        f"Error processing successful response for job index {job_index} (Server ID: {server_job_id}):"
                        f" {e}"
                    )
                    job_state.state = JobStateEnum.FAILED
                    raise  # Re-raise unexpected errors

            elif response.response_code == 2:  # Job Not Ready (e.g., HTTP 202)
                # Raise TimeoutError to signal the calling retry loop in fetch_job_result
                logger.debug(
                    f"Job index {job_index} (Server ID: {server_job_id}) not ready (Response Code: 2). Signaling retry."
                )
                # Do not change job state here, remains SUBMITTED
                raise TimeoutError(f"Job not ready: {response.response_reason}")

            else:  # Failure from RestClient (response_code == 1, including 404, 400, 500, conn errors)
                # Log the failure reason from the ResponseSchema
                error_msg = (
                    f"Terminal failure fetching result for client index {job_index} (Server ID: {server_job_id}). "
                    f"Code: {response.response_code}, Reason: {response.response_reason}"
                )
                logger.error(error_msg)
                job_state.state = JobStateEnum.FAILED  # Mark job as failed in the client
                # Do NOT pop the state for failed jobs here
                # Raise RuntimeError to indicate a terminal failure for this fetch attempt
                raise RuntimeError(error_msg)

        except (TimeoutError, ValueError, RuntimeError):
            # Re-raise specific handled exceptions
            raise
        except Exception as err:
            # Catch unexpected errors during the process (e.g., in _get_and_check_job_state)
            logger.exception(f"Unexpected error during fetch process for job index {job_index}: {err}")
            # Attempt to mark state as FAILED if possible and state object exists
            if "job_state" in locals() and hasattr(job_state, "state"):
                job_state.state = JobStateEnum.FAILED
            raise  # Re-raise the original exception

    # The Pythonic invocation and the CLI invocation approach currently have different approaches to timeouts
    # This distinction is made obvious by provided two separate functions. One for "_cli" and one for
    # direct Python use. This is the "_cli" approach
    def fetch_job_result_cli(self, job_ids: Union[str, List[str]], data_only: bool = False):
        if isinstance(job_ids, str):
            job_ids = [job_ids]

        return [self._fetch_job_result(job_id, data_only=data_only) for job_id in job_ids]

    def fetch_job_result(
        self,
        job_indices: Union[str, List[str]],
        timeout: int = 100,
        fetch_batch_size: int = 128,
        max_job_retries: int = None,
        retry_delay: float = 5.0,
        verbose: bool = False,
        completion_callback: Optional[Callable[[Dict, str], None]] = None,
        return_failures: bool = False,
        data_only: bool = False,
    ) -> Union[List[Any], Tuple[List[Any], List[Tuple[str, str]]]]:
        """
        Fetches job results using a batched approach with controlled concurrency and retries.

        Args:
            job_indices: A job ID or list of job IDs to fetch results for.
            timeout: Timeout (seconds) for each underlying fetch connection/read attempt.
            fetch_batch_size: Maximum number of jobs to fetch concurrently.
            max_job_retries: Maximum retries for jobs returning "Not Ready".
            retry_delay: Delay (seconds) between retrying jobs that weren't ready.
            verbose: Log more details about retries.
            completion_callback: Callback executed on successful fetch of a job's result data.
            return_failures: If True, return (results, failures) tuple.
            data_only: If True, attempt to return only the 'data' field from results. (Passed down).

        Returns:
          - If `return_failures=False`: List of successful results [result_data].
          - If `return_failures=True`: Tuple of ([successful results], [failures=(job_index, error_msg)]).

        Raises:
            Propagates exceptions from pre-flight checks or unexpected internal errors.
        """
        if isinstance(job_indices, str):
            job_indices = [job_indices]

        return self._fetch_job_results_batched(
            job_indices=job_indices,
            timeout=timeout,
            fetch_batch_size=fetch_batch_size,
            max_job_retries=max_job_retries,
            retry_delay=retry_delay,
            verbose=verbose,
            completion_callback=completion_callback,
            return_failures=return_failures,
            data_only=data_only,
        )

    def _fetch_job_results_batched(
        self,
        job_indices: List[str],
        timeout: int,
        fetch_batch_size: int,
        max_job_retries: int,
        retry_delay: float,
        verbose: bool,
        completion_callback: Optional[Callable[[Dict, str], None]],
        return_failures: bool,
        data_only: bool,
    ) -> Union[List[Any], Tuple[List[Any], List[Tuple[str, str]]]]:
        """
        Internal worker to fetch results for multiple jobs using batching and external retries.

        Args:
            job_indices: List of client-side job indices to fetch.
            timeout: Timeout (seconds) for individual fetch connect/read attempts.
            fetch_batch_size: Max number of concurrent fetch operations.
            max_job_retries: Max retries for a job if it returns "Not Ready".
            retry_delay: Delay (seconds) between retry cycles.
            verbose: If True, log more detailed retry info.
            completion_callback: Called for each successfully completed job *result*.
            return_failures: If True, return (results, failures) tuple.
            data_only: Passed down to _fetch_job_result.

        Returns:
            List of results or (results, failures) tuple.
        """
        total_jobs = len(job_indices)
        if total_jobs == 0:
            return ([], []) if return_failures else []

        job_ids_to_process = list(job_indices)  # Queue of jobs yet to be submitted
        retry_job_ids = []  # Jobs waiting for retry
        retry_counts = defaultdict(int)  # Tracks retries per job_index
        results = []  # Stores successful results
        failures = []  # Stores (job_index, error_message) tuples
        active_futures: Dict[Future, str] = {}  # Maps active Future instances to job_index

        executor = self._worker_pool

        effective_fetch_timeout = (timeout, None)
        logger.info(
            f"Starting batched fetch for {total_jobs} jobs. Batch size: {fetch_batch_size}, "
            f"Max retries: {max_job_retries}, Retry delay: {retry_delay}s"
        )

        processed_count = 0  # Counter for jobs that finished (success or terminal failure)

        while processed_count < total_jobs:
            # --- Submit New/Retry Tasks ---
            can_submit = fetch_batch_size - len(active_futures)

            # Prioritize retries
            submit_now = min(can_submit, len(retry_job_ids))
            if submit_now > 0:
                ids_to_retry = retry_job_ids[:submit_now]
                del retry_job_ids[:submit_now]  # Consume submitted retries
                can_submit -= submit_now
                if verbose:
                    logger.info(f"Submitting {len(ids_to_retry)} jobs from retry queue.")
                for job_index in ids_to_retry:
                    future = executor.submit(
                        self._fetch_job_result,  # Direct call to single-job fetcher
                        job_index,
                        timeout=effective_fetch_timeout,
                        data_only=data_only,
                    )
                    active_futures[future] = job_index

            # Submit new jobs if space available
            submit_now = min(can_submit, len(job_ids_to_process))
            if submit_now > 0:
                ids_to_process = job_ids_to_process[:submit_now]
                del job_ids_to_process[:submit_now]  # Consume submitted new jobs
                if verbose:
                    logger.info(f"Submitting {len(ids_to_process)} new jobs for fetch.")
                for job_index in ids_to_process:
                    future = executor.submit(
                        self._fetch_job_result, job_index, timeout=effective_fetch_timeout, data_only=data_only
                    )
                    active_futures[future] = job_index

            # --- Wait for and Process Completed Futures ---
            if not active_futures:
                # This might happen if all remaining jobs hit max retries quickly
                if len(results) + len(failures) >= total_jobs:
                    break  # All jobs accounted for
                else:
                    # Should not happen if loop condition is correct, but adds robustness
                    logger.warning("No active futures, but not all jobs accounted for. Waiting...")
                    time.sleep(max(retry_delay, 0.5))  # Wait before checking submission logic again
                    continue

            # Wait for at least one future to complete. Use a small timeout to allow
            # the loop to cycle reasonably often, even if fetches are slow.
            done, _ = wait(list(active_futures.keys()), timeout=1.0, return_when=FIRST_COMPLETED)

            needs_retry_delay = False
            for future in done:
                job_index = active_futures.pop(future)  # Remove processed future
                will_retry = False
                try:
                    # Result is Tuple[Any, str, Optional[str]] from _fetch_job_result
                    result_data, _, trace_id = future.result()  # Raises exceptions from _fetch_job_result

                    # Success Case
                    if verbose:
                        logger.info(f"Successfully fetched result for job {job_index} (Trace: {trace_id})")
                    results.append(result_data)  # Store the actual result payload
                    if completion_callback:
                        try:
                            # Pass the full result dict if needed by callback, or just result_data
                            # Assuming callback expects the data part from _fetch_job_result's tuple
                            completion_callback(result_data, job_index)
                        except Exception as cb_err:
                            logger.error(f"Error in completion_callback for job {job_index}: {cb_err}")

                except TimeoutError as e:  # Raised by _fetch_job_result for "Not Ready"
                    retry_counts[job_index] += 1
                    if max_job_retries is None or retry_counts[job_index] <= max_job_retries:
                        if verbose:
                            logger.info(
                                f"Job {job_index} not ready (Attempt {retry_counts[job_index]}/{max_job_retries})."
                                f" Will retry."
                            )
                        retry_job_ids.append(job_index)  # Add back to retry queue
                        will_retry = True
                        needs_retry_delay = True
                    else:
                        error_msg = (
                            f"Job {job_index} failed: Exceeded max retries ({max_job_retries})"
                            f" waiting for readiness. Last reason: {e}"
                        )
                        logger.error(error_msg)
                        failures.append((job_index, error_msg))

                except (ValueError, RuntimeError) as e:  # Terminal error from _fetch_job_result
                    error_msg = f"Job {job_index} failed: Terminal error during fetch: {e}"
                    logger.error(error_msg)
                    failures.append((job_index, str(e)))
                    # State should have been set to FAILED inside _fetch_job_result

                except Exception as e:  # Other unexpected errors
                    error_msg = f"Job {job_index} failed: Unexpected error processing fetch future: {e}"
                    logger.exception(error_msg)
                    failures.append((job_index, f"Unexpected future processing error: {e}"))
                    # Attempt to mark state FAILED if _fetch_job_result didn't
                    try:
                        job_state = self._get_and_check_job_state(job_index, required_state=None)
                        if job_state and job_state.state not in [
                            JobStateEnum.FAILED
                        ]:  # Avoid overwriting specific failures
                            job_state.state = JobStateEnum.FAILED
                    except Exception as state_err:
                        logger.error(
                            f"Could not update state to FAILED for job {job_index} after unexpected error: {state_err}"
                        )

                finally:
                    if not will_retry:
                        processed_count += 1

            # Apply delay only if a retry was triggered in this cycle
            if needs_retry_delay and retry_delay > 0:
                logger.debug(f"Waiting {retry_delay}s due to jobs needing retry...")
                time.sleep(retry_delay)

        # --- Loop Finished ---
        if len(results) + len(failures) != total_jobs:
            logger.warning(
                f"Batch fetch completed. Final counts mismatch: Results={len(results)},"
                f" Failures={len(failures)}, Total={total_jobs}"
            )
            # Add any remaining jobs from retry/processing queues as failures?
            remaining_ids = set(retry_job_ids) | set(job_ids_to_process) | set(active_futures.values())
            for job_index in remaining_ids:
                if not any(f[0] == job_index for f in failures):
                    failures.append((job_index, "Job did not complete within fetch process"))

        logger.info(f"Batch fetch finished. Success: {len(results)}, Failures: {len(failures)}")

        if return_failures:
            return results, failures
        else:
            if failures:
                logger.warning(
                    f"Completed fetching batch. {len(results)} succeeded, {len(failures)} failed (check logs)."
                )
            return results

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
            job_state.trace_id = future.result()[0]  # Trace_id from `submit_job` endpoint submission
            job_state.future = None

    def fetch_job_result_async(self, job_ids: Union[str, List[str]], data_only: bool = True) -> Dict[Future, str]:
        """
        Fetches job results for a list or a single job ID asynchronously and returns a mapping of futures to job IDs.

        Parameters:
            job_ids (Union[str, List[str]]): A single job ID or a list of job IDs.
            timeout (float): Timeout (connect, read) for fetching each job result, in seconds.
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
            future = self._worker_pool.submit(self.fetch_job_result_cli, job_id, data_only)
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

            response = self._message_client.submit_message(job_queue_id, message, for_nv_ingest=True)
            x_trace_id = response.trace_id
            transaction_id = response.transaction_id
            job_id = "" if transaction_id is None else transaction_id.replace('"', "")
            logger.debug(f"Submitted job {job_index} to queue {job_queue_id} and got back job ID {job_id}")

            job_state.state = JobStateEnum.SUBMITTED
            job_state.job_id = job_id

            # Free up memory -- payload should never be used again, and we don't want to keep it around.
            job_state.job_spec.payload = None

            return x_trace_id
        except Exception as err:
            err_msg = f"Failed to submit job {job_index} to queue {job_queue_id}: {err}"
            logger.exception(err_msg)
            job_state.state = JobStateEnum.FAILED

            raise

    def submit_job(
        self, job_indices: Union[str, List[str]], job_queue_id: str, batch_size: int = 10
    ) -> List[Union[Dict, None]]:
        if isinstance(job_indices, str):
            job_indices = [job_indices]

        results = []
        total_batches = math.ceil(len(job_indices) / batch_size)

        submission_errors = []
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = batch_start + batch_size
            batch = job_indices[batch_start:batch_end]

            # Submit each batch of jobs
            for job_id in batch:
                try:
                    x_trace_id = self._submit_job(job_id, job_queue_id)
                except Exception as e:  # Even if one fails, we should continue with the rest of the batch.
                    submission_errors.append(e)
                    continue
                results.append(x_trace_id)

        if submission_errors:
            error_msg = str(submission_errors[0])
            if len(submission_errors) > 1:
                error_msg += f"... [{len(submission_errors) - 1} more messages truncated]"
            raise type(submission_errors[0])(error_msg)
        return results

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

    def create_jobs_for_batch(self, files_batch: List[str], tasks: Dict[str, Any]) -> List[str]:
        """
        Create and submit job specifications (JobSpecs) for a batch of files, returning the job IDs.
        This function takes a batch of files, processes each file to extract its content and type,
        creates a job specification (JobSpec) for each file, and adds tasks from the provided task
        list. It then submits the jobs to the client and collects their job IDs.

        Parameters
        ----------
        files_batch : List[str]
            A list of file paths to be processed. Each file is assumed to be in a format compatible
            with the `extract_file_content` function, which extracts the file's content and type.
        tasks : Dict[str, Any]
            A dictionary of tasks to be added to each job. The keys represent task names, and the
            values represent task specifications or configurations. Standard tasks include "split",
            "extract", "store", "caption", "dedup", "filter", "embed".

        Returns
        -------
        Tuple[List[JobSpec], List[str]]
            A Tuple containing the list of JobSpecs and list of job IDs corresponding to the submitted jobs.
            Each job ID is returned by the client's `add_job` method.

        Raises
        ------
        ValueError
            If there is an error extracting the file content or type from any of the files, a
            ValueError will be logged, and the corresponding file will be skipped.

        Notes
        -----
        - The function assumes that a utility function `extract_file_content` is defined elsewhere,
          which extracts the content and type from the provided file paths.
        - For each file, a `JobSpec` is created with relevant metadata, including document type and
          file content. Various tasks are conditionally added based on the provided `tasks` dictionary.
        - The job specification includes tracing options with a timestamp (in nanoseconds) for
          diagnostic purposes.

        Examples
        --------
        Suppose you have a batch of files and tasks to process:
        >>> files_batch = ["file1.txt", "file2.pdf"]
        >>> tasks = {"split": ..., "extract_txt": ..., "store": ...}
        >>> client = NvIngestClient()
        >>> job_ids = client.create_job_specs_for_batch(files_batch, tasks)
        >>> print(job_ids)
        ['job_12345', 'job_67890']

        In this example, jobs are created and submitted for the files in `files_batch`, with the
        tasks in `tasks` being added to each job specification. The returned job IDs are then
        printed.

        See Also
        --------
        create_job_specs_for_batch: Function that creates job specifications for a batch of files.
        JobSpec : The class representing a job specification.
        """
        if not isinstance(tasks, dict):
            raise ValueError("`tasks` must be a dictionary of task names -> task specifications.")

        job_specs = create_job_specs_for_batch(files_batch)

        job_ids = []
        for job_spec in job_specs:
            logger.debug(f"Tasks: {tasks.keys()}")
            for task in tasks:
                logger.debug(f"Task: {task}")

            file_type = job_spec.document_type

            seen_tasks = set()  # For tracking tasks and rejecting duplicate tasks.

            for task_name, task_config in tasks.items():
                if task_name.lower().startswith("extract_"):
                    task_file_type = task_name.split("_", 1)[1]
                    if file_type.lower() != task_file_type.lower():
                        continue
                elif not is_valid_task_type(task_name.upper()):
                    raise ValueError(f"Invalid task type: '{task_name}'")

                if str(task_config) in seen_tasks:
                    raise ValueError(f"Duplicate task detected: {task_name} with config {task_config}")

                job_spec.add_task(task_config)

                seen_tasks.add(str(task_config))

            job_id = self.add_job(job_spec)
            job_ids.append(job_id)

        return job_ids
