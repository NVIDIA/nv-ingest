# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=broad-except

import concurrent
import json
import logging
import math
import time
from collections import defaultdict
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
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
from nv_ingest_client.util.processing import handle_future_result, IngestJobFailure
from nv_ingest_client.util.util import create_job_specs_for_batch, check_ingest_result

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


class _ConcurrentProcessor:
    """
    Manages the asynchronous submission and result fetching of jobs using a
    client's public methods, mirroring the batching structure of the CLI path.

    This processor takes a list of pre-created job indices, submits them in
    batches via the client's `submit_job_async`, and then fetches results
    for each batch using `fetch_job_result_async`. It processes results as
    they become available within the batch using `as_completed`. Retries due
    to job readiness timeouts are handled by adding the job index to the next
    processing batch.
    """

    def __init__(
        self,
        client: "NvIngestClient",
        job_indices: List[str],
        job_queue_id: Optional[str],
        batch_size: int,
        timeout: Tuple[int, Union[float, None]],
        max_job_retries: Optional[int],
        completion_callback: Optional[Callable[[Dict[str, Any], str], None]],
        fail_on_submit_error: bool,
        verbose: bool = False,
    ):
        """
        Initializes the concurrent processor.

        Parameters
        ----------
        client : NvIngestClient
            The client instance used for job operations. Requires methods:
            `_worker_pool`, `submit_job_async`, `fetch_job_result_async`,
            `_get_job_state_object`.
        job_indices : List[str]
            List of pre-created, unique job indices to process.
        job_queue_id : Optional[str]
            The ID of the job queue required for submitting new jobs via
            `submit_job_async`.
        batch_size : int
            Maximum number of jobs to include in each processing batch.
        timeout : Tuple[int, Union[float, None]]
            Timeout configuration potentially used by underlying client fetch
            operations. Its direct usage depends on the client implementation.
        max_job_retries : Optional[int]
            Maximum number of times to retry fetching a job result if the
            initial fetch attempt times out (indicating the job is not ready).
            `None` indicates infinite retries.
        completion_callback : Optional[Callable[[Dict[str, Any], str], None]]
            A callback function executed upon the successful completion and
            fetch of a job. It receives the full response dictionary and the
            job index.
        fail_on_submit_error : bool
            If True, the entire process will stop and raise an error if
            initiating job submission or fetching fails for a batch.
        verbose : bool, optional
            If True, enables detailed debug logging. Default is False.

        Raises
        ------
        AttributeError
            If the provided client object is missing required methods or
            attributes.
        TypeError
            If the client's `_worker_pool` is not a `ThreadPoolExecutor`.
        """
        self.client = client
        self.all_job_indices_list: List[str] = list(job_indices)
        self.job_queue_id = job_queue_id
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_job_retries = max_job_retries
        self.completion_callback = completion_callback
        self.fail_on_submit_error = fail_on_submit_error
        self.verbose = verbose

        # State variables managed across batch cycles
        self.retry_job_ids: List[str] = []
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.results: List[Dict[str, Any]] = []  # Stores successful results (full dicts)
        self.failures: List[Tuple[str, str]] = []  # (job_index, error_message)

        # --- Initial Checks ---
        if not self.job_queue_id:
            logger.warning("job_queue_id is not set; submission of new jobs will fail.")

    # --------------------------------------------------------------------------
    # Private Methods
    # --------------------------------------------------------------------------

    def _handle_processing_failure(self, job_index: str, error_msg: str, is_submission_failure: bool = False) -> None:
        """
        Handles terminal failures during job initiation or processing.

        Logs the error, records the failure, cleans up retry counts, and
        attempts to update the job's state locally in the client. This method
        does not increment the overall processed count itself.

        Parameters
        ----------
        job_index : str
            The unique identifier of the job that failed.
        error_msg : str
            A message describing the reason for the failure.
        is_submission_failure : bool, optional
            If True, indicates the failure occurred during the submission or
            fetch initiation phase, rather than during result processing.
            Default is False.
        """
        log_prefix = "Initiation failed" if is_submission_failure else "Processing failed"
        # Log validation failures less prominently if they are noisy
        if "validation failed" in error_msg:
            logger.warning(f"{log_prefix} for {job_index}: {error_msg}")
        else:
            logger.error(f"{log_prefix} for {job_index}: {error_msg}")

        # Record failure only once per job index
        if not any(f[0] == job_index for f in self.failures):
            failed_job_spec = self.client._job_index_to_job_spec.get(job_index)
            self.failures.append((f"{job_index}:{failed_job_spec.source_id}", error_msg))
        elif self.verbose:
            logger.debug(f"Failure already recorded for {job_index}")

        # Cleanup retry count if it exists for this job
        if job_index in self.retry_counts:
            del self.retry_counts[job_index]

        # Attempt to mark state as FAILED locally in the client (best effort)
        try:
            # Use a method assumed to safely get the state object
            job_state = self.client._get_job_state_object(job_index)
            # Check state exists and is not already terminal before updating
            if (
                job_state and hasattr(job_state, "state") and job_state.state not in ["FAILED", "COMPLETED"]
            ):  # Use actual Enum names/values if available
                job_state.state = "FAILED"  # Use actual Enum value
                if self.verbose:
                    logger.debug(f"Marked job {job_index} state as FAILED locally " f"after error.")
        except Exception as state_update_err:
            # Ignore errors during error handling state update, but log if verbose
            if self.verbose:
                logger.warning(
                    f"Could not update state to FAILED for job {job_index} " f"after failure: {state_update_err}"
                )

    def _handle_processing_success(self, job_index: str, result_data: Dict[str, Any], trace_id: Optional[str]) -> None:
        """
        Handles the successful fetch and retrieval of a job result.

        Stores the result, cleans up retry counts, and triggers the completion
        callback if provided. This method does not increment the overall
        processed count itself.

        Parameters
        ----------
        job_index : str
            The unique identifier of the successfully processed job.
        result_data : Dict[str, Any]
            The full response dictionary fetched for the job.
        trace_id : Optional[str]
            The trace identifier associated with the fetch operation, if available.
        """
        if self.verbose:
            trace_info = f" (Trace: {trace_id})" if trace_id else ""
            logger.info(f"Successfully fetched result for job {job_index}{trace_info}")

        is_failed, description = check_ingest_result(result_data)

        if is_failed:
            failed_job_spec = self.client._job_index_to_job_spec.get(job_index)
            self.failures.append((f"{job_index}:{failed_job_spec.source_id}", description))
        else:
            self.results.append(result_data.get("data"))

        # Cleanup retry count if it exists
        if job_index in self.retry_counts:
            del self.retry_counts[job_index]

        # Execute completion callback if provided
        if self.completion_callback:
            try:
                self.completion_callback(result_data, job_index)
            except Exception as cb_err:
                logger.error(f"Error in completion_callback for {job_index}: {cb_err}", exc_info=True)

    def _log_final_status(self, total_jobs: int) -> None:
        """
        Logs the final processing summary and checks for count discrepancies.

        Parameters
        ----------
        total_jobs : int
            The total number of jobs that were initially intended for processing.
        """
        final_processed_count = len(self.results) + len(self.failures)
        logger.info(
            f"Batch processing finished. Success: {len(self.results)}, "
            f"Failures: {len(self.failures)}. "
            f"Total accounted for: {final_processed_count}/{total_jobs}"
        )
        if final_processed_count != total_jobs:
            logger.warning(
                "Final accounted count doesn't match total jobs. " "Some jobs may have been lost or unaccounted for."
            )
            # Attempt to identify potentially lost jobs (best effort)
            processed_indices = {f[0] for f in self.failures}
            # Assuming results contain the job index or can be mapped back
            # If result_data is dict and has 'jobIndex':
            try:
                result_indices = {r.get("jobIndex") for r in self.results if isinstance(r, dict) and "jobIndex" in r}
                # Filter out None if get returns None
                result_indices = {idx for idx in result_indices if idx is not None}
                processed_indices.update(result_indices)
            except Exception:
                logger.warning("Could not reliably extract job indices from results for final check.")

            initial_indices = set(self.all_job_indices_list)
            unaccounted_indices = initial_indices - processed_indices

            if unaccounted_indices:
                logger.warning(f"Potentially unaccounted for jobs: {unaccounted_indices}")
                # Optionally add them to failures
                # for idx in unaccounted_indices:
                #     if not any(f[0] == idx for f in self.failures):
                #         self.failures.append((idx, "Job lost or unaccounted for at exit"))

    # --------------------------------------------------------------------------
    # Public Methods
    # --------------------------------------------------------------------------

    def run(self) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        """
        Executes the main processing loop in batches.

        This method orchestrates the job processing by repeatedly determining
        a batch of jobs (including retries), initiating their submission (if new)
        and fetching, and then processing the results of that batch as they
        complete.

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]
            A tuple containing two lists:
            1. A list of successfully fetched job results (full dictionaries).
            2. A list of tuples for failed jobs, where each tuple contains
               (job_index, error_message).

        Raises
        ------
        ValueError
             If `submit_job_async` is required but `job_queue_id` was not provided.
        RuntimeError
             If `fail_on_submit_error` is True and a batch submission or fetch
             initiation error occurs.
        """
        total_jobs = len(self.all_job_indices_list)
        # Tracks indices for which submission has been initiated at least once
        submitted_new_indices_count = 0

        logger.info(f"Starting batch processing for {total_jobs} jobs with batch " f"size {self.batch_size}.")

        # Main loop: continues as long as there are new jobs to submit
        # or jobs waiting for retry.
        while (submitted_new_indices_count < total_jobs) or self.retry_job_ids:

            # --- Determine Jobs for Current Batch ---
            current_batch_job_indices: List[str] = []

            # Add retries from the previous batch first
            if self.retry_job_ids:
                num_retries = len(self.retry_job_ids)
                current_batch_job_indices.extend(self.retry_job_ids)
                if self.verbose:
                    logger.debug(f"Adding {num_retries} retry jobs to current batch.")
                # Clear the list; retries for *this* batch will be collected later
                self.retry_job_ids = []

            # Determine and add new jobs to the batch
            num_already_in_batch = len(current_batch_job_indices)
            if (num_already_in_batch < self.batch_size) and (submitted_new_indices_count < total_jobs):
                num_new_to_add = min(self.batch_size - num_already_in_batch, total_jobs - submitted_new_indices_count)
                start_idx = submitted_new_indices_count
                end_idx = submitted_new_indices_count + num_new_to_add
                current_batch_new_job_indices = self.all_job_indices_list[start_idx:end_idx]

                if self.verbose:
                    logger.debug(f"Adding {len(current_batch_new_job_indices)} new " f"jobs to current batch.")

                # Initiate async submission for ONLY the NEW jobs
                if current_batch_new_job_indices:
                    if not self.job_queue_id:
                        error_msg = "Cannot submit new jobs: job_queue_id is not set."
                        logger.error(error_msg)
                        # Fail these jobs immediately
                        for job_index in current_batch_new_job_indices:
                            self._handle_processing_failure(job_index, error_msg, is_submission_failure=True)
                        # Mark as "submitted" (to prevent reprocessing) but failed
                        submitted_new_indices_count += len(current_batch_new_job_indices)
                        if self.fail_on_submit_error:
                            raise ValueError(error_msg)
                    else:
                        try:
                            # Fire-and-forget submission initiation
                            _ = self.client.submit_job_async(current_batch_new_job_indices, self.job_queue_id)
                            # Add successfully initiated jobs to the overall batch list
                            current_batch_job_indices.extend(current_batch_new_job_indices)
                            # Update count of total initiated jobs
                            submitted_new_indices_count += len(current_batch_new_job_indices)
                        except Exception as e:
                            error_msg = (
                                f"Batch async submission initiation failed for "
                                f"{len(current_batch_new_job_indices)} new jobs: {e}"
                            )
                            logger.error(error_msg, exc_info=True)
                            # Fail these jobs immediately
                            for job_index in current_batch_new_job_indices:
                                self._handle_processing_failure(
                                    job_index, f"Batch submission initiation error: {e}", is_submission_failure=True
                                )
                            # Mark as "submitted" (to prevent reprocessing) but failed
                            submitted_new_indices_count += len(current_batch_new_job_indices)
                            if self.fail_on_submit_error:
                                raise RuntimeError(error_msg) from e

            # If nothing ended up in the batch (e.g., only submission failures)
            if not current_batch_job_indices:
                if self.verbose:
                    logger.debug("No jobs identified for fetching in this batch iteration.")
                # If there are no retries pending either, break the loop
                if not self.retry_job_ids and submitted_new_indices_count >= total_jobs:
                    logger.debug("Exiting loop: No jobs to fetch and no retries pending.")
                    break
                continue  # Otherwise, proceed to next iteration

            # --- Initiate Fetching for the Current Batch ---
            try:
                if self.verbose:
                    logger.debug(
                        f"Calling fetch_job_result_async for "
                        f"{len(current_batch_job_indices)} jobs in current batch."
                    )
                # Use data_only=False to get full response for callback/results
                batch_futures_dict = self.client.fetch_job_result_async(current_batch_job_indices, data_only=False)

                # Check for discrepancies where client might not return all futures
                if len(batch_futures_dict) != len(current_batch_job_indices):
                    returned_indices = set(batch_futures_dict.values())
                    missing_indices = [idx for idx in current_batch_job_indices if idx not in returned_indices]
                    logger.error(
                        f"fetch_job_result_async discrepancy: Expected "
                        f"{len(current_batch_job_indices)}, got "
                        f"{len(batch_futures_dict)}. Missing: {missing_indices}"
                    )
                    # Fail the missing ones explicitly
                    for missing_idx in missing_indices:
                        self._handle_processing_failure(
                            missing_idx, "Future not returned by fetch_job_result_async", is_submission_failure=True
                        )
                    if self.fail_on_submit_error:
                        raise RuntimeError("fetch_job_result_async failed to return all " "expected futures.")
                    # Continue processing only the futures we received
                    current_batch_job_indices = list(returned_indices)

            except Exception as fetch_init_err:
                error_msg = (
                    f"fetch_job_result_async failed for batch "
                    f"({len(current_batch_job_indices)} jobs): {fetch_init_err}"
                )
                logger.error(error_msg, exc_info=True)
                logger.warning(
                    f"Marking all {len(current_batch_job_indices)} jobs in " f"failed fetch initiation batch as failed."
                )
                # Fail all jobs intended for this batch
                for job_index in current_batch_job_indices:
                    self._handle_processing_failure(
                        job_index, f"Fetch initiation failed for batch: {fetch_init_err}", is_submission_failure=True
                    )
                if self.fail_on_submit_error:
                    raise RuntimeError(
                        f"Stopping due to fetch initiation failure: {fetch_init_err}"
                    ) from fetch_init_err
                continue  # Skip processing results for this failed batch

            # --- Process Results for the Current Batch ---
            if not batch_futures_dict:
                if self.verbose:
                    logger.debug("No futures returned/available for processing in this batch.")
                continue  # Skip processing if no futures

            batch_timeout = 600.0  # Timeout for waiting on the whole batch
            try:
                # Process futures as they complete within this batch
                for future in as_completed(batch_futures_dict.keys(), timeout=batch_timeout):
                    job_index = batch_futures_dict[future]
                    try:
                        # Expect list with one tuple: [(data, index, trace)]
                        result_list = future.result()
                        if not isinstance(result_list, list) or len(result_list) != 1:
                            raise ValueError(f"Expected list length 1, got {len(result_list)}")

                        result_tuple = result_list[0]
                        if not isinstance(result_tuple, (tuple, list)) or len(result_tuple) != 3:
                            raise ValueError(f"Expected tuple/list length 3, got {len(result_tuple)}")

                        full_response_dict, fetched_job_index, trace_id = result_tuple

                        if fetched_job_index != job_index:
                            logger.warning(f"Mismatch: Future for {job_index} returned " f"{fetched_job_index}")

                        self._handle_processing_success(job_index, full_response_dict, trace_id)

                    except TimeoutError:
                        # Handle job not ready - check retry policy
                        self.retry_counts[job_index] += 1
                        if self.max_job_retries is None or self.retry_counts[job_index] <= self.max_job_retries:
                            if self.verbose:
                                logger.info(
                                    f"Job {job_index} not ready, adding to next "
                                    f"batch's retry list (Attempt "
                                    f"{self.retry_counts[job_index]}/"
                                    f"{self.max_job_retries or 'inf'})."
                                )
                            # Collect for the *next* batch
                            self.retry_job_ids.append(job_index)
                        else:
                            error_msg = f"Exceeded max fetch retries " f"({self.max_job_retries}) for job {job_index}."
                            logger.error(error_msg)
                            self._handle_processing_failure(job_index, error_msg)

                    except (ValueError, RuntimeError) as e:
                        logger.error(f"Job {job_index} failed processing result: {e}", exc_info=self.verbose)
                        self._handle_processing_failure(job_index, f"Error processing result: {e}")
                    except Exception as e:
                        logger.exception(f"Unhandled error processing future for job {job_index}: {e}")
                        self._handle_processing_failure(job_index, f"Unhandled error processing future: {e}")
                    # No finally block incrementing count here; tracking is batch-based

            except TimeoutError:
                # `as_completed` timed out waiting for remaining futures in batch
                logger.error(
                    f"Batch processing timed out after {batch_timeout}s waiting "
                    f"for futures. Some jobs in batch may be lost or incomplete."
                )
                # Identify and fail remaining futures
                remaining_indices_in_batch = []
                for f, idx in batch_futures_dict.items():
                    if not f.done():
                        remaining_indices_in_batch.append(idx)
                        f.cancel()  # Attempt to cancel underlying task
                logger.warning(
                    f"Jobs potentially lost/cancelled due to batch timeout: " f"{remaining_indices_in_batch}"
                )
                for idx in remaining_indices_in_batch:
                    self._handle_processing_failure(idx, f"Batch processing timed out after {batch_timeout}s")
            # End of processing for this batch cycle

        # --- Final Logging ---
        self._log_final_status(total_jobs)

        return self.results, self.failures


class NvIngestClient:
    """
    A client class for interacting with the nv-ingest service, supporting custom client allocators.
    """

    def __init__(
        self,
        message_client_allocator: Type[MessageBrokerClientBase] = RestClient,
        message_client_hostname: Optional[str] = "localhost",
        message_client_port: Optional[int] = 7670,
        message_client_kwargs: Optional[Dict[str, Any]] = None,
        msg_counter_id: Optional[str] = "nv-ingest-message-id",
        worker_pool_size: int = 8,
    ) -> None:
        """
        Initialize the NvIngestClient.

        Parameters
        ----------
        message_client_allocator : Type[MessageBrokerClientBase], optional
            Callable that creates the message broker client. Defaults to RestClient.
        message_client_hostname : str, optional
            Hostname of the REST/message service. Defaults to "localhost".
        message_client_port : int, optional
            Port of the REST/message service. Defaults to 7670.
        message_client_kwargs : dict, optional
            Extra keyword arguments passed to the client allocator.
        msg_counter_id : str, optional
            Identifier for message counting. Defaults to "nv-ingest-message-id".
        worker_pool_size : int, optional
            Number of workers in the thread pool. Defaults to 1.

        Returns
        -------
        None
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
        required_state: Optional[Union[JobStateEnum, List[JobStateEnum]]] = None,
    ) -> JobState:
        """
        Retrieve and optionally validate the state of a job.

        Parameters
        ----------
        job_index : str
            The client-side identifier of the job.
        required_state : JobStateEnum or list of JobStateEnum, optional
            State or list of states the job must currently be in. If provided and
            the job is not in one of these states, an error is raised.

        Returns
        -------
        JobState
            The state object for the specified job.

        Raises
        ------
        ValueError
            If the job does not exist or is not in an allowed state.
        """
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

    def job_count(self) -> int:
        """
        Get the number of jobs currently tracked by the client.

        Returns
        -------
        int
            The total count of jobs in internal state tracking.
        """
        return len(self._job_states)

    def _add_single_job(self, job_spec: JobSpec) -> str:
        """
        Add a single job specification to internal tracking.

        Parameters
        ----------
        job_spec : JobSpec
            The specification object describing the job.

        Returns
        -------
        str
            The newly generated job index.
        """
        job_index = self._generate_job_index()

        self._job_states[job_index] = JobState(job_spec=job_spec)

        return job_index

    def add_job(self, job_spec: Union[BatchJobSpec, JobSpec]) -> Union[str, List[str]]:
        """
        Add one or more jobs to the client for later processing.

        Parameters
        ----------
        job_spec : JobSpec or BatchJobSpec
            A single job specification or a batch containing multiple specs.

        Returns
        -------
        str or list of str
            The job index for a single spec, or a list of indices for a batch.

        Raises
        ------
        ValueError
            If an unsupported type is provided.
        """
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
        payload: Dict[str, Any],
        source_id: str,
        source_name: str,
        document_type: Optional[str] = None,
        tasks: Optional[List[Task]] = None,
        extended_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Construct and register a new job from provided metadata.

        Parameters
        ----------
        payload : dict
            The data payload for the job.
        source_id : str
            Identifier of the data source.
        source_name : str
            Human-readable name for the source.
        document_type : str, optional
            Type of document (inferred from source_name if omitted).
        tasks : list of Task, optional
            Initial set of processing tasks to attach.
        extended_options : dict, optional
            Extra parameters for advanced configuration.

        Returns
        -------
        str
            The client-side job index.

        Raises
        ------
        ValueError
            If job creation parameters are invalid.
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
        """
        Attach an existing Task object to a pending job.

        Parameters
        ----------
        job_index : str
            The client-side identifier of the target job.
        task : Task
            The task instance to add.
        """
        job_state = self._get_and_check_job_state(job_index, required_state=JobStateEnum.PENDING)

        job_state.job_spec.add_task(task)

    def create_task(
        self,
        job_index: Union[str, int],
        task_type: TaskType,
        task_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create and attach a new task to a pending job by type and parameters.

        Parameters
        ----------
        job_index : str or int
            Identifier of the job to modify.
        task_type : TaskType
            Enum specifying the kind of task to create.
        task_params : dict, optional
            Parameters for the new task.

        Raises
        ------
        ValueError
            If the job does not exist or is not pending.
        """
        task_params = task_params or {}

        return self.add_task(job_index, task_factory(task_type, **task_params))

    def _fetch_job_result(
        self,
        job_index: str,
        timeout: Tuple[int, Optional[float]] = (100, None),
        data_only: bool = False,
    ) -> Tuple[Any, str, Optional[str]]:
        """
        Retrieve the result of a submitted job, handling status codes.

        Parameters
        ----------
        job_index : str
            Client-side job identifier.
        timeout : tuple
            Timeouts (connect, read) for the fetch operation.
        data_only : bool, optional
            If True, return only the 'data' portion of the payload.

        Returns
        -------
        result_data : any
            Parsed job result or full JSON payload.
        job_index : str
            Echoes the client-side job ID.
        trace_id : str or None
            Trace identifier from the message client.

        Raises
        ------
        TimeoutError
            If the job is not yet ready (HTTP 202).
        RuntimeError
            For terminal server errors (HTTP 404/500, etc.).
        ValueError
            On JSON decoding errors or missing state.
        Exception
            For unexpected issues.
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

    def fetch_job_result_cli(
        self,
        job_ids: Union[str, List[str]],
        data_only: bool = False,
    ) -> List[Tuple[Any, str, Optional[str]]]:
        """
        Fetch job results via CLI semantics (synchronous list return).

        Parameters
        ----------
        job_ids : str or list of str
            Single or multiple client-side job identifiers.
        data_only : bool, optional
            If True, extract only the 'data' field. Default is False.

        Returns
        -------
        list of (result_data, job_index, trace_id)
            List of tuples for each fetched job.
        """
        if isinstance(job_ids, str):
            job_ids = [job_ids]

        return [self._fetch_job_result(job_id, data_only=data_only) for job_id in job_ids]

    def process_jobs_concurrently(
        self,
        job_indices: Union[str, List[str]],
        job_queue_id: Optional[str] = None,
        concurrency_limit: int = 64,
        timeout: int = 100,
        max_job_retries: Optional[int] = None,
        retry_delay: float = 5.0,
        fail_on_submit_error: bool = False,
        completion_callback: Optional[Callable[[Any, str], None]] = None,
        return_failures: bool = False,
        data_only: bool = True,
        verbose: bool = False,
    ) -> Union[List[Any], Tuple[List[Any], List[Tuple[str, str]]]]:
        """
        Submit and fetch multiple jobs concurrently.

        Parameters
        ----------
        job_indices : str or list of str
            Single or multiple job indices to process.
        job_queue_id : str, optional
            Queue identifier for submission.
        concurrency_limit : int, optional
            Max number of simultaneous in-flight jobs. Default is 128.
        timeout : int, optional
            Timeout in seconds per fetch attempt. Default is 100.
        max_job_retries : int, optional
            Max retries for 'not ready' jobs. None for infinite. Default is None.
        retry_delay : float, optional
            Delay in seconds between retry cycles. Default is 5.0.
        fail_on_submit_error : bool, optional
            If True, abort on submission error. Default is False.
        completion_callback : callable, optional
            Called on each successful fetch as (result_data, job_index).
        return_failures : bool, optional
            If True, return (results, failures). Default is False.
        data_only : bool, optional
            If True, return only payload 'data'. Default is True.
        verbose : bool, optional
            If True, enable debug logging. Default is False.

        Returns
        -------
        results : list
            List of successful job results when `return_failures` is False.
        results, failures : tuple
            Tuple of (successful results, failure tuples) when `return_failures` is True.

        Raises
        ------
        RuntimeError
            If `fail_on_submit_error` is True and a submission fails.
        """
        # Normalize single index to list
        if isinstance(job_indices, str):
            job_indices = [job_indices]

        # Handle empty input
        if not job_indices:
            return ([], []) if return_failures else []

        # Prepare timeout tuple for fetch calls
        effective_timeout: Tuple[int, None] = (timeout, None)

        # Delegate to the concurrent processor
        processor = _ConcurrentProcessor(
            client=self,
            batch_size=64,
            job_indices=job_indices,
            job_queue_id=job_queue_id,
            timeout=effective_timeout,
            max_job_retries=max_job_retries,
            completion_callback=completion_callback,
            fail_on_submit_error=fail_on_submit_error,
            verbose=verbose,
        )

        results, failures = processor.run()

        if return_failures:
            return results, failures

        if failures:
            logger.warning(f"{len(failures)} job(s) failed during concurrent processing." " Check logs for details.")
        return results

    def _ensure_submitted(self, job_ids: Union[str, List[str]]) -> None:
        """
        Block until all specified jobs have been marked submitted.

        Parameters
        ----------
        job_ids : str or list of str
            One or more job indices expected to reach a SUBMITTED state.
        """
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
        self,
        job_indices: Union[str, List[str]],
        job_queue_id: str,
        batch_size: int = 10,
    ) -> List[str]:
        """
        Submit one or more jobs in batches.

        Parameters
        ----------
        job_indices : str or list of str
            Job indices to submit.
        job_queue_id : str
            Queue identifier for submission.
        batch_size : int, optional
            Maximum number of jobs per batch. Default is 10.

        Returns
        -------
        list of str
            Trace identifiers for each submitted job.

        Raises
        ------
        Exception
            Propagates first error if any job in a batch fails.
        """
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

    def fetch_job_result(
        self,
        job_ids: Union[str, List[str]],
        timeout: float = 100,
        max_retries: Optional[int] = None,
        retry_delay: float = 1,
        verbose: bool = False,
        completion_callback: Optional[Callable[[Dict, str], None]] = None,
        return_failures: bool = False,
    ) -> Union[List[Tuple[Optional[Dict], str]], Tuple[List[Tuple[Optional[Dict], str]], List[Tuple[str, str]]]]:
        """
        Fetches job results for multiple job IDs concurrently with individual timeouts and retry logic.

        Args:
            job_ids (Union[str, List[str]]): A job ID or list of job IDs to fetch results for.
            timeout (float): Timeout for each fetch operation, in seconds.
            max_retries (Optional[int]): Maximum number of retries for jobs that are not ready yet.
            retry_delay (float): Delay between retry attempts, in seconds.
            verbose (bool): If True, logs additional information.
            completion_callback (Optional[Callable[[Dict, str], None]]): A callback function that is executed each time
             a job result is successfully fetched. It receives two arguments: the job result (a dict) and the job ID.
            return_failures (bool): If True, returns a separate list of failed jobs.

        Returns:
          - If `return_failures=False`: List[Tuple[Optional[Dict], str]]
            - A list of tuples, each containing the job result (or None on failure) and the job ID.
          - If `return_failures=True`: Tuple[List[Tuple[Optional[Dict], str]], List[Tuple[str, str]]]
            - A tuple of:
              - List of successful job results.
              - List of failures containing job ID and error message.

        Raises:
            ValueError: If there is an error in decoding the job result.
            TimeoutError: If the fetch operation times out.
            Exception: For all other unexpected issues.
        """

        if isinstance(job_ids, str):
            job_ids = [job_ids]

        results = []
        failures = []

        def fetch_with_retries(job_id: str):
            retries = 0
            while (max_retries is None) or (retries < max_retries):
                try:
                    # Attempt to fetch the job result
                    result = self._fetch_job_result(job_id, timeout, data_only=False)
                    return result, job_id
                except TimeoutError:
                    if verbose:
                        logger.info(
                            f"Job {job_id} is not ready. Retrying {retries + 1}/{max_retries if max_retries else '∞'} "
                            f"after {retry_delay} seconds."
                        )
                    retries += 1
                    time.sleep(retry_delay)  # Wait before retrying
                except (RuntimeError, Exception) as err:
                    logger.error(f"Error while fetching result for job ID {job_id}: {err}")
                    return None, job_id
            logger.error(f"Max retries exceeded for job {job_id}.")
            return None, job_id

        # Use ThreadPoolExecutor to fetch results concurrently
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_with_retries, job_id): job_id for job_id in job_ids}

            # Collect results as futures complete
            for future in as_completed(futures):
                job_id = futures[future]
                try:
                    result, _ = handle_future_result(future, timeout=timeout)

                    # Append a tuple of (result data, job_id). (Using result.get("data") if result is valid.)
                    results.append(result.get("data"))
                    # Run the callback if provided and the result is valid
                    if completion_callback and result:
                        completion_callback(result, job_id)
                except concurrent.futures.TimeoutError as e:
                    error_msg = (
                        f"Timeout while fetching result for job ID {job_id}: "
                        f"{self._job_index_to_job_spec[job_id].source_id}"
                    )
                    logger.error(error_msg)
                    failures.append((self._job_index_to_job_spec[job_id].source_id, str(e)))
                except json.JSONDecodeError as e:
                    error_msg = (
                        f"Decoding error while processing job ID {job_id}: "
                        f"{self._job_index_to_job_spec[job_id].source_id}\n{e}"
                    )
                    logger.error(error_msg)
                    failures.append((self._job_index_to_job_spec[job_id].source_id, str(e)))
                except RuntimeError as e:
                    error_msg = (
                        f"Error while processing job ID {job_id}: "
                        f"{self._job_index_to_job_spec[job_id].source_id}\n{e}"
                    )
                    logger.error(error_msg)
                    failures.append((self._job_index_to_job_spec[job_id].source_id, str(e)))
                except IngestJobFailure as e:
                    error_msg = (
                        f"Error while processing job ID {job_id}: "
                        f"{self._job_index_to_job_spec[job_id].source_id}\n{e.description}"
                    )
                    logger.error(error_msg)
                    failures.append((self._job_index_to_job_spec[job_id].source_id, e.annotations))
                except Exception as e:
                    error_msg = (
                        f"Error while fetching result for job ID {job_id}: "
                        f"{self._job_index_to_job_spec[job_id].source_id}\n{e}"
                    )
                    logger.error(error_msg)
                    failures.append((self._job_index_to_job_spec[job_id].source_id, str(e)))
                finally:
                    # Clean up the job spec mapping
                    del self._job_index_to_job_spec[job_id]

        if return_failures:
            return results, failures

        return results

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
