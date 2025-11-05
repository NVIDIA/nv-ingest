# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=broad-except

import concurrent
import json
import logging
import math
import os
import time
import threading
import copy
from statistics import mean, median
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
from nv_ingest_client.util.util import (
    create_job_specs_for_batch,
    check_ingest_result,
    apply_pdf_split_config_to_job_specs,
)

logger = logging.getLogger(__name__)


def _compute_resident_times(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute resident_time entries from entry/exit pairs if not already present.

    This ensures consistency between split jobs (where server computes resident_time)
    and non-split jobs (where we compute it client-side).

    Parameters
    ----------
    trace_dict : Dict[str, Any]
        Trace dictionary with entry/exit pairs

    Returns
    -------
    Dict[str, Any]
        Trace dictionary with resident_time entries added
    """
    if not trace_dict or not isinstance(trace_dict, dict):
        return trace_dict

    # Check if resident_time already exists (server-computed for split jobs)
    has_resident = any(k.startswith("trace::resident_time::") for k in trace_dict.keys())
    if has_resident:
        return trace_dict  # Already computed by server

    # Compute resident_time from entry/exit pairs
    result = dict(trace_dict)
    stages = set()

    # Find all unique stages
    for key in trace_dict:
        if key.startswith("trace::entry::"):
            stages.add(key.replace("trace::entry::", ""))

    # Compute resident_time for each stage
    for stage in stages:
        entry_key = f"trace::entry::{stage}"
        exit_key = f"trace::exit::{stage}"
        if entry_key in trace_dict and exit_key in trace_dict:
            result[f"trace::resident_time::{stage}"] = trace_dict[exit_key] - trace_dict[entry_key]

    return result


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
    Manages asynchronous submission and result fetching while keeping a steady
    pool of up to `batch_size` in-flight jobs:
    - Retries (202/TimeoutError) are re-queued immediately.
    - New jobs are submitted as capacity frees up.
    - Fetches are started for jobs added each cycle.
    - We always attempt to keep the executor saturated up to `batch_size`.
    """

    def __init__(
        self,
        client: "NvIngestClient",
        job_indices: List[str],
        job_queue_id: Optional[str],
        batch_size: int,
        timeout: Tuple[int, Union[float, None]],
        max_job_retries: Optional[int],
        retry_delay: float,
        initial_fetch_delay: float,
        completion_callback: Optional[Callable[[Dict[str, Any], str], None]],
        fail_on_submit_error: bool,
        stream_to_callback_only: bool,
        return_full_response: bool,
        verbose: bool = False,
        return_traces: bool = False,
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
        return_traces : bool, optional
            If True, parent-level trace data for each completed job is stored.

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
        self.retry_delay = retry_delay
        self.initial_fetch_delay = initial_fetch_delay
        self.completion_callback = completion_callback
        self.fail_on_submit_error = fail_on_submit_error
        self.stream_to_callback_only = stream_to_callback_only
        self.return_full_response = return_full_response
        self.verbose = verbose
        self.return_traces = return_traces

        # State variables managed across batch cycles
        self.retry_job_ids: List[str] = []
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.results: List[Dict[str, Any]] = []  # Stores successful results (full dicts)
        self.failures: List[Tuple[str, str]] = []  # (job_index, error_message)
        self.traces: List[Optional[Dict[str, Any]]] = []

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

        if trace_id:
            self.client.register_parent_trace_id(trace_id)

        if is_failed:
            failed_job_spec = self.client._job_index_to_job_spec.get(job_index)
            self.failures.append((f"{job_index}:{failed_job_spec.source_id}", description))
        elif self.stream_to_callback_only:
            self.results.append(job_index)
        else:
            # When requested, return the full response envelope (includes 'trace' and 'annotations')
            self.results.append(result_data if self.return_full_response else result_data.get("data"))

        # Extract trace data for all successful (non-failed) jobs
        if self.return_traces and not is_failed:
            trace_payload = result_data.get("trace") if result_data else None
            # Compute resident_time if not already present (for consistency)
            if trace_payload:
                trace_payload = _compute_resident_times(trace_payload)
            self.traces.append(trace_payload if trace_payload else None)

        # Cleanup retry count if it exists
        if job_index in self.retry_counts:
            del self.retry_counts[job_index]

        # Execute completion callback if provided
        if self.completion_callback:
            try:
                self.completion_callback(result_data.get("data"), job_index)
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

    # --------------------------------------------------------------------------
    # Declarative Helper Methods (behavior preserved)
    # --------------------------------------------------------------------------

    def _collect_retry_jobs_for_batch(self) -> List[str]:
        """
        Collect retry jobs for this batch, mirroring handler behavior (no pacing filter).

        Returns
        -------
        List[str]
            The list of job indices that should be retried in this batch.
        """
        if not self.retry_job_ids:
            return []

        # Take all retries this cycle and clear the list (handler resets per-iteration)
        eligible: List[str] = list(self.retry_job_ids)
        self.retry_job_ids = []
        if eligible and self.verbose:
            logger.debug(f"Adding {len(eligible)} retry jobs to current batch.")
        return eligible

    def _schedule_retry(self, job_index: str) -> None:
        """
        Schedule an immediate retry for a job (no pacing), mirroring handler behavior.
        """
        if job_index not in self.retry_job_ids:
            self.retry_job_ids.append(job_index)

    def _select_new_jobs_for_batch(
        self,
        submitted_new_indices_count: int,
        total_jobs: int,
        already_in_batch: int,
    ) -> Tuple[List[str], int]:
        """
        Determine the slice of new jobs to include in the current batch based on
        remaining capacity and unsubmitted jobs.

        Note: This does NOT change submitted_new_indices_count. The original code
        increments that counter only after submission is attempted/handled.
        """
        if (already_in_batch < self.batch_size) and (submitted_new_indices_count < total_jobs):
            num_new_to_add = min(self.batch_size - already_in_batch, total_jobs - submitted_new_indices_count)
            start_idx = submitted_new_indices_count
            end_idx = submitted_new_indices_count + num_new_to_add
            new_job_indices = self.all_job_indices_list[start_idx:end_idx]

            if self.verbose:
                logger.debug(f"Adding {len(new_job_indices)} new jobs to current batch.")

            return new_job_indices, submitted_new_indices_count

        return [], submitted_new_indices_count

    def _submit_new_jobs_async(
        self,
        current_batch_new_job_indices: List[str],
        current_batch_job_indices: List[str],
        submitted_new_indices_count: int,
    ) -> Tuple[List[str], int]:
        """
        Initiate asynchronous submission for the new jobs selected for this batch.

        Mirrors the original inline submission block, including error handling and
        fail_on_submit_error semantics. Returns potentially updated batch indices and
        submitted count.
        """
        if not current_batch_new_job_indices:
            return current_batch_job_indices, submitted_new_indices_count

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
            return current_batch_job_indices, submitted_new_indices_count

        try:
            # Fire-and-forget submission initiation
            _ = self.client.submit_job_async(current_batch_new_job_indices, self.job_queue_id)
            # Add successfully initiated jobs to the overall batch list
            current_batch_job_indices.extend(current_batch_new_job_indices)
            # Update count of total initiated jobs
            submitted_new_indices_count += len(current_batch_new_job_indices)
            return current_batch_job_indices, submitted_new_indices_count
        except Exception as e:
            error_msg = (
                f"Batch async submission initiation failed for {len(current_batch_new_job_indices)} new jobs: {e}"
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
            return current_batch_job_indices, submitted_new_indices_count

    def _initiate_fetch_for_batch(self, current_batch_job_indices: List[str]) -> Tuple[Dict[Future, str], List[str]]:
        """
        Initiate fetching for the prepared batch and ensure consistency of returned futures.

        Returns
        -------
        batch_futures_dict : Dict[Future, str]
            Mapping of futures to their associated job indices.
        normalized_job_indices : List[str]
            The job indices normalized to those actually returned by the client if a discrepancy occurs.
        """
        if self.verbose:
            logger.debug(f"Calling fetch_job_result_async for {len(current_batch_job_indices)} jobs.")
        batch_futures_dict: Dict[Future, str] = (
            self.client.fetch_job_result_async(current_batch_job_indices, data_only=False, timeout=None)
            if current_batch_job_indices
            else {}
        )

        # Check for discrepancies where client might not return all futures
        if current_batch_job_indices and (len(batch_futures_dict) != len(current_batch_job_indices)):
            returned_indices = set(batch_futures_dict.values())
            missing_indices = [idx for idx in current_batch_job_indices if idx not in returned_indices]
            logger.error(
                f"fetch_job_result_async discrepancy: Expected {len(current_batch_job_indices)}, got "
                f"{len(batch_futures_dict)}. Missing: {missing_indices}"
            )
            # Fail the missing ones explicitly
            for missing_idx in missing_indices:
                self._handle_processing_failure(
                    missing_idx, "Future not returned by fetch_job_result_async", is_submission_failure=True
                )
            if self.fail_on_submit_error:
                raise RuntimeError("fetch_job_result_async failed to return all expected futures.")
            # Continue processing only the futures we received
            normalized_job_indices = list(returned_indices)
        else:
            normalized_job_indices = list(current_batch_job_indices)

        return batch_futures_dict, normalized_job_indices

    def run(self) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], List[Optional[Dict[str, Any]]]]:
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
        submitted_new_indices_count = 0  # Tracks indices for which submission has been initiated at least once

        logger.debug(f"Starting batch processing for {total_jobs} jobs with batch size {self.batch_size}.")

        # Keep up to batch_size jobs in-flight at all times
        inflight_futures: Dict[Future, str] = {}

        while (submitted_new_indices_count < total_jobs) or self.retry_job_ids or inflight_futures:
            # 1) Top up from retries first
            capacity = max(0, self.batch_size - len(inflight_futures))
            to_fetch: List[str] = []
            if capacity > 0 and self.retry_job_ids:
                take = min(capacity, len(self.retry_job_ids))
                retry_now = self.retry_job_ids[:take]
                self.retry_job_ids = self.retry_job_ids[take:]
                to_fetch.extend(retry_now)
                capacity -= len(retry_now)

            # 2) Then add new jobs up to capacity
            if capacity > 0 and (submitted_new_indices_count < total_jobs):
                new_count = min(capacity, total_jobs - submitted_new_indices_count)
                new_job_indices = self.all_job_indices_list[
                    submitted_new_indices_count : submitted_new_indices_count + new_count
                ]

                if not self.job_queue_id:
                    error_msg = "Cannot submit new jobs: job_queue_id is not set."
                    logger.error(error_msg)
                    for job_index in new_job_indices:
                        self._handle_processing_failure(job_index, error_msg, is_submission_failure=True)
                    submitted_new_indices_count += len(new_job_indices)
                    if self.fail_on_submit_error:
                        raise ValueError(error_msg)
                else:
                    try:
                        _ = self.client.submit_job_async(new_job_indices, self.job_queue_id)
                        submitted_new_indices_count += len(new_job_indices)
                        to_fetch.extend(new_job_indices)
                    except Exception as e:
                        error_msg = f"Batch async submission initiation failed for {len(new_job_indices)} new jobs: {e}"
                        logger.error(error_msg, exc_info=True)
                        for job_index in new_job_indices:
                            self._handle_processing_failure(
                                job_index, f"Batch submission initiation error: {e}", is_submission_failure=True
                            )
                        submitted_new_indices_count += len(new_job_indices)
                        if self.fail_on_submit_error:
                            raise RuntimeError(error_msg) from e

            # 3) Launch fetches for the jobs we added to this cycle
            if to_fetch:
                try:
                    new_futures = self.client.fetch_job_result_async(to_fetch, data_only=False, timeout=None)
                    inflight_futures.update(new_futures)
                except Exception as fetch_init_err:
                    logger.error(
                        f"fetch_job_result_async failed to start for {len(to_fetch)} jobs: {fetch_init_err}",
                        exc_info=True,
                    )
                    for job_index in to_fetch:
                        self._handle_processing_failure(
                            job_index, f"Fetch initiation error: {fetch_init_err}", is_submission_failure=True
                        )
                    if self.fail_on_submit_error:
                        raise RuntimeError(
                            f"Stopping due to fetch initiation failure: {fetch_init_err}"
                        ) from fetch_init_err

            # 4) If nothing left anywhere, exit
            if not inflight_futures and not self.retry_job_ids and submitted_new_indices_count >= total_jobs:
                logger.debug("Exiting loop: No in-flight jobs, no retries, and all jobs submitted.")
                break

            # 5) Wait for at least one in-flight future to complete, then process done ones
            if inflight_futures:
                done, _ = concurrent.futures.wait(
                    set(inflight_futures.keys()), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    job_index = inflight_futures.pop(future, None)
                    if job_index is None:
                        continue
                    try:
                        result_list = future.result()
                        if not isinstance(result_list, list) or len(result_list) != 1:
                            raise ValueError(f"Expected list length 1, got {len(result_list)}")
                        result_tuple = result_list[0]
                        if not isinstance(result_tuple, (tuple, list)) or len(result_tuple) != 3:
                            raise ValueError(f"Expected tuple/list length 3, got {len(result_tuple)}")
                        full_response_dict, fetched_job_index, trace_id = result_tuple
                        if fetched_job_index != job_index:
                            logger.warning(f"Mismatch: Future for {job_index} returned {fetched_job_index}")
                        self._handle_processing_success(job_index, full_response_dict, trace_id)
                    except TimeoutError:
                        # Not ready -> immediate retry
                        self.retry_counts[job_index] += 1
                        if self.max_job_retries is None or self.retry_counts[job_index] <= self.max_job_retries:
                            if self.verbose:
                                logger.info(
                                    f"Job {job_index} not ready, scheduling retry "
                                    f"(Attempt {self.retry_counts[job_index]}/{self.max_job_retries or 'inf'})."
                                )
                            self._schedule_retry(job_index)
                        else:
                            error_msg = f"Exceeded max fetch retries ({self.max_job_retries}) for job {job_index}."
                            logger.error(error_msg)
                            self._handle_processing_failure(job_index, error_msg)
                    except (ValueError, RuntimeError) as e:
                        logger.error(f"Job {job_index} failed processing result: {e}", exc_info=self.verbose)
                        self._handle_processing_failure(job_index, f"Error processing result: {e}")
                    except Exception as e:
                        logger.exception(f"Unhandled error processing future for job {job_index}: {e}")
                        self._handle_processing_failure(job_index, f"Unhandled error processing future: {e}")

        # --- Final Logging ---
        self._log_final_status(total_jobs)

        return self.results, self.failures, self.traces if self.return_traces else []


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
            Extra keyword arguments passed to the client allocator. For RestClient,
            can include 'api_version' (e.g., 'v1' or 'v2'). Defaults to 'v1'.
        msg_counter_id : str, optional
            Identifier for message counting. Defaults to "nv-ingest-message-id".
        worker_pool_size : int, optional
            Number of workers in the thread pool. Defaults to 8.

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

        # Initialize the worker pool with the specified size (used for both submit and fetch)
        self._worker_pool = ThreadPoolExecutor(max_workers=worker_pool_size)

        # Telemetry state and controls
        self._telemetry_lock = threading.Lock()
        self._telemetry_enabled: bool = bool(int(os.getenv("NV_INGEST_CLIENT_TELEMETRY", "1")))
        try:
            self._telemetry_max_calls: int = int(os.getenv("NV_INGEST_CLIENT_TELEMETRY_MAX_CALLS", "10000"))
        except ValueError:
            self._telemetry_max_calls = 10000
        self._telemetry = {}
        self._completed_parent_trace_ids: List[str] = []  # 1054
        self.reset_telemetry()

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

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def enable_telemetry(self, enabled: bool) -> None:
        with self._telemetry_lock:
            self._telemetry_enabled = bool(enabled)

    def reset_telemetry(self) -> None:
        with self._telemetry_lock:
            self._telemetry = {
                "started_at": time.time(),
                "submit": {"count": 0, "calls": []},
                "fetch": {"count": 0, "last_ts": None, "intervals": [], "calls": []},
                "per_job": {},
            }

    def _t_per_job(self, job_index: str) -> Dict[str, Any]:
        pj = self._telemetry["per_job"].get(job_index)
        if pj is None:
            pj = {"submits": [], "fetch_attempts": [], "timeouts_202": 0, "failures": 0, "first_success_ts": None}
            self._telemetry["per_job"][job_index] = pj
        return pj

    def _t_append_capped(self, arr: List[Any], item: Any) -> None:
        if len(arr) < self._telemetry_max_calls:
            arr.append(item)

    def _t_record_submit(self, job_index: str, status: str, ts: float, trace_id: Optional[str]) -> None:
        if not self._telemetry_enabled:
            return
        with self._telemetry_lock:
            self._telemetry["submit"]["count"] += 1
            self._t_append_capped(
                self._telemetry["submit"]["calls"],
                {"job": job_index, "status": status, "ts": ts, "trace": trace_id},
            )
            pj = self._t_per_job(job_index)
            self._t_append_capped(pj["submits"], ts)

    def _t_record_fetch_attempt(self, job_index: str, ts: float) -> None:
        if not self._telemetry_enabled:
            return
        with self._telemetry_lock:
            self._telemetry["fetch"]["count"] += 1
            last = self._telemetry["fetch"]["last_ts"]
            if last is not None:
                delta = ts - float(last)
                if delta >= 0:
                    self._t_append_capped(self._telemetry["fetch"]["intervals"], delta)
            self._telemetry["fetch"]["last_ts"] = ts
            pj = self._t_per_job(job_index)
            self._t_append_capped(pj["fetch_attempts"], ts)

    def _t_record_fetch_outcome(self, job_index: str, code: int, ts: float, ok: bool, trace_id: Optional[str]) -> None:
        if not self._telemetry_enabled:
            return
        with self._telemetry_lock:
            self._t_append_capped(
                self._telemetry["fetch"]["calls"],
                {"job": job_index, "code": code, "ok": ok, "ts": ts, "trace": trace_id},
            )
            pj = self._t_per_job(job_index)
            if code == 2:  # 202 not ready
                pj["timeouts_202"] += 1
            if ok and pj["first_success_ts"] is None:
                pj["first_success_ts"] = ts
            if not ok and code not in (0, 2):
                pj["failures"] += 1

    def get_telemetry(self) -> Dict[str, Any]:
        with self._telemetry_lock:
            return copy.deepcopy(self._telemetry)

    def summarize_telemetry(self) -> Dict[str, Any]:
        with self._telemetry_lock:
            submit_count = self._telemetry["submit"]["count"]
            fetch_count = self._telemetry["fetch"]["count"]
            intervals = list(self._telemetry["fetch"]["intervals"])
            intervals.sort()
            avg = mean(intervals) if intervals else 0.0
            p50 = median(intervals) if intervals else 0.0
            # p95 via index
            p95 = intervals[int(0.95 * (len(intervals) - 1))] if intervals else 0.0
            per_job = self._telemetry["per_job"]
            # Aggregate per-job stats
            jobs = len(per_job)
            total_timeouts = sum(pj.get("timeouts_202", 0) for pj in per_job.values())
            total_failures = sum(pj.get("failures", 0) for pj in per_job.values())
            return {
                "submit_count": submit_count,
                "fetch_count": fetch_count,
                "fetch_interval_avg": avg,
                "fetch_interval_p50": p50,
                "fetch_interval_p95": p95,
                "jobs_tracked": jobs,
                "timeouts_202_total": total_timeouts,
                "failures_total": total_failures,
            }

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
        ts_attempt = time.time()
        self._t_record_fetch_attempt(job_index, ts_attempt)
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
                    self._t_record_fetch_outcome(job_index, 0, time.time(), ok=True, trace_id=job_state.trace_id)
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

            elif response.response_code == 2:  # Job Not Ready (e.g., HTTP 202, or r-2 from SimpleBroker)
                # Raise TimeoutError to signal the calling retry loop in fetch_job_result
                # Do not change job state here, remains SUBMITTED
                self._t_record_fetch_outcome(job_index, 2, time.time(), ok=False, trace_id=job_state.trace_id)
                raise TimeoutError(f"Job not ready: {response.response_reason}")

            else:
                # Log the failure reason from the ResponseSchema
                error_msg = (
                    f"Terminal failure fetching result for client index {job_index} (Server ID: {server_job_id}). "
                    f"Code: {response.response_code}, Reason: {response.response_reason}"
                )
                logger.error(error_msg)
                job_state.state = JobStateEnum.FAILED  # Mark job as failed in the client
                # Do NOT pop the state for failed jobs here
                # Raise RuntimeError to indicate a terminal failure for this fetch attempt
                self._t_record_fetch_outcome(job_index, 1, time.time(), ok=False, trace_id=job_state.trace_id)
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
            try:
                self._t_record_fetch_outcome(job_index, 1, time.time(), ok=False, trace_id=None)
            except Exception:
                pass
            raise  # Re-raise the original exception

    def fetch_job_result_cli(
        self,
        job_ids: Union[str, List[str]],
        data_only: bool = False,
        timeout: Optional[Tuple[int, Optional[float]]] = None,
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

        eff_timeout: Tuple[int, Optional[float]] = timeout if timeout is not None else (100, None)
        return [self._fetch_job_result(job_id, timeout=eff_timeout, data_only=data_only) for job_id in job_ids]

    def _validate_batch_size(self, batch_size: Optional[int]) -> int:
        """
        Validates and returns a sanitized batch_size value.

        Parameters
        ----------
        batch_size : Optional[int]
            The batch_size value to validate. None uses value from
            NV_INGEST_BATCH_SIZE environment variable or default 32.

        Returns
        -------
        int
            Validated batch_size value.
        """
        # Handle None/default case
        if batch_size is None:
            try:
                batch_size = int(os.getenv("NV_INGEST_CLIENT_BATCH_SIZE", "32"))
            except ValueError:
                batch_size = 32

        # Validate type and range
        if not isinstance(batch_size, int):
            logger.warning(f"batch_size must be an integer, got {type(batch_size).__name__}. Using default 32.")
            return 32

        if batch_size < 1:
            logger.warning(f"batch_size must be >= 1, got {batch_size}. Using default 32.")
            return 32

        # Performance guidance warnings
        if batch_size < 8:
            logger.warning(f"batch_size {batch_size} is very small and may impact performance.")
        elif batch_size > 128:
            logger.warning(f"batch_size {batch_size} is large and may increase memory usage.")

        return batch_size

    def process_jobs_concurrently(
        self,
        job_indices: Union[str, List[str]],
        job_queue_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        concurrency_limit: int = 64,
        timeout: int = 100,
        max_job_retries: Optional[int] = None,
        retry_delay: float = 0.5,
        initial_fetch_delay: float = 0.3,
        fail_on_submit_error: bool = False,
        completion_callback: Optional[Callable[[Any, str], None]] = None,
        return_failures: bool = False,
        data_only: bool = True,
        stream_to_callback_only: bool = False,
        return_full_response: bool = False,
        verbose: bool = False,
        return_traces: bool = False,
    ) -> Union[
        List[Any],
        Tuple[List[Any], List[Tuple[str, str]]],
        Tuple[List[Any], List[Tuple[str, str]], List[Optional[Dict[str, Any]]]],
    ]:
        """
        Submit and fetch multiple jobs concurrently.

        Parameters
        ----------
        job_indices : str or list of str
            Single or multiple job indices to process.
        job_queue_id : str, optional
            Queue identifier for submission.
        batch_size : int, optional
            Maximum number of jobs to process in each internal batch.
            Higher values may improve throughput but increase memory usage.
            Must be >= 1. Default is 32.
        concurrency_limit : int, optional
            Max number of simultaneous in-flight jobs. Default is 64.
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
        return_full_response : bool, optional
            If True, results contain the full response envelopes (including 'trace' and 'annotations').
            Ignored when stream_to_callback_only=True. Default is False.
        verbose : bool, optional
            If True, enable debug logging. Default is False.
        return_traces : bool, optional
            If True, parent-level aggregated trace metrics are extracted and returned. Default is False.

        Returns
        -------
        results : list
            List of successful job results when `return_failures` is False.
        results, failures : tuple
            Tuple of (successful results, failure tuples) when `return_failures` is True.
        results, failures, traces : tuple
            Tuple of (successful results, failure tuples, trace dicts) when both
            `return_failures` and `return_traces` are True.

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
            if return_failures and return_traces:
                return [], [], []
            elif return_failures:
                return [], []
            else:
                return []

        # Validate and set batch_size
        validated_batch_size = self._validate_batch_size(batch_size)

        # Prepare timeout tuple to mirror handler behavior: finite connect, unbounded read (long-poll)
        effective_timeout: Tuple[int, Optional[float]] = (int(timeout), None)

        # Delegate to the concurrent processor
        processor = _ConcurrentProcessor(
            client=self,
            batch_size=validated_batch_size,
            job_indices=job_indices,
            job_queue_id=job_queue_id,
            timeout=effective_timeout,
            max_job_retries=max_job_retries,
            retry_delay=retry_delay,
            initial_fetch_delay=initial_fetch_delay,
            completion_callback=completion_callback,
            fail_on_submit_error=fail_on_submit_error,
            stream_to_callback_only=stream_to_callback_only,
            return_full_response=return_full_response,
            verbose=verbose,
            return_traces=return_traces,
        )

        results, failures, traces = processor.run()

        if return_failures and return_traces:
            return results, failures, traces
        elif return_failures:
            return results, failures
        elif return_traces:
            return results, traces

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

    def fetch_job_result_async(
        self,
        job_ids: Union[str, List[str]],
        data_only: bool = True,
        timeout: Optional[Tuple[int, Optional[float]]] = None,
    ) -> Dict[Future, str]:
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
            future = self._worker_pool.submit(self.fetch_job_result_cli, job_id, data_only, timeout)
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

            try:
                self._t_record_submit(job_index, "ok", time.time(), x_trace_id)
            except Exception:
                pass
            return x_trace_id
        except Exception as err:
            err_msg = f"Failed to submit job {job_index} to queue {job_queue_id}: {err}"
            logger.exception(err_msg)
            job_state.state = JobStateEnum.FAILED
            try:
                self._t_record_submit(job_index, "fail", time.time(), None)
            except Exception:
                pass
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

        if return_failures:
            return results, failures

        return results

    def create_jobs_for_batch(
        self, files_batch: List[str], tasks: Dict[str, Any], pdf_split_page_count: int = None
    ) -> List[str]:
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
        pdf_split_page_count : int, optional
            Number of pages per PDF chunk for splitting (1-128). If provided, this will be added
            to the job spec's extended_options for PDF files.

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

        # Apply PDF split config if provided
        if pdf_split_page_count is not None:
            apply_pdf_split_config_to_job_specs(job_specs, pdf_split_page_count)

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

    def register_parent_trace_id(self, trace_id: Optional[str]) -> None:
        """Record a parent trace identifier once its aggregation completed."""

        if not trace_id:
            return

        if trace_id not in self._completed_parent_trace_ids:
            self._completed_parent_trace_ids.append(trace_id)

    def consume_completed_parent_trace_ids(self) -> List[str]:
        """Return and clear the set of completed parent trace identifiers."""

        trace_ids = list(self._completed_parent_trace_ids)
        self._completed_parent_trace_ids.clear()
        return trace_ids
