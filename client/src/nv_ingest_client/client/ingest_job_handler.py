# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, NVIDIA

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# Reuse existing CLI utilities to avoid duplicating behavior
from concurrent.futures import as_completed
from nv_ingest_client.cli.util.processing import (
    save_response_data,
    process_response,
)
from nv_ingest_client.util.processing import handle_future_result

logger = logging.getLogger(__name__)


class IngestJobHandler:
    """
    A modular job handler that mirrors the CLI's create_and_process_jobs flow,
    so the same proven scheduling/retry behavior can be reused by other entry points.

    Usage:
        handler = IngestJobHandler(client, files, tasks, output_dir, batch_size)
        total_files, trace_times, total_pages, trace_ids = handler.run()
    """

    def __init__(
        self,
        client: Any,
        files: List[str],
        tasks: Dict[str, Any],
        output_directory: str,
        batch_size: int,
        fail_on_error: bool = False,
        save_images_separately: bool = False,
        show_progress: bool = True,
        show_telemetry: bool = False,
        job_queue_id: str = "ingest_task_queue",
    ) -> None:
        self.client = client
        self.files = files
        self.tasks = tasks
        self.output_directory = output_directory
        self.batch_size = batch_size
        self.fail_on_error = fail_on_error
        self.save_images_separately = save_images_separately
        self.show_progress = show_progress
        self.show_telemetry = show_telemetry
        self.job_queue_id = job_queue_id
        self._pbar = None

    def _generate_job_batch_for_iteration(
        self,
        processed: int,
        batch_size: int,
        retry_job_ids: List[str],
    ) -> Tuple[List[str], Dict[str, str], int]:
        """
        Build the next batch of jobs for processing and submit newly created jobs.

        This method mirrors the CLI batching semantics: it prioritizes retry jobs,
        then creates new jobs up to the given ``batch_size``, submits those new jobs
        asynchronously to the configured queue, and returns the combined list of
        job indices for this iteration. It also updates the internal progress bar
        when configured and advances the processed-file counter.

        Parameters
        ----------
        processed : int
            The number of files already considered in prior iterations. Used to
            compute the next slice of files for which to create jobs.
        batch_size : int
            Maximum number of jobs to include in this iteration. Retry jobs are
            inserted first; any remaining capacity is filled by new jobs.
        retry_job_ids : List[str]
            Job indices to retry due to prior timeouts or transient errors. These
            are inserted at the front of the batch.

        Returns
        -------
        job_indices : List[str]
            Ordered list of job indices to process this iteration. Contains
            ``retry_job_ids`` first, followed by any newly created job indices.
        job_index_map_updates : Dict[str, str]
            Mapping from job index to the source file path used to create the job.
            Only includes entries for newly created jobs from this iteration.
        processed : int
            Updated number of files considered after creating new jobs in this
            iteration.

        Raises
        ------
        RuntimeError
            If one or more job specs cannot be created (e.g., unreadable files)
            and ``self.fail_on_error`` is True.

        Notes
        -----
        - Side effects:
          - Creates JobSpecs via ``self.client.create_jobs_for_batch(...)``.
          - Submits newly created jobs via ``self.client.submit_job_async(..., self.job_queue_id)``.
          - Updates the class-owned progress bar (``self._pbar``) to account for
            missing jobs when some files fail to produce specs and
            ``self.fail_on_error`` is False.
        - This method does not perform fetching; it only prepares and submits
          jobs for the current iteration.
        - The ``processed`` counter advances by the number of files attempted in
          this iteration, even if some job specs are missing (unless
          ``self.fail_on_error`` is True).

        Examples
        --------
        >>> handler = IngestJobHandler(client, files, tasks, "/tmp/out", batch_size=32)
        >>> retry_ids = []
        >>> job_ids, idx_map, processed = handler._generate_job_batch_for_iteration(
        ...     processed=0, batch_size=32, retry_job_ids=retry_ids
        ... )
        >>> len(job_ids) <= 32
        True
        """
        job_indices: List[str] = []
        job_index_map_updates: Dict[str, str] = {}
        cur_job_count: int = 0

        if retry_job_ids:
            job_indices.extend(retry_job_ids)
            cur_job_count = len(job_indices)

        if (cur_job_count < batch_size) and (processed < len(self.files)):
            new_job_count: int = min(batch_size - cur_job_count, len(self.files) - processed)
            batch_files: List[str] = self.files[processed : processed + new_job_count]

            new_job_indices: List[str] = self.client.create_jobs_for_batch(batch_files, self.tasks)
            if len(new_job_indices) != new_job_count:
                missing_jobs: int = new_job_count - len(new_job_indices)
                error_msg: str = (
                    f"Missing {missing_jobs} job specs -- this is likely due to bad reads or file corruption"
                )
                logger.warning(error_msg)

                if self.fail_on_error:
                    raise RuntimeError(error_msg)

                if self._pbar:
                    self._pbar.update(missing_jobs)

            job_index_map_updates = {job_index: file for job_index, file in zip(new_job_indices, batch_files)}
            processed += new_job_count
            # Submit newly created jobs asynchronously to the configured queue
            _ = self.client.submit_job_async(new_job_indices, self.job_queue_id)
            job_indices.extend(new_job_indices)

        return job_indices, job_index_map_updates, processed

    def run(self) -> Tuple[int, Dict[str, List[float]], int, Dict[str, str]]:
        total_files: int = len(self.files)
        total_pages_processed: int = 0
        trace_times: Dict[str, List[float]] = defaultdict(list)
        trace_ids: Dict[str, str] = defaultdict(list)  # type: ignore
        failed_jobs: List[str] = []
        retry_job_ids: List[str] = []
        job_id_map: Dict[str, str] = {}
        retry_counts: Dict[str, int] = defaultdict(int)

        start_time_ns: int = time.time_ns()
        progress_ctx = tqdm(total=total_files, desc="Processing files", unit="file") if self.show_progress else None
        self._pbar = progress_ctx
        try:
            processed: int = 0
            while (processed < len(self.files)) or retry_job_ids:
                # Create a batch (retries first, then new jobs up to batch_size)
                job_ids, job_id_map_updates, processed = self._generate_job_batch_for_iteration(
                    processed,
                    self.batch_size,
                    retry_job_ids,
                )
                job_id_map.update(job_id_map_updates)
                retry_job_ids = []

                futures_dict: Dict[Any, str] = self.client.fetch_job_result_async(job_ids, data_only=False)
                for future in as_completed(futures_dict.keys()):
                    try:
                        # Block as each future completes; this mirrors CLI behavior
                        future_response, trace_id = handle_future_result(future)
                        job_id: str = futures_dict[future]
                        trace_ids[job_id_map[job_id]] = trace_id

                        first_page_metadata = future_response["data"][0]["metadata"]
                        file_page_counts: Dict[str, int] = {
                            first_page_metadata["source_metadata"]["source_name"]: first_page_metadata[
                                "content_metadata"
                            ]["hierarchy"]["page_count"]
                        }

                        if self.output_directory:
                            save_response_data(
                                future_response,
                                self.output_directory,
                                images_to_disk=self.save_images_separately,
                            )

                        total_pages_processed += file_page_counts[list(file_page_counts.keys())[0]]
                        elapsed_time: float = (time.time_ns() - start_time_ns) / 1e9
                        if self._pbar and elapsed_time > 0:
                            pages_per_sec: float = total_pages_processed / elapsed_time
                            self._pbar.set_postfix(pages_per_sec=f"{pages_per_sec:.2f}")

                        process_response(future_response, trace_times)

                    except TimeoutError:
                        job_id = futures_dict[future]
                        src_name = job_id_map[job_id]
                        retry_counts[src_name] += 1
                        retry_job_ids.append(job_id)
                    except json.JSONDecodeError as e:
                        job_id = futures_dict[future]
                        src_name = job_id_map[job_id]
                        logger.error(f"Decoding error while processing {job_id}({src_name}): {e}")
                        failed_jobs.append(f"{job_id}::{src_name}")
                    except RuntimeError as e:
                        job_id = futures_dict[future]
                        src_name = job_id_map[job_id]
                        logger.error(f"Error while processing '{job_id}' - ({src_name}):\n{e}")
                        failed_jobs.append(f"{job_id}::{src_name}")
                    except Exception as e:
                        job_id = futures_dict[future]
                        src_name = job_id_map[job_id]
                        logger.exception(f"Unhandled error while processing {job_id}({src_name}): {e}")
                        failed_jobs.append(f"{job_id}::{src_name}")
                    finally:
                        if self._pbar:
                            # Do not update the pbar if this job is going to be retried
                            if futures_dict[future] not in retry_job_ids:
                                self._pbar.update(1)
        finally:
            if self._pbar:
                self._pbar.close()
                self._pbar = None

        # Optionally print telemetry summary
        if self.show_telemetry and hasattr(self.client, "summarize_telemetry"):
            try:
                summary = self.client.summarize_telemetry()
                logger.info("NvIngestClient Telemetry Summary: %s", json.dumps(summary, indent=2))
            except Exception:
                pass

        return total_files, trace_times, total_pages_processed, trace_ids
