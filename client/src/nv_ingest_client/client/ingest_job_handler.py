# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import time
import os
import io
import base64
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# Reuse existing CLI utilities to avoid duplicating behavior
from concurrent.futures import as_completed
from nv_ingest_client.util.util import check_ingest_result
from PIL import Image

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
        pdf_split_page_count: int = None,
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
        self.pdf_split_page_count = pdf_split_page_count
        self._pbar = None
        # Internal state used across iterations
        self._retry_job_ids: List[str] = []
        self._processed: int = 0
        self._job_ids_batch: List[str] = []
        self._job_id_map: Dict[str, str] = {}
        self._trace_times: Dict[str, List[float]] = defaultdict(list)
        # Constants
        self._IMAGE_TYPES: set = {"png", "bmp", "jpeg", "jpg", "tiff"}

    # ---------------------------
    # Progress bar helpers
    # ---------------------------
    def _init_progress_bar(self, total: int) -> None:
        if self.show_progress:
            self._pbar = tqdm(total=total, desc="Processing files", unit="file")
        else:
            self._pbar = None

    def _update_progress(self, n: int = 1, pages_per_sec: float | None = None) -> None:
        if not self._pbar:
            return
        if pages_per_sec is not None:
            self._pbar.set_postfix(pages_per_sec=f"{pages_per_sec:.2f}")
        self._pbar.update(n)

    def _close_progress_bar(self) -> None:
        if self._pbar:
            self._pbar.close()
            self._pbar = None

    def _generate_job_batch_for_iteration(self) -> None:
        """
        Build the next batch of jobs for processing and submit newly created jobs.

        This method mirrors the CLI batching semantics: it prioritizes retry jobs,
        then creates new jobs up to the given ``batch_size``, submits those new jobs
        asynchronously to the configured queue, and returns the combined list of
        job indices for this iteration. It also updates the internal progress bar
        when configured and advances the processed-file counter.

        Side Effects
        ------------
        - Populates/overwrites ``self._job_ids_batch`` with the ordered job indices to
          process this iteration (``retry`` first, then newly created jobs).
        - Updates ``self._job_id_map`` with any new mappings from job index to source file path
          for jobs created in this iteration.

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
        >>> handler._generate_job_batch_for_iteration()
        >>> len(handler._job_ids_batch) <= 32
        True
        """
        job_indices: List[str] = []
        job_index_map_updates: Dict[str, str] = {}
        cur_job_count: int = 0

        if self._retry_job_ids:
            job_indices.extend(self._retry_job_ids)
            cur_job_count = len(job_indices)

        if (cur_job_count < self.batch_size) and (self._processed < len(self.files)):
            new_job_count: int = min(self.batch_size - cur_job_count, len(self.files) - self._processed)
            batch_files: List[str] = self.files[self._processed : self._processed + new_job_count]

            new_job_indices: List[str] = self.client.create_jobs_for_batch(
                batch_files, self.tasks, pdf_split_page_count=self.pdf_split_page_count
            )
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
            self._processed += new_job_count
            # Submit newly created jobs asynchronously to the configured queue
            _ = self.client.submit_job_async(new_job_indices, self.job_queue_id)
            job_indices.extend(new_job_indices)

        # Save into class state
        self._job_ids_batch = job_indices
        # Merge new mappings (do not drop existing entries for retry jobs)
        self._job_id_map.update(job_index_map_updates)

    def _handle_future_result(self, future, timeout: int = 10):
        """
        Handle the result of a completed future job and process annotations.

        Parameters
        ----------
        future : concurrent.futures.Future
            Future representing an asynchronous job.
        timeout : int, optional
            Maximum seconds to wait for the future result.

        Returns
        -------
        Tuple[Dict[str, Any], str]
            The decoded result dictionary and the trace_id for the job.

        Raises
        ------
        RuntimeError
            If the job result indicates failure per check_ingest_result.
        """
        result, _, trace_id = future.result(timeout=timeout)[0]
        if ("annotations" in result) and result["annotations"]:
            annotations = result["annotations"]
            for key, value in annotations.items():
                logger.debug(f"Annotation: {key} -> {json.dumps(value, indent=2)}")

        failed, description = check_ingest_result(result)
        if failed:
            raise RuntimeError(f"Ingest job failed: {description}")

        return result, trace_id

    def _process_response(self, response: Dict[str, Any]) -> None:
        """
        Extract trace timing entries from a response and accumulate per-stage elapsed times
        into ``self._trace_times``.

        Parameters
        ----------
        response : Dict[str, Any]
            Full response payload containing an optional ``trace`` dictionary with
            entry/exit timestamps.
        """
        trace_data: Dict[str, Any] = response.get("trace", {})
        for key, entry_time in trace_data.items():
            if "entry" in key:
                exit_key: str = key.replace("entry", "exit")
                exit_time: Any = trace_data.get(exit_key)
                if exit_time:
                    stage_parts = key.split("::")
                    if len(stage_parts) >= 3:
                        stage_name: str = stage_parts[2]
                        elapsed_time: int = exit_time - entry_time
                        self._trace_times[stage_name].append(elapsed_time)

    def _save_response_data(
        self, response: Dict[str, Any], output_directory: str, images_to_disk: bool = False
    ) -> None:
        """
        Save the response data into categorized metadata JSON files and optionally save images to disk.

        Parameters
        ----------
        response : Dict[str, Any]
            Full response payload with a "data" list of documents.
        output_directory : str
            Output directory where per-type metadata JSON files (and any media) are written.
        images_to_disk : bool, optional
            If True, decode and write image contents to disk and replace content with a file URL.
        """
        if ("data" not in response) or (not response["data"]):
            logger.debug("Data is not in the response or response.data is empty")
            return

        response_data = response["data"]
        if not isinstance(response_data, list) or len(response_data) == 0:
            logger.debug("Response data is not a list or the list is empty.")
            return

        doc_meta_base = response_data[0]["metadata"]
        source_meta = doc_meta_base["source_metadata"]
        doc_name = source_meta["source_id"]
        clean_doc_name = os.path.basename(doc_name)
        output_name = f"{clean_doc_name}.metadata.json"

        # Organize by document type
        doc_map: Dict[str, List[Dict[str, Any]]] = {}
        for document in response_data:
            meta: Dict[str, Any] = document.get("metadata", {})
            content_meta: Dict[str, Any] = meta.get("content_metadata", {})
            doc_type: str = content_meta.get("type", "unknown")
            doc_map.setdefault(doc_type, []).append(document)

        for doc_type, documents in doc_map.items():
            doc_type_path = os.path.join(output_directory, doc_type)
            os.makedirs(doc_type_path, exist_ok=True)

            if doc_type in ("image", "structured") and images_to_disk:
                for i, doc in enumerate(documents):
                    meta: Dict[str, Any] = doc.get("metadata", {})
                    image_content = meta.get("content")
                    image_type = (
                        meta.get("image_metadata", {}).get("image_type", "png").lower()
                        if doc_type == "image"
                        else "png"
                    )

                    if image_content and image_type in self._IMAGE_TYPES:
                        try:
                            image_data = base64.b64decode(image_content)
                            image = Image.open(io.BytesIO(image_data))

                            image_ext = "jpg" if image_type == "jpeg" else image_type
                            image_filename = f"{clean_doc_name}_{i}.{image_ext}"
                            image_output_path = os.path.join(doc_type_path, "media", image_filename)
                            os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                            image.save(image_output_path, format=image_ext.upper())

                            meta["content"] = ""
                            meta["content_url"] = os.path.realpath(image_output_path)
                            logger.debug(f"Saved image to {image_output_path}")
                        except Exception as e:
                            logger.error(f"Failed to save image {i} for {clean_doc_name}: {e}")

            # Write the metadata JSON file for this type
            with open(os.path.join(doc_type_path, output_name), "w") as f:
                f.write(json.dumps(documents, indent=2))

    def run(self) -> Tuple[int, Dict[str, List[float]], int, Dict[str, str]]:
        total_files: int = len(self.files)
        total_pages_processed: int = 0
        trace_ids: Dict[str, str] = defaultdict(list)  # type: ignore
        failed_jobs: List[str] = []
        retry_counts: Dict[str, int] = defaultdict(int)
        pages_per_sec: float = None

        start_time_ns: int = time.time_ns()
        self._init_progress_bar(total_files)
        pages_per_sec: float = None
        try:
            self._processed = 0
            while (self._processed < len(self.files)) or self._retry_job_ids:
                # Create a batch (retries first, then new jobs up to batch_size)
                self._generate_job_batch_for_iteration()
                job_id_map = self._job_id_map
                self._retry_job_ids = []

                futures_dict: Dict[Any, str] = self.client.fetch_job_result_async(self._job_ids_batch, data_only=False)
                for future in as_completed(futures_dict.keys()):
                    pages_per_sec = None
                    try:
                        # Block as each future completes; this mirrors CLI behavior
                        future_response, trace_id = self._handle_future_result(future)
                        job_id: str = futures_dict[future]
                        trace_ids[job_id_map[job_id]] = trace_id

                        # Extract page count: prefer V2 metadata location, fall back to V1
                        page_count = None
                        source_name = None

                        # Try V2 metadata location first (top-level metadata.total_pages)
                        if "metadata" in future_response and future_response["metadata"]:
                            response_metadata = future_response["metadata"]
                            page_count = response_metadata.get("total_pages")
                            source_name = response_metadata.get("original_source_name")

                        # Fall back to V1 location (first data element's hierarchy.page_count)
                        if page_count is None and future_response.get("data"):
                            try:
                                first_page_metadata = future_response["data"][0]["metadata"]
                                page_count = first_page_metadata["content_metadata"]["hierarchy"]["page_count"]
                                source_name = first_page_metadata["source_metadata"]["source_name"]
                            except (KeyError, IndexError, TypeError):
                                # If we can't extract from V1 location, use defaults
                                pass

                        # Use extracted values or defaults
                        if page_count is None:
                            page_count = 0  # Default if not found
                        if source_name is None:
                            source_name = "unknown_source"

                        file_page_counts: Dict[str, int] = {source_name: page_count}

                        if self.output_directory:
                            self._save_response_data(
                                future_response,
                                self.output_directory,
                                images_to_disk=self.save_images_separately,
                            )

                        total_pages_processed += file_page_counts[list(file_page_counts.keys())[0]]
                        elapsed_time: float = (time.time_ns() - start_time_ns) / 1e9
                        if elapsed_time > 0:
                            pages_per_sec: float = total_pages_processed / elapsed_time
                        else:
                            pages_per_sec = None

                        self._process_response(future_response)

                    except TimeoutError:
                        job_id = futures_dict[future]
                        src_name = job_id_map[job_id]
                        retry_counts[src_name] += 1
                        self._retry_job_ids.append(job_id)
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
                        # Do not update the pbar if this job is going to be retried
                        if futures_dict[future] not in self._retry_job_ids:
                            self._update_progress(1, pages_per_sec)
        finally:
            self._close_progress_bar()

        # Optionally print telemetry summary
        if self.show_telemetry and hasattr(self.client, "summarize_telemetry"):
            try:
                summary = self.client.summarize_telemetry()
                logger.info("NvIngestClient Telemetry Summary: %s", json.dumps(summary, indent=2))
            except Exception:
                pass

        return total_files, self._trace_times, total_pages_processed, trace_ids
