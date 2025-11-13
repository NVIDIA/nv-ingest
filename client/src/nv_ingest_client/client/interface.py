# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import collections
import glob
import gzip
import json
import logging
import os
import shutil
import tempfile
import threading
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from functools import wraps
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from urllib.parse import urlparse

import fsspec
from nv_ingest_api.internal.enums.common import PipelinePhase
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskCaptionSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskDedupSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskEmbedSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskExtractSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskFilterSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskSplitSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskStoreEmbedSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskStoreSchema
from nv_ingest_api.util.introspection.function_inspect import infer_udf_function_name
from nv_ingest_client.client.client import NvIngestClient
from nv_ingest_client.client.util.processing import get_valid_filename
from nv_ingest_client.client.util.processing import save_document_results_to_jsonl
from nv_ingest_client.primitives import BatchJobSpec
from nv_ingest_client.primitives.jobs import JobStateEnum
from nv_ingest_client.primitives.tasks import CaptionTask
from nv_ingest_client.primitives.tasks import DedupTask
from nv_ingest_client.primitives.tasks import EmbedTask
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import FilterTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import StoreTask
from nv_ingest_client.primitives.tasks import StoreEmbedTask
from nv_ingest_client.primitives.tasks import UDFTask
from nv_ingest_client.util.processing import check_schema
from nv_ingest_client.util.system import ensure_directory_with_permissions
from nv_ingest_client.util.util import filter_function_kwargs, apply_pdf_split_config_to_job_specs
from nv_ingest_client.util.vdb import VDB, get_vdb_op_cls
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_JOB_QUEUE_ID = "ingest_task_queue"


def get_max_filename_length(path="."):
    return os.pathconf(path, "PC_NAME_MAX")


def safe_filename(base_dir, filename, suffix=""):
    max_name = os.pathconf(base_dir, "PC_NAME_MAX")
    # Account for suffix (like ".jsonl") in the allowed length
    allowed = max_name - len(suffix)
    # If filename too long, truncate and append suffix
    if len(filename) > allowed:
        filename = filename[:allowed]
    return filename + suffix


def ensure_job_specs(func):
    """Decorator to ensure _job_specs is initialized before calling task methods."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._job_specs is None:
            raise ValueError(
                "Job specifications are not initialized because some files are "
                "remote or not accesible locally. Ensure file paths are correct, "
                "and call `.load()` first if files are remote."
            )
        return func(self, *args, **kwargs)

    return wrapper


class LazyLoadedList(collections.abc.Sequence):
    def __init__(self, filepath: str, expected_len: Optional[int] = None, compression: Optional[str] = None):
        self.filepath = filepath
        self._len: Optional[int] = expected_len  # Store pre-calculated length
        self._offsets: Optional[List[int]] = None
        self.compression = compression

        if self._len == 0:
            self._offsets = []

        self._open = gzip.open if self.compression == "gzip" else open

    def __iter__(self) -> Iterator[Any]:
        try:
            with self._open(self.filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        except FileNotFoundError:
            logger.error(f"LazyLoadedList: File not found {self.filepath}")
            return iter([])
        except json.JSONDecodeError as e:
            logger.error(f"LazyLoadedList: JSON decode error in {self.filepath} during iteration: {e}")
            raise

    def _build_index(self):
        if self._offsets is not None:
            return

        self._offsets = []
        line_count = 0
        try:
            with self._open(self.filepath, "rb") as f:
                while True:
                    current_pos = f.tell()
                    line = f.readline()
                    if not line:  # End of file
                        break
                    self._offsets.append(current_pos)
                    line_count += 1
            self._len = line_count
        except FileNotFoundError:
            logger.error(f"LazyLoadedList: File not found while building index: {self.filepath}")
            self._offsets = []
            self._len = 0
        except Exception as e:
            logger.error(
                f"LazyLoadedList: Error building index for {self.filepath}: {e}",
                exc_info=True,
            )
            self._offsets = []
            self._len = 0

    def __len__(self) -> int:
        if self._len is not None:
            return self._len

        if self._offsets is not None:
            self._len = len(self._offsets)
            return self._len
        self._build_index()

        return self._len if self._len is not None else 0

    def __getitem__(self, idx: int) -> Any:
        if not isinstance(idx, int):
            raise TypeError(f"List indices must be integers or slices, not {type(idx).__name__}")

        if self._offsets is None:
            self._build_index()

        if idx < 0:
            if self._len is None:
                self._build_index()
            if self._len == 0:
                raise IndexError("Index out of range for empty list")
            idx = self._len + idx

        if self._offsets is None or not (0 <= idx < len(self._offsets)):
            if self._offsets is None or self._len == 0:
                raise IndexError(f"Index {idx} out of range (list is likely empty or file error for {self.filepath})")
            raise IndexError(f"Index {idx} out of range for {self.filepath} (len: {len(self._offsets)})")

        try:
            with self._open(self.filepath, "rb") as f:
                f.seek(self._offsets[idx])
                line_bytes = f.readline()
                return json.loads(line_bytes.decode("utf-8"))
        except FileNotFoundError:
            raise IndexError(f"File not found when accessing item at index {idx} from {self.filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON at indexed line for index {idx} in {self.filepath}: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error in __getitem__ for index {idx} in {self.filepath}: {e}",
                exc_info=True,
            )
            raise

    def __repr__(self):
        return (
            f"<LazyLoadedList file='{os.path.basename(self.filepath)}', "
            f"len={self.__len__() if self._len is not None else '?'}>"
        )

    def get_all_items(self) -> List[Any]:
        return list(self.__iter__())


class Ingestor:
    """
    Ingestor provides an interface for building, managing, and running data ingestion jobs
    through NvIngestClient, allowing for chainable task additions and job state tracking.

    Parameters
    ----------
    documents : List[str]
        List of document paths to be processed.
    client : Optional[NvIngestClient], optional
        An instance of NvIngestClient. If not provided, a client is created.
    job_queue_id : str, optional
        The ID of the job queue for job submission, default is "ingest_task_queue".
    """

    def __init__(
        self,
        documents: Optional[List[str]] = None,
        client: Optional[NvIngestClient] = None,
        job_queue_id: str = DEFAULT_JOB_QUEUE_ID,
        **kwargs,
    ):
        self._documents = documents or []
        self._client = client
        self._job_queue_id = job_queue_id
        self._vdb_bulk_upload = None
        self._purge_results_after_vdb_upload = True

        if self._client is None:
            client_kwargs = filter_function_kwargs(NvIngestClient, **kwargs)
            self._create_client(**client_kwargs)

        self._all_local = False  # Track whether all files are confirmed as local
        self._job_specs = None
        self._job_ids = None
        self._job_states = None
        self._job_id_to_source_id = {}

        if self._check_files_local():
            self._job_specs = BatchJobSpec(self._documents)
            self._all_local = True

        self._output_config = None
        self._created_temp_output_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._output_config and (self._output_config["cleanup"] is True):
            dir_to_cleanup = self._output_config["output_directory"]
            try:
                shutil.rmtree(dir_to_cleanup)
            except FileNotFoundError:
                logger.warning(
                    f"Directory to be cleaned up not found (might have been removed already): {dir_to_cleanup}"
                )
            except OSError as e:
                logger.error(f"Error removing {dir_to_cleanup}: {e}")

    def _create_client(self, **kwargs) -> None:
        """
        Creates an instance of NvIngestClient if `_client` is not set.

        Raises
        ------
        ValueError
            If `_client` already exists.
        """
        if self._client is not None:
            raise ValueError("self._client already exists.")

        self._client = NvIngestClient(**kwargs)

    @staticmethod
    def _is_remote(pattern: str) -> bool:
        parsed = urlparse(pattern)
        return parsed.scheme in ("http", "https", "s3", "gs", "gcs", "ftp")

    @staticmethod
    def _is_glob(pattern: str) -> bool:
        # only treat '*' and '[' (and '?' when not remote) as glob chars
        wildcard = {"*", "["}
        if not Ingestor._is_remote(pattern):
            wildcard.add("?")
        return any(ch in pattern for ch in wildcard)

    def _check_files_local(self) -> bool:
        """
        Check if all specified document files are local and exist.

        Returns
        -------
        bool
            False immediately if any pattern is a remote URI.
            Local glob-patterns may match zero files (they’re skipped).
            Returns False if any explicit local path is missing
            or any matched file no longer exists.
        """
        if not self._documents:
            return False

        for pattern in self._documents:
            # FAIL on any remote URI
            if self._is_remote(pattern):
                logger.error(f"Remote URI in local-check: {pattern}")
                return False

            # local glob: OK to match zero files
            if self._is_glob(pattern):
                matches = glob.glob(pattern, recursive=True)
                if not matches:
                    logger.debug(f"No files for glob, skipping: {pattern}")
                    continue
            else:
                # explicit local path must exist
                if not os.path.exists(pattern):
                    logger.error(f"Local file not found: {pattern}")
                    return False
                matches = [pattern]

            # verify all matched files still exist
            for fp in matches:
                if not os.path.exists(fp):
                    logger.error(f"Matched file disappeared: {fp}")
                    return False

        return True

    def files(self, documents: Union[str, List[str]]) -> "Ingestor":
        """
        Add documents (local paths, globs, or remote URIs) for processing.

        Remote URIs will force `_all_local=False`. Local globs that match
        nothing are fine. Explicit local paths that don't exist cause
        `_all_local=False`.
        """
        if isinstance(documents, str):
            documents = [documents]
        if not documents:
            return self

        self._documents.extend(documents)
        self._all_local = False

        if self._check_files_local():
            self._job_specs = BatchJobSpec(self._documents)
            self._all_local = True

        return self

    def load(self, **kwargs) -> "Ingestor":
        """
        Ensure all document files are accessible locally, downloading if necessary.

        For each document in `_documents`, checks if the file exists locally. If not,
        attempts to download the file to a temporary directory using `fsspec`. Updates
        `_documents` with paths to local copies, initializes `_job_specs`, and sets
        `_all_local` to True upon successful loading.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for remote file access via `fsspec`.

        Returns
        -------
        Ingestor
            Returns self for chaining after ensuring all files are accessible locally.
        """
        if self._all_local:
            return self

        temp_dir = tempfile.mkdtemp()

        local_files = []
        for pattern_or_path in self._documents:
            files_local = glob.glob(pattern_or_path, recursive=True)
            if files_local:
                for local_path in files_local:
                    local_files.append(local_path)
            else:
                with fsspec.open(pattern_or_path, **kwargs) as f:
                    parsed_url = urlparse(f.path)
                    original_name = os.path.basename(parsed_url.path)
                    local_path = os.path.join(temp_dir, original_name)
                    with open(local_path, "wb") as local_file:
                        shutil.copyfileobj(f, local_file)
                    local_files.append(local_path)

        self._documents = local_files
        self._job_specs = BatchJobSpec(self._documents)
        self._all_local = True

        return self

    def ingest(
        self,
        show_progress: bool = False,
        return_failures: bool = False,
        save_to_disk: bool = False,
        return_traces: bool = False,
        **kwargs: Any,
    ) -> Union[List[Any], Tuple[Any, ...]]:
        """
        Ingest documents by submitting jobs and fetching results concurrently.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. Default is False.
        return_failures : bool, optional
            If True, return a tuple (results, failures); otherwise, return only results. Default is False.
        save_to_disk : bool, optional
            If True, save results to disk and return LazyLoadedList proxies. Default is False.
        return_traces : bool, optional
            If True, return trace metrics alongside results. Default is False.
            Traces contain timing metrics (entry, exit, resident_time) for each stage.
        **kwargs : Any
            Additional keyword arguments for the underlying client methods.
            Optional flags include `include_parent_trace_ids=True` to also return
            parent job trace identifiers (V2 API only).

        Returns
        -------
        list or tuple
            Returns vary based on flags:
            - Default: list of results
            - return_failures=True: (results, failures)
            - return_traces=True: (results, traces)
            - return_failures=True, return_traces=True: (results, failures, traces)
            - Additional combinations with include_parent_trace_ids kwarg

        Notes
        -----
        Trace metrics include timing data for each processing stage. For detailed
        usage and examples, see src/nv_ingest/api/v2/README.md
        """
        if save_to_disk and (not self._output_config):
            self.save_to_disk()

        include_parent_trace_ids = bool(kwargs.pop("include_parent_trace_ids", False))

        self._prepare_ingest_run()

        # Add jobs locally first
        if self._job_specs is None:
            raise RuntimeError("Job specs missing.")
        self._job_ids = self._client.add_job(self._job_specs)

        final_results_payload_list: Union[List[List[Dict[str, Any]]], List[LazyLoadedList]] = []

        # Lock for thread-safe appending to final_results_payload_list by I/O tasks
        results_lock = threading.Lock() if self._output_config else None

        io_executor: Optional[ThreadPoolExecutor] = None
        io_futures: List[Future] = []

        if self._output_config:
            io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="IngestorDiskIO")

        def _perform_save_task(doc_data, job_id, source_name):
            # This function runs in the io_executor
            try:
                output_dir = self._output_config["output_directory"]
                clean_source_basename = get_valid_filename(os.path.basename(source_name))
                file_name, file_ext = os.path.splitext(clean_source_basename)
                file_suffix = f".{file_ext.strip('.')}.results.jsonl"
                if self._output_config["compression"] == "gzip":
                    file_suffix += ".gz"
                jsonl_filepath = os.path.join(output_dir, safe_filename(output_dir, file_name, file_suffix))

                num_items_saved = save_document_results_to_jsonl(
                    doc_data,
                    jsonl_filepath,
                    source_name,
                    ensure_parent_dir_exists=False,
                    compression=self._output_config["compression"],
                )

                if num_items_saved > 0:
                    results = LazyLoadedList(
                        jsonl_filepath, expected_len=num_items_saved, compression=self._output_config["compression"]
                    )
                    if results_lock:
                        with results_lock:
                            final_results_payload_list.append(results)
                    else:  # Should not happen if io_executor is used
                        final_results_payload_list.append(results)
            except Exception as e_save:
                logger.error(
                    f"Disk save I/O task error for job {job_id} (source: {source_name}): {e_save}",
                    exc_info=True,
                )

        def _disk_save_callback(
            results_data: Dict[str, Any],
            job_id: str,
        ):
            source_name = "unknown_source_in_callback"
            job_spec = self._client._job_index_to_job_spec.get(job_id)
            if job_spec:
                source_name = job_spec.source_name
            else:
                try:
                    if results_data:
                        source_name = results_data[0]["metadata"]["source_metadata"]["source_id"]
                except (IndexError, KeyError, TypeError):
                    source_name = f"{job_id}"

            if not results_data:
                logger.warning(f"No data in response for job {job_id} (source: {source_name}). Skipping save.")
                if pbar:
                    pbar.update(1)
                return

            if io_executor:
                future = io_executor.submit(_perform_save_task, results_data, job_id, source_name)
                io_futures.append(future)
            else:  # Fallback to blocking save if no I/O pool
                _perform_save_task(results_data, job_id, source_name)

            if pbar:
                pbar.update(1)

        def _in_memory_callback(
            results_data: Dict[str, Any],
            job_id: str,
        ):
            if pbar:
                pbar.update(1)

        pbar = tqdm(total=len(self._job_ids), desc="Processing", unit="doc") if show_progress else None
        callback: Optional[Callable] = None

        if self._output_config:
            callback = _disk_save_callback
            stream_to_callback_only = True

            output_dir = self._output_config["output_directory"]
            os.makedirs(output_dir, exist_ok=True)
        else:
            callback = _in_memory_callback
            stream_to_callback_only = False

        # Default concurrent-processing parameters
        DEFAULT_TIMEOUT: int = 100
        DEFAULT_MAX_RETRIES: int = None
        DEFAULT_VERBOSE: bool = False

        timeout: int = kwargs.pop("timeout", DEFAULT_TIMEOUT)
        max_job_retries: int = kwargs.pop("max_job_retries", DEFAULT_MAX_RETRIES)
        verbose: bool = kwargs.pop("verbose", DEFAULT_VERBOSE)

        proc_kwargs = filter_function_kwargs(self._client.process_jobs_concurrently, **kwargs)

        # Telemetry controls (optional)
        enable_telemetry: Optional[bool] = kwargs.pop("enable_telemetry", None)
        show_telemetry: Optional[bool] = kwargs.pop("show_telemetry", None)
        if show_telemetry is None:
            # Fallback to env NV_INGEST_CLIENT_SHOW_TELEMETRY (0/1), default off
            try:
                show_telemetry = bool(int(os.getenv("NV_INGEST_CLIENT_SHOW_TELEMETRY", "0")))
            except ValueError:
                show_telemetry = False
        # If user explicitly wants to show telemetry but did not specify enable_telemetry,
        # ensure collection is enabled so summary isn't empty.
        if enable_telemetry is None and show_telemetry:
            enable_telemetry = True
        if enable_telemetry is not None and hasattr(self._client, "enable_telemetry"):
            self._client.enable_telemetry(bool(enable_telemetry))

        # Call process_jobs_concurrently
        proc_result = self._client.process_jobs_concurrently(
            job_indices=self._job_ids,
            job_queue_id=self._job_queue_id,
            timeout=timeout,
            max_job_retries=max_job_retries,
            completion_callback=callback,
            return_failures=True,
            stream_to_callback_only=stream_to_callback_only,
            verbose=verbose,
            return_traces=return_traces,
            **proc_kwargs,
        )

        # Unpack result based on return_traces flag
        if return_traces:
            results, failures, traces_list = proc_result
        else:
            results, failures = proc_result
            traces_list = []  # Empty list when traces not requested

        if show_progress and pbar:
            pbar.close()

        if io_executor:
            for future in as_completed(io_futures):
                try:
                    future.result()
                except Exception as e_io:
                    logger.error(f"A disk I/O task failed: {e_io}", exc_info=True)
            io_executor.shutdown(wait=True)

        if self._output_config:
            results = final_results_payload_list

        if self._vdb_bulk_upload:
            if len(failures) > 0:
                # Calculate success metrics
                total_jobs = len(results) + len(failures)
                successful_jobs = len(results)

                if return_failures:
                    # Emit message about partial success
                    logger.warning(
                        f"Job was not completely successful. "
                        f"{successful_jobs} out of {total_jobs} records completed successfully. "
                        f"Uploading successful results to vector database."
                    )

                    # Upload only the successful results
                    if successful_jobs > 0:
                        self._vdb_bulk_upload.run(results)

                        if self._purge_results_after_vdb_upload:
                            logger.info("Purging saved results from disk after successful VDB upload.")
                            self._purge_saved_results(results)

                else:
                    # Original behavior: raise RuntimeError
                    raise RuntimeError(
                        "Failed to ingest documents, unable to complete vdb bulk upload due to "
                        f"no successful results. {len(failures)} out of {total_jobs} records failed "
                    )
            else:
                # No failures - proceed with normal upload
                self._vdb_bulk_upload.run(results)

                if self._purge_results_after_vdb_upload:
                    logger.info("Purging saved results from disk after successful VDB upload.")
                    self._purge_saved_results(results)

        # Print telemetry summary if requested
        if show_telemetry:
            try:
                summary = self._client.summarize_telemetry()
                # Print to stdout and log for convenience
                print("NvIngestClient Telemetry Summary:", json.dumps(summary, indent=2))
                logger.info("NvIngestClient Telemetry Summary: %s", json.dumps(summary, indent=2))
            except Exception:
                pass

        parent_trace_ids = self._client.consume_completed_parent_trace_ids() if include_parent_trace_ids else []

        # Build return tuple based on requested outputs
        # Order: results, failures (if requested), traces (if requested), parent_trace_ids (if requested)
        returns = [results]

        if return_failures:
            returns.append(failures)
        if return_traces:
            returns.append(traces_list)
        if include_parent_trace_ids:
            returns.append(parent_trace_ids)

        return tuple(returns) if len(returns) > 1 else results

    def ingest_async(self, **kwargs: Any) -> Future:
        """
        Asynchronously submits jobs and returns a single future that completes when all jobs have finished.

        Parameters
        ----------
        kwargs : dict
            Additional parameters for the `submit_job_async` method.

        Returns
        -------
        Future
            A future that completes when all submitted jobs have reached a terminal state.
        """
        self._prepare_ingest_run()

        self._job_ids = self._client.add_job(self._job_specs)

        future_to_job_id = self._client.submit_job_async(self._job_ids, self._job_queue_id, **kwargs)
        self._job_states = {job_id: self._client._get_and_check_job_state(job_id) for job_id in self._job_ids}

        combined_future = Future()
        submitted_futures = set(future_to_job_id.keys())
        completed_futures = set()
        future_results = []
        vdb_future = None

        def _done_callback(future):
            job_id = future_to_job_id[future]
            job_state = self._job_states[job_id]
            try:
                result = self._client.fetch_job_result(job_id)
                if job_state.state != JobStateEnum.COMPLETED:
                    job_state.state = JobStateEnum.COMPLETED
            except Exception:
                result = None
                if job_state.state != JobStateEnum.FAILED:
                    job_state.state = JobStateEnum.FAILED
            completed_futures.add(future)
            future_results.extend(result)
            if completed_futures == submitted_futures:
                combined_future.set_result(future_results)

        for future in future_to_job_id:
            future.add_done_callback(_done_callback)

        if self._vdb_bulk_upload:
            executor = ThreadPoolExecutor(max_workers=1)
            vdb_future = executor.submit(self._vdb_bulk_upload.run_async, combined_future)

        return combined_future if not vdb_future else vdb_future

    @ensure_job_specs
    def _prepare_ingest_run(self):
        """
        Prepares the ingest run by ensuring tasks are added to the batch job specification.

        If no tasks are specified in `_job_specs`, this method invokes `all_tasks()` to add
        a default set of tasks to the job specification.
        """
        if (not self._job_specs.tasks) or all(not tasks for tasks in self._job_specs.tasks.values()):
            self.all_tasks()

    def all_tasks(self) -> "Ingestor":
        """
        Adds a default set of tasks to the batch job specification.

        The default tasks include extracting text, tables, charts, images, deduplication,
        filtering, splitting, and embedding tasks.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        # fmt: off
        self.extract(extract_text=True, extract_tables=True, extract_charts=True, extract_images=True) \
            .dedup() \
            .filter() \
            .split() \
            .embed() \
            .store_embed()
        # .store() \
        # fmt: on
        return self

    @ensure_job_specs
    def dedup(self, **kwargs: Any) -> "Ingestor":
        """
        Adds a DedupTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the DedupTask.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        # Extract content_type and build params dict for API schema
        content_type = kwargs.pop("content_type", "text")  # Default to "text" if not specified
        params = kwargs  # Remaining parameters go into params dict

        # Validate with API schema
        api_options = {
            "content_type": content_type,
            "params": params,
        }
        task_options = check_schema(IngestTaskDedupSchema, api_options, "dedup", json.dumps(api_options))

        # Extract individual parameters from API schema for DedupTask constructor
        dedup_params = {
            "content_type": task_options.content_type,
            "filter": task_options.params.filter,
        }
        dedup_task = DedupTask(**dedup_params)
        self._job_specs.add_task(dedup_task)

        return self

    @ensure_job_specs
    def embed(self, **kwargs: Any) -> "Ingestor":
        """
        Adds an EmbedTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the EmbedTask.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        # Filter out deprecated parameters before API schema validation
        # The EmbedTask constructor handles these deprecated parameters with warnings
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["text", "tables"]}

        _ = check_schema(IngestTaskEmbedSchema, filtered_kwargs, "embed", json.dumps(filtered_kwargs))

        # Pass original kwargs to EmbedTask constructor so it can handle deprecated parameters
        embed_task = EmbedTask(**kwargs)
        self._job_specs.add_task(embed_task)

        return self

    @ensure_job_specs
    def extract(self, **kwargs: Any) -> "Ingestor":
        """
        Adds an ExtractTask for each document type to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the ExtractTask.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        extract_tables = kwargs.pop("extract_tables", True)
        extract_charts = kwargs.pop("extract_charts", True)
        extract_page_as_image = kwargs.pop("extract_page_as_image", False)
        table_output_format = kwargs.pop("table_output_format", "markdown")

        # Defaulting to False since enabling infographic extraction reduces throughput.
        # Users have to set to True if infographic extraction is required.
        extract_infographics = kwargs.pop("extract_infographics", False)

        for file_type in self._job_specs.file_types:
            # Let user override document_type if user explicitly sets document_type.
            if "document_type" in kwargs:
                document_type = kwargs.pop("document_type")
                if document_type != file_type:
                    logger.warning(
                        f"User-specified document_type '{document_type}' overrides the inferred type '{file_type}'.",
                    )
            else:
                document_type = file_type

            task_options = dict(
                document_type=document_type,
                extract_tables=extract_tables,
                extract_charts=extract_charts,
                extract_infographics=extract_infographics,
                extract_page_as_image=extract_page_as_image,
                table_output_format=table_output_format,
                **kwargs,
            )

            # Extract method from task_options for API schema
            method = task_options.pop("extract_method", None)
            if method is None:
                # Let ExtractTask constructor handle default method selection
                method = "pdfium"  # Default fallback

            # Build params dict for API schema
            params = {k: v for k, v in task_options.items() if k != "document_type"}

            # Map document type to API schema expected values
            # Handle common file extension to DocumentTypeEnum mapping
            document_type_mapping = {
                "txt": "text",
                "md": "text",
                "sh": "text",
                "json": "text",
                "jpg": "jpeg",
                "jpeg": "jpeg",
                "png": "png",
                "pdf": "pdf",
                "docx": "docx",
                "pptx": "pptx",
                "html": "html",
                "bmp": "bmp",
                "tiff": "tiff",
                "svg": "svg",
                "mp3": "mp3",
                "wav": "wav",
            }

            # Use mapped document type for API schema validation
            api_document_type = document_type_mapping.get(document_type.lower(), document_type)

            # Validate with API schema
            api_task_options = {
                "document_type": api_document_type,
                "method": method,
                "params": params,
            }

            check_schema(IngestTaskExtractSchema, api_task_options, "extract", json.dumps(api_task_options))

            # Create ExtractTask with mapped document type for API schema compatibility
            extract_task_params = {"document_type": api_document_type, "extract_method": method, **params}
            extract_task = ExtractTask(**extract_task_params)
            self._job_specs.add_task(extract_task, document_type=document_type)

        return self

    @ensure_job_specs
    def filter(self, **kwargs: Any) -> "Ingestor":
        """
        Adds a FilterTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the FilterTask.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        # Restructure parameters to match API schema structure
        params_fields = {"min_size", "max_aspect_ratio", "min_aspect_ratio", "filter"}
        params = {k: v for k, v in kwargs.items() if k in params_fields}
        top_level = {k: v for k, v in kwargs.items() if k not in params_fields}

        # Build API schema structure
        api_kwargs = top_level.copy()
        if params:
            api_kwargs["params"] = params

        task_options = check_schema(IngestTaskFilterSchema, api_kwargs, "filter", json.dumps(api_kwargs))

        # Extract individual parameters from API schema for FilterTask constructor
        filter_params = {
            "content_type": task_options.content_type,
            "min_size": task_options.params.min_size,
            "max_aspect_ratio": task_options.params.max_aspect_ratio,
            "min_aspect_ratio": task_options.params.min_aspect_ratio,
            "filter": task_options.params.filter,
        }
        filter_task = FilterTask(**filter_params)
        self._job_specs.add_task(filter_task)

        return self

    @ensure_job_specs
    def split(self, **kwargs: Any) -> "Ingestor":
        """
        Adds a SplitTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the SplitTask.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        task_options = check_schema(IngestTaskSplitSchema, kwargs, "split", json.dumps(kwargs))
        extract_task = SplitTask(**task_options.model_dump())
        self._job_specs.add_task(extract_task)

        return self

    @ensure_job_specs
    def store(self, **kwargs: Any) -> "Ingestor":
        """
        Adds a StoreTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the StoreTask.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        # Handle parameter name mapping: store_method -> method for API schema
        if "store_method" in kwargs:
            kwargs["method"] = kwargs.pop("store_method")

        # Provide default method if not specified (matching client StoreTask behavior)
        if "method" not in kwargs:
            kwargs["method"] = "minio"

        task_options = check_schema(IngestTaskStoreSchema, kwargs, "store", json.dumps(kwargs))

        # Map API schema fields back to StoreTask constructor parameters
        store_params = {
            "structured": task_options.structured,
            "images": task_options.images,
            "store_method": task_options.method,  # Map method back to store_method
            "params": task_options.params,
        }
        store_task = StoreTask(**store_params)
        self._job_specs.add_task(store_task)

        return self

    @ensure_job_specs
    def store_embed(self, **kwargs: Any) -> "Ingestor":
        """
        Adds a StoreEmbedTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the StoreEmbedTask.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        task_options = check_schema(IngestTaskStoreEmbedSchema, kwargs, "store_embedding", json.dumps(kwargs))
        store_task = StoreEmbedTask(**task_options.model_dump())
        self._job_specs.add_task(store_task)

        return self

    def udf(
        self,
        udf_function: str,
        udf_function_name: Optional[str] = None,
        phase: Optional[Union[PipelinePhase, int, str]] = None,
        target_stage: Optional[str] = None,
        run_before: bool = False,
        run_after: bool = False,
    ) -> "Ingestor":
        """
        Adds a UDFTask to the batch job specification.

        Parameters
        ----------
        udf_function : str
            UDF specification. Supports three formats:
            1. Inline function: 'def my_func(control_message): ...'
            2. Import path: 'my_module.my_function'
            3. File path: '/path/to/file.py:function_name'
        udf_function_name : str, optional
            Name of the function to execute from the UDF specification.
            If not provided, attempts to infer from udf_function.
        phase : Union[PipelinePhase, int, str], optional
            Pipeline phase to execute UDF. Accepts phase names ('extract', 'split', 'embed', 'response')
            or numbers (1-4). Cannot be used with target_stage.
        target_stage : str, optional
            Specific stage name to target for UDF execution. Cannot be used with phase.
        run_before : bool, optional
            If True and target_stage is specified, run UDF before the target stage. Default: False.
        run_after : bool, optional
            If True and target_stage is specified, run UDF after the target stage. Default: False.

        Returns
        -------
        Ingestor
            Returns self for chaining.

        Raises
        ------
        ValueError
            If udf_function_name cannot be inferred and is not provided explicitly,
            or if both phase and target_stage are specified, or if neither is specified.
        """
        # Validate mutual exclusivity of phase and target_stage
        if phase is not None and target_stage is not None:
            raise ValueError("Cannot specify both 'phase' and 'target_stage'. Please specify only one.")
        elif phase is None and target_stage is None:
            # Default to response phase for backward compatibility
            phase = PipelinePhase.RESPONSE

        # Try to infer udf_function_name if not provided
        if udf_function_name is None:
            udf_function_name = infer_udf_function_name(udf_function)
            if udf_function_name is None:
                raise ValueError(
                    f"Could not infer UDF function name from '{udf_function}'. "
                    "Please specify 'udf_function_name' explicitly."
                )
            logger.info(f"Inferred UDF function name: {udf_function_name}")

        # Use UDFTask constructor with explicit parameters
        udf_task = UDFTask(
            udf_function=udf_function,
            udf_function_name=udf_function_name,
            phase=phase,
            target_stage=target_stage,
            run_before=run_before,
            run_after=run_after,
        )
        self._job_specs.add_task(udf_task)

        return self

    def vdb_upload(self, purge_results_after_upload: bool = True, **kwargs: Any) -> "Ingestor":
        """
        Adds a VdbUploadTask to the batch job specification.

        Parameters
        ----------
        purge_results_after_upload : bool, optional
            If True, the saved result files will be deleted from disk after a successful
            upload. This requires `save_to_disk()` to be active. Defaults to True
        kwargs : dict
            Parameters specific to the VdbUploadTask.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        vdb_op = kwargs.pop("vdb_op", "milvus")
        if isinstance(vdb_op, str):
            op_cls = get_vdb_op_cls(vdb_op)
            vdb_op = op_cls(**kwargs)
        elif isinstance(vdb_op, VDB):
            vdb_op = vdb_op
        else:
            raise ValueError(f"Invalid type for op: {type(vdb_op)}, must be type VDB or str.")

        self._vdb_bulk_upload = vdb_op
        self._purge_results_after_vdb_upload = purge_results_after_upload

        return self

    def save_to_disk(
        self,
        output_directory: Optional[str] = None,
        cleanup: bool = True,
        compression: Optional[str] = "gzip",
    ) -> "Ingestor":
        """Configures the Ingestor to save results to disk instead of memory.

        This method enables disk-based storage for ingestion results. When called,
        the `ingest()` method will write the output for each processed document to a
        separate JSONL file. The return value of `ingest()` will be a list of
        `LazyLoadedList` objects, which are memory-efficient proxies to these files.

        The output directory can be specified directly, via an environment variable,
        or a temporary directory will be created automatically.

        Parameters
        ----------
        output_directory : str, optional
            The path to the directory where result files (.jsonl) will be saved.
            If not provided, it defaults to the value of the environment variable
            `NV_INGEST_CLIENT_SAVE_TO_DISK_OUTPUT_DIRECTORY`. If the environment
            variable is also not set, a temporary directory will be created.
            Defaults to None.
        cleanup : bool, optional)
            If True, the entire `output_directory` will be recursively deleted
            when the Ingestor's context is exited (i.e., when used in a `with`
            statement).
            Defaults to True.
        compression : str, optional
            The compression algorithm to use for the saved result files.
            Currently, the only supported value is `'gzip'`. To disable
            compression, set this parameter to `None`. Defaults to `'gzip'`,
            which significantly reduces the disk space required for results.
            When enabled, files are saved with a `.gz` suffix (e.g., `results.jsonl.gz`).

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        output_directory = output_directory or os.getenv("NV_INGEST_CLIENT_SAVE_TO_DISK_OUTPUT_DIRECTORY")

        if not output_directory:
            self._created_temp_output_dir = tempfile.mkdtemp(prefix="ingestor_results_")
            output_directory = self._created_temp_output_dir

        self._output_config = {
            "output_directory": output_directory,
            "cleanup": cleanup,
            "compression": compression,
        }
        ensure_directory_with_permissions(output_directory)

        return self

    def _purge_saved_results(self, saved_results: List[LazyLoadedList]):
        """
        Deletes the .jsonl files associated with the results and the temporary
        output directory if it was created by this Ingestor instance.
        """
        if not self._output_config:
            logger.warning("Purge requested, but save_to_disk was not configured. No files to purge.")
            return

        deleted_files_count = 0
        for result_item in saved_results:
            if isinstance(result_item, LazyLoadedList) and hasattr(result_item, "filepath"):
                filepath = result_item.filepath
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        deleted_files_count += 1
                        logger.debug(f"Purged result file: {filepath}")
                except OSError as e:
                    logger.error(f"Error purging result file {filepath}: {e}", exc_info=True)

        logger.info(f"Purged {deleted_files_count} saved result file(s).")

        if self._created_temp_output_dir:
            logger.info(f"Removing temporary output directory: {self._created_temp_output_dir}")
            try:
                shutil.rmtree(self._created_temp_output_dir)
                self._created_temp_output_dir = None  # Reset flag after successful removal
            except OSError as e:
                logger.error(
                    f"Error removing temporary output directory {self._created_temp_output_dir}: {e}",
                    exc_info=True,
                )

    @ensure_job_specs
    def caption(self, **kwargs: Any) -> "Ingestor":
        """
        Adds a CaptionTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the CaptionTask.

        Returns
        -------
        Ingestor
            Returns self for chaining.
        """
        task_options = check_schema(IngestTaskCaptionSchema, kwargs, "caption", json.dumps(kwargs))

        # Extract individual parameters from API schema for CaptionTask constructor
        caption_params = {
            "api_key": task_options.api_key,
            "endpoint_url": task_options.endpoint_url,
            "prompt": task_options.prompt,
            "model_name": task_options.model_name,
        }
        caption_task = CaptionTask(**caption_params)
        self._job_specs.add_task(caption_task)

        return self

    @ensure_job_specs
    def pdf_split_config(self, pages_per_chunk: int = 32) -> "Ingestor":
        """
        Configure PDF splitting behavior for V2 API.

        Parameters
        ----------
        pages_per_chunk : int, optional
            Number of pages per PDF chunk (default: 32)
            Server enforces boundaries: min=1, max=128

        Returns
        -------
        Ingestor
            Self for method chaining

        Notes
        -----
        - Only affects V2 API endpoints with PDF splitting support
        - Server will clamp values outside [1, 128] range
        - Smaller chunks = more parallelism but more overhead
        - Larger chunks = less overhead but reduced concurrency
        """
        MIN_PAGES = 1
        MAX_PAGES = 128

        # Warn if value will be clamped by server
        if pages_per_chunk < MIN_PAGES:
            logger.warning(f"pages_per_chunk={pages_per_chunk} is below minimum. Server will clamp to {MIN_PAGES}.")
        elif pages_per_chunk > MAX_PAGES:
            logger.warning(f"pages_per_chunk={pages_per_chunk} exceeds maximum. Server will clamp to {MAX_PAGES}.")

        # Flatten all job specs and apply PDF config using shared utility
        all_job_specs = [spec for job_specs in self._job_specs._file_type_to_job_spec.values() for spec in job_specs]
        apply_pdf_split_config_to_job_specs(all_job_specs, pages_per_chunk)

        return self

    def _count_job_states(self, job_states: set[JobStateEnum]) -> int:
        """
        Counts the jobs in specified states.

        Parameters
        ----------
        job_states : set
            Set of JobStateEnum states to count.

        Returns
        -------
        int
            Count of jobs in specified states.
        """
        count = 0
        for job_id, job_state in self._job_states.items():
            if job_state.state in job_states:
                count += 1
        return count

    def completed_jobs(self) -> int:
        """
        Counts the jobs that have completed successfully.

        Returns
        -------
        int
            Number of jobs in the COMPLETED state.
        """
        completed_job_states = {JobStateEnum.COMPLETED}

        return self._count_job_states(completed_job_states)

    def failed_jobs(self) -> int:
        """
        Counts the jobs that have failed.

        Returns
        -------
        int
            Number of jobs in the FAILED state.
        """
        failed_job_states = {JobStateEnum.FAILED}

        return self._count_job_states(failed_job_states)

    def cancelled_jobs(self) -> int:
        """
        Counts the jobs that have been cancelled.

        Returns
        -------
        int
            Number of jobs in the CANCELLED state.
        """
        cancelled_job_states = {JobStateEnum.CANCELLED}

        return self._count_job_states(cancelled_job_states)

    def remaining_jobs(self) -> int:
        """
        Counts the jobs that are not in a terminal state.

        Returns
        -------
        int
            Number of jobs that are neither completed, failed, nor cancelled.
        """
        terminal_jobs = self.completed_jobs() + self.failed_jobs() + self.cancelled_jobs()

        return len(self._job_states) - terminal_jobs

    def get_status(self) -> Dict[str, str]:
        """
        Returns a dictionary mapping document identifiers to their current status in the pipeline.

        This method is designed for use with async ingestion to poll the status of submitted jobs.
        For each document submitted to the ingestor, the method returns its current processing state.

        Returns
        -------
        Dict[str, str]
            A dictionary where:
            - Keys are document identifiers (source names or source IDs)
            - Values are status strings representing the current state:
              * "pending": Job created but not yet submitted
              * "submitted": Job submitted and waiting for processing
              * "processing": Job is currently being processed
              * "completed": Job finished successfully
              * "failed": Job encountered an error
              * "cancelled": Job was cancelled
              * "unknown": Job state could not be determined (initial state)

        Examples
        --------
        >>> ingestor = Ingestor(documents=["doc1.pdf", "doc2.pdf"], client=client)
        >>> ingestor.extract().embed()
        >>> future = ingestor.ingest_async()
        >>>
        >>> # Poll status while processing
        >>> status = ingestor.get_status()
        >>> print(status)
        {'doc1.pdf': 'processing', 'doc2.pdf': 'submitted'}
        >>>
        >>> # Check again after some time
        >>> status = ingestor.get_status()
        >>> print(status)
        {'doc1.pdf': 'completed', 'doc2.pdf': 'processing'}

        Notes
        -----
        - This method is most useful when called after `ingest_async()` to track progress
        - If called before any jobs are submitted, returns an empty dictionary or
          documents with "unknown" status
        - The method accesses internal job state from the client, so it reflects
          the most current known state
        """
        status_dict = {}

        if not self._job_states:
            # If job states haven't been initialized yet (before ingest_async is called)
            # Return unknown status for all documents
            for doc in self._documents:
                doc_name = os.path.basename(doc) if isinstance(doc, str) else str(doc)
                status_dict[doc_name] = "unknown"
            return status_dict

        # Map job IDs to their states and source identifiers
        for job_id, job_state in self._job_states.items():
            # Get the job spec to find the source identifier
            job_spec = self._client._job_index_to_job_spec.get(job_id)

            if job_spec:
                # Use source_name as the key (the document name)
                source_identifier = job_spec.source_name
            else:
                # Fallback to job_id if we can't find the spec
                source_identifier = f"job_{job_id}"

            # Map the JobStateEnum to a user-friendly string
            state_mapping = {
                JobStateEnum.PENDING: "pending",
                JobStateEnum.SUBMITTED_ASYNC: "submitted",
                JobStateEnum.SUBMITTED: "submitted",
                JobStateEnum.PROCESSING: "processing",
                JobStateEnum.COMPLETED: "completed",
                JobStateEnum.FAILED: "failed",
                JobStateEnum.CANCELLED: "cancelled",
            }

            status_dict[source_identifier] = state_mapping.get(job_state.state, "unknown")

        return status_dict
