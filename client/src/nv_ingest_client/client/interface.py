# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import collections
import glob
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
from nv_ingest_client.primitives.tasks import StoreEmbedTask
from nv_ingest_client.primitives.tasks import StoreTask
from nv_ingest_client.primitives.tasks.caption import CaptionTaskSchema
from nv_ingest_client.primitives.tasks.dedup import DedupTaskSchema
from nv_ingest_client.primitives.tasks.embed import EmbedTaskSchema
from nv_ingest_client.primitives.tasks.extract import ExtractTaskSchema
from nv_ingest_client.primitives.tasks.filter import FilterTaskSchema
from nv_ingest_client.primitives.tasks.split import SplitTaskSchema
from nv_ingest_client.primitives.tasks.store import StoreEmbedTaskSchema
from nv_ingest_client.primitives.tasks.store import StoreTaskSchema
from nv_ingest_client.util.processing import check_schema
from nv_ingest_client.util.system import ensure_directory_with_permissions
from nv_ingest_client.util.util import filter_function_kwargs
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
    def __init__(self, filepath: str, expected_len: Optional[int] = None):
        self.filepath = filepath
        self._len: Optional[int] = expected_len  # Store pre-calculated length
        self._offsets: Optional[List[int]] = None

        if self._len == 0:
            self._offsets = []

    def __iter__(self) -> Iterator[Any]:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
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
            with open(self.filepath, "rb") as f:
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
            with open(self.filepath, "rb") as f:
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
        **kwargs: Any,
    ) -> Union[
        List[List[Dict[str, Any]]],  # In-memory: List of (response['data'] for each doc)
        List[LazyLoadedList],  # Disk: List of proxies, one per original doc
        Tuple[
            Union[List[List[Dict[str, Any]]], List[LazyLoadedList]],
            List[Tuple[str, str]],
        ],
    ]:  # noqa: E501
        """
        Ingest documents by submitting jobs and fetching results concurrently.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display a progress bar. Default is False.
        return_failures : bool, optional
            If True, return a tuple (results, failures); otherwise, return only results. Default is False.
        **kwargs : Any
            Additional keyword arguments for the underlying client methods. Supported keys:
            'concurrency_limit', 'timeout', 'max_job_retries', 'retry_delay',
            'data_only', 'verbose'. Unrecognized keys are passed through to
            process_jobs_concurrently.

        Returns
        -------
        results : list of dict
            List of successful job results when `return_failures` is False.
        results, failures : tuple (list of dict, list of tuple of str)
            Tuple containing successful results and failure information when `return_failures` is True.
        """
        if save_to_disk and (not self._output_config):
            self.save_to_disk()

        self._prepare_ingest_run()

        # Add jobs locally first
        if self._job_specs is None:
            raise RuntimeError("Job specs missing.")
        self._job_ids = self._client.add_job(self._job_specs)

        final_results_payload_list: Union[List[List[Dict[str, Any]]], List[LazyLoadedList]] = []

        # Lock for thread-safe appends to final_results_payload_list by I/O tasks
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
                file_suffix = f".{file_ext}.results.jsonl"
                jsonl_filepath = os.path.join(output_dir, safe_filename(output_dir, file_name, file_suffix))

                num_items_saved = save_document_results_to_jsonl(
                    doc_data,
                    jsonl_filepath,
                    source_name,
                    ensure_parent_dir_exists=False,
                )

                if num_items_saved > 0:
                    results = LazyLoadedList(jsonl_filepath, expected_len=num_items_saved)
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

        _return_failures = return_failures
        if self._vdb_bulk_upload:
            return_failures = True

        results, failures = self._client.process_jobs_concurrently(
            job_indices=self._job_ids,
            job_queue_id=self._job_queue_id,
            timeout=timeout,
            max_job_retries=max_job_retries,
            completion_callback=callback,
            return_failures=True,
            stream_to_callback_only=stream_to_callback_only,
            verbose=verbose,
            **proc_kwargs,
        )

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
                raise RuntimeError(f"Failed to ingest documents, unable to complete vdb bulk upload: {failures}")

            self._vdb_bulk_upload.run(results)

        return_failures = _return_failures

        return (results, failures) if return_failures else results

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
            self._vdb_bulk_upload.run(combined_future.result())

        return combined_future

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
        task_options = check_schema(DedupTaskSchema, kwargs, "dedup", json.dumps(kwargs))
        dedup_task = DedupTask(**task_options.model_dump())
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
        task_options = check_schema(EmbedTaskSchema, kwargs, "embed", json.dumps(kwargs))
        embed_task = EmbedTask(**task_options.model_dump())
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
                **kwargs,
            )
            task_options = check_schema(ExtractTaskSchema, task_options, "extract", json.dumps(task_options))

            extract_task = ExtractTask(**task_options.model_dump())
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
        task_options = check_schema(FilterTaskSchema, kwargs, "filter", json.dumps(kwargs))
        filter_task = FilterTask(**task_options.model_dump())
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
        task_options = check_schema(SplitTaskSchema, kwargs, "split", json.dumps(kwargs))
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
        task_options = check_schema(StoreTaskSchema, kwargs, "store", json.dumps(kwargs))
        store_task = StoreTask(**task_options.model_dump())
        self._job_specs.add_task(store_task)

        return self

    @ensure_job_specs
    def store_embed(self, **kwargs: Any) -> "Ingestor":
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
        task_options = check_schema(StoreEmbedTaskSchema, kwargs, "store_embedding", json.dumps(kwargs))
        store_task = StoreEmbedTask(**task_options.model_dump())
        self._job_specs.add_task(store_task)

        return self

    def vdb_upload(self, **kwargs: Any) -> "Ingestor":
        """
        Adds a VdbUploadTask to the batch job specification.

        Parameters
        ----------
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

        return self

    def save_to_disk(
        self,
        output_directory: Optional[str] = None,
    ) -> "Ingestor":
        if not output_directory:
            self._created_temp_output_dir = tempfile.mkdtemp(prefix="ingestor_results_")
            output_directory = self._created_temp_output_dir

        self._output_config = {
            "output_directory": output_directory,
        }
        ensure_directory_with_permissions(output_directory)

        return self

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
        task_options = check_schema(CaptionTaskSchema, kwargs, "caption", json.dumps(kwargs))
        caption_task = CaptionTask(**task_options.model_dump())
        self._job_specs.add_task(caption_task)

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
