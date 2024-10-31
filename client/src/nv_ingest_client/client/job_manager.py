# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from concurrent.futures import Future
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from nv_ingest_client.client.client import NvIngestClient
from nv_ingest_client.primitives import BatchJobSpec
from nv_ingest_client.primitives.jobs import JobStateEnum
from nv_ingest_client.primitives.tasks import DedupTask
from nv_ingest_client.primitives.tasks import EmbedTask
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import FilterTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import StoreTask
from nv_ingest_client.primitives.tasks import VdbUploadTask
from nv_ingest_client.primitives.tasks.chart_extraction import ChartExtractionTask
from nv_ingest_client.primitives.tasks.table_extraction import TableExtractionTask
from nv_ingest_client.util.util import filter_function_kwargs


DEFAULT_JOB_QUEUE_ID = "morpheus_task_queue"


class NvIngestJobManager:
    """
    NvIngestJobManager provides an interface for building, managing, and running data ingestion jobs
    through NvIngestClient, allowing for chainable task additions and job state tracking.

    Parameters
    ----------
    documents : List[str]
        List of document paths to be processed.
    client : Optional[NvIngestClient], optional
        An instance of NvIngestClient. If not provided, a client is created.
    job_queue_id : str, optional
        The ID of the job queue for job submission, default is "morpheus_task_queue".
    """

    def __init__(
        self,
        documents: List[str],
        client: Optional[NvIngestClient] = None,
        job_queue_id: str = DEFAULT_JOB_QUEUE_ID,
        **kwargs,
    ):
        self._documents = documents
        self._client = client
        self._job_queue_id = job_queue_id

        if self._client is None:
            client_kwargs = filter_function_kwargs(NvIngestClient, **kwargs)
            self._create_client(**client_kwargs)

        self._job_specs = BatchJobSpec(self._documents)
        self._job_ids = None
        self._job_states = None

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

    def run(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Synchronously submits jobs to the NvIngestClient and fetches the results.

        Parameters
        ----------
        kwargs : dict
            Additional parameters for `submit_job` and `fetch_job_result` methods of NvIngestClient.

        Returns
        -------
        List[Dict]
            Result of each job after execution.
        """
        self._job_ids = self._client.add_job(self._job_specs)

        submit_kwargs = filter_function_kwargs(self._client.submit_job, **kwargs)
        self._job_states = self._client.submit_job(self._job_ids, self._job_queue_id, **submit_kwargs)

        fetch_kwargs = filter_function_kwargs(self._client.fetch_job_result, **kwargs)
        result = self._client.fetch_job_result(self._job_ids, **fetch_kwargs)

        return result

    def run_async(self, **kwargs: Any) -> Future:
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
            future_results.append(result)
            if completed_futures == submitted_futures:
                combined_future.set_result(future_results)

        for future in future_to_job_id:
            future.add_done_callback(_done_callback)

        return combined_future

    def dedup(self, **kwargs: Any) -> "NvIngestJobManager":
        """
        Adds a DedupTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the DedupTask.

        Returns
        -------
        NvIngestJobManager
            Returns self for chaining.
        """
        dedup_task = DedupTask(**kwargs)
        self._job_specs.add_task(dedup_task)

        return self

    def embed(self, **kwargs: Any) -> "NvIngestJobManager":
        """
        Adds an EmbedTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the EmbedTask.

        Returns
        -------
        NvIngestJobManager
            Returns self for chaining.
        """
        embed_task = EmbedTask(**kwargs)
        self._job_specs.add_task(embed_task)

        return self

    def extract(self, **kwargs: Any) -> "NvIngestJobManager":
        """
        Adds an ExtractTask for each document type to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the ExtractTask.

        Returns
        -------
        NvIngestJobManager
            Returns self for chaining.
        """
        extract_tables = kwargs.get("extract_tables", False)
        extract_charts = kwargs.get("extract_charts", False)

        for document_type in self._job_specs.file_types:
            extract_task = ExtractTask(document_type, **kwargs)
            self._job_specs.add_task(extract_task, document_type=document_type)

            if extract_tables is True:
                self._job_specs.add_task(TableExtractionTask())
            if extract_charts is True:
                self._job_specs.add_task(ChartExtractionTask())

        return self

    def filter(self, **kwargs: Any) -> "NvIngestJobManager":
        """
        Adds a FilterTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the FilterTask.

        Returns
        -------
        NvIngestJobManager
            Returns self for chaining.
        """
        filter_task = FilterTask(**kwargs)
        self._job_specs.add_task(filter_task)

        return self

    def split(self, **kwargs: Any) -> "NvIngestJobManager":
        """
        Adds a SplitTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the SplitTask.

        Returns
        -------
        NvIngestJobManager
            Returns self for chaining.
        """
        extract_task = SplitTask(**kwargs)
        self._job_specs.add_task(extract_task)

        return self

    def store(self, **kwargs: Any) -> "NvIngestJobManager":
        """
        Adds a StoreTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the StoreTask.

        Returns
        -------
        NvIngestJobManager
            Returns self for chaining.
        """
        store_task = StoreTask(**kwargs)
        self._job_specs.add_task(store_task)

        return self

    def vdb_upload(self, **kwargs: Any) -> "NvIngestJobManager":
        """
        Adds a VdbUploadTask to the batch job specification.

        Parameters
        ----------
        kwargs : dict
            Parameters specific to the VdbUploadTask.

        Returns
        -------
        NvIngestJobManager
            Returns self for chaining.
        """
        vdb_upload_task = VdbUploadTask(**kwargs)
        self._job_specs.add_task(vdb_upload_task)

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
