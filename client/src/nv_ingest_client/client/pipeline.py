# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import inspect
from concurrent.futures import Future
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

DEFAULT_JOB_QUEUE_ID = "morpheus_task_queue"


class NvIngestPipeline:
    def __init__(
        self,
        documents: List[str],
        client: Optional[NvIngestClient] = None,
        job_queue_id=DEFAULT_JOB_QUEUE_ID,
    ):
        self._documents = documents
        self._client = client
        self._job_queue_id = job_queue_id

        if self._client is None:
            self._create_client()

        self._job_specs = BatchJobSpec(self._documents)
        self._job_ids = None
        self._job_states = None

    def _create_client(self):
        if self._client is not None:
            raise ValueError("self._client already exists.")

        self._client = NvIngestClient()

    def run(self, **kwargs):
        self._job_ids = self._client.add_job(self._job_specs)

        submit_args = list(inspect.signature(self._client.submit_job).parameters)
        submit_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in submit_args}
        self._job_states = self._client.submit_job(self._job_ids, self._job_queue_id, **submit_dict)

        fetch_args = list(inspect.signature(self._client.fetch_job_result).parameters)
        fetch_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in fetch_args}
        result = self._client.fetch_job_result(self._job_ids, **fetch_dict)

        return result

    def run_async(self, **kwargs):
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

    def dedup(self, **kwargs):
        dedup_task = DedupTask(**kwargs)
        self._job_specs.add_task(dedup_task)

        return self

    def embed(self, **kwargs):
        embed_task = EmbedTask(**kwargs)
        self._job_specs.add_task(embed_task)

        return self

    def extract(self, **kwargs):
        for document_type in self._job_specs.file_types:
            extract_task = ExtractTask(document_type, **kwargs)
            self._job_specs.add_task(extract_task, document_type=document_type)

        return self

    def filter(self, **kwargs):
        filter_task = FilterTask(**kwargs)
        self._job_specs.add_task(filter_task)

        return self

    def split(self, **kwargs):
        extract_task = SplitTask(**kwargs)
        self._job_specs.add_task(extract_task)

        return self

    def store(self, **kwargs):
        store_task = StoreTask(**kwargs)
        self._job_specs.add_task(store_task)

        return self

    def vdb_upload(self, **kwargs):
        vdb_upload_task = VdbUploadTask(**kwargs)
        self._job_specs.add_task(vdb_upload_task)

        return self

    def _count_job_states(self, job_states):
        count = 0
        for job_id, job_state in self._job_states.items():
            if job_state.state in job_states:
                count += 1
        return count

    def completed_jobs(self):
        completed_job_states = {JobStateEnum.COMPLETED}

        return self._count_job_states(completed_job_states)

    def failed_jobs(self):
        failed_job_states = {JobStateEnum.FAILED}

        return self._count_job_states(failed_job_states)

    def cancelled_jobs(self):
        cancelled_job_states = {JobStateEnum.CANCELLED}

        return self._count_job_states(cancelled_job_states)

    def remaining_jobs(self):
        terminal_jobs = self.completed_jobs() + self.failed_jobs() + self.cancelled_jobs()

        return len(self._job_states) - terminal_jobs
