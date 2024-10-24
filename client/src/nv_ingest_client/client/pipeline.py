# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List
from typing import Optional

from nv_ingest_client.client.client import NvIngestClient
from nv_ingest_client.primitives import BatchJobSpec
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
        self, documents: List[str], client: Optional[NvIngestClient] = None, job_queue_id=DEFAULT_JOB_QUEUE_ID,
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
        self._client = NvIngestClient()

    def run(self):
        self._job_ids = self._client.add_job(self._job_specs)
        self._job_states = self._client.submit_job(self._job_ids, self._job_queue_id)

        result = self._client.fetch_job_result(self._job_ids)

        return result

    def run_async(self):
        self._job_ids = self._client.add_job(self._job_specs)

        futures = self._client.submit_job_async(self._job_ids, self._job_queue_id)

        return futures

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
