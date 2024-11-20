# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
import os
import uuid
from json import JSONDecodeError
from typing import Any

from typing import List
from nv_ingest.schemas import validate_ingest_job
from nv_ingest.schemas.message_wrapper_schema import MessageWrapper
from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest.util.message_brokers.redis.redis_client import RedisClient
from nv_ingest.schemas.processing_job_schema import ProcessingJob

logger = logging.getLogger("uvicorn")


class RedisIngestService(IngestServiceMeta):
    """Submits Jobs to via Redis"""

    _concurrency_level = os.getenv("CONCURRENCY_LEVEL", 10)
    _client_kwargs = "{}"
    __shared_instance = None

    _pending_jobs = []

    @staticmethod
    def getInstance():
        """Static Access Method"""
        if RedisIngestService.__shared_instance is None:
            redis_host = os.getenv("MESSAGE_CLIENT_HOST", "localhost")
            redis_port = os.getenv("MESSAGE_CLIENT_PORT", "6379")
            redis_task_queue = os.getenv("REDIS_MORPHEUS_TASK_QUEUE", "morpheus_task_queue")
            RedisIngestService.__shared_instance = RedisIngestService(redis_host, redis_port, redis_task_queue)

        return RedisIngestService.__shared_instance

    def __init__(self, redis_hostname: str, redis_port: int, redis_task_queue: str):
        self._redis_hostname = redis_hostname
        self._redis_port = redis_port
        self._redis_task_queue = redis_task_queue
        self._cache_prefix = "processing_cache:"
        self._bulk_vdb_cache_prefix = "vdb_bulk_upload_cache:"

        self._ingest_client = RedisClient(host=self._redis_hostname, port=self._redis_port,
                                          max_pool_size=self._concurrency_level)

    async def submit_job(self, job_spec: MessageWrapper) -> str:
        try:
            json_data = job_spec.dict()["payload"]
            job_spec = json.loads(json_data)
            validate_ingest_job(job_spec)

            job_id = str(uuid.uuid4())
            job_spec["job_id"] = job_id

            self._ingest_client.submit_message(self._redis_task_queue, json.dumps(job_spec))

            return job_id

        except JSONDecodeError as err:
            logger.error("Error: %s", err)
            raise

        except Exception as err:
            logger.error("Error: %s", err)
            raise

    async def fetch_job(self, job_id: str) -> Any:
        # Fetch message with a timeout
        message = self._ingest_client.fetch_message(f"response_{job_id}", timeout=5)
        if message is None:
            raise TimeoutError()

        return message
    
    async def set_processing_cache(self, job_id: str, jobs_data: List[ProcessingJob]) -> None:
        """Store processing jobs data using simple key-value"""
        cache_key = f"{self._cache_prefix}{job_id}"
        try:
            self._ingest_client.get_client().set(
                cache_key,
                json.dumps([job.dict() for job in jobs_data]),
                ex=3600
            )
        except Exception as err:
            logger.error(f"Error setting cache for {cache_key}: {err}")
            raise
    
    async def get_processing_cache(self, job_id: str) -> List[ProcessingJob]:
        """Retrieve processing jobs data using simple key-value"""
        cache_key = f"{self._cache_prefix}{job_id}"
        try:
            data = self._ingest_client.get_client().get(cache_key)
            if data is None:
                return []
            return [ProcessingJob(**job) for job in json.loads(data)]
        except Exception as err:
            logger.error(f"Error getting cache for {cache_key}: {err}")
            raise


    async def set_vdb_bulk_upload_status(self, job_id: str, task_id: str):
        """Set the task_id for the vdb upload task"""
        cache_key = f"{self._bulk_vdb_cache_prefix}{job_id}"
        try:
            self._ingest_client.get_client().set(
                cache_key,
                task_id,
                ex=3600
            )
        except Exception as err:
            logger.error(f"Error setting cache for {cache_key}: {err}")
            raise


    async def get_vdb_bulk_upload_status(self, job_id: str) -> str:
        """Get the task_id for the VDB upload task to query Milvus for status"""
        cache_key = f"{self._bulk_vdb_cache_prefix}{job_id}"
        try:
            data = self._ingest_client.get_client().get(cache_key)
            return data
        except Exception as err:
            logger.error(f"Error getting cache for {cache_key}: {err}")
            raise
