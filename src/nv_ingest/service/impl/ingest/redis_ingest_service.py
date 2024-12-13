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
from json import JSONDecodeError
from typing import Any

from nv_ingest.schemas import validate_ingest_job
from nv_ingest.schemas.message_wrapper_schema import MessageWrapper
from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest.util.message_brokers.redis.redis_client import RedisClient

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

        self._ingest_client = RedisClient(
            host=self._redis_hostname, port=self._redis_port, max_pool_size=self._concurrency_level
        )

    async def submit_job(self, job_spec: MessageWrapper, trace_id: str) -> str:
        try:
            json_data = job_spec.dict()["payload"]
            job_spec = json.loads(json_data)
            validate_ingest_job(job_spec)

            job_spec["job_id"] = trace_id

            self._ingest_client.submit_message(self._redis_task_queue, json.dumps(job_spec))

            return trace_id

        except JSONDecodeError as err:
            logger.error("Error: %s", err)
            raise

        except Exception as err:
            logger.error("Error: %s", err)
            raise

    async def fetch_job(self, job_id: str) -> Any:
        # Fetch message with a timeout
        message = self._ingest_client.fetch_message(f"{job_id}", timeout=5)
        if message is None:
            raise TimeoutError()

        return message
