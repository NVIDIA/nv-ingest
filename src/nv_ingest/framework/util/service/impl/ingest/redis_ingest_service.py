# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
from json import JSONDecodeError
from typing import Optional, Dict, Any

from typing import List

import redis

from nv_ingest.framework.schemas.framework_message_wrapper_schema import MessageWrapper
from nv_ingest.framework.schemas.framework_processing_job_schema import ProcessingJob
from nv_ingest.framework.util.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import validate_ingest_job
from nv_ingest_api.util.service_clients.client_base import FetchMode
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

logger = logging.getLogger("uvicorn")


def get_fetch_mode_from_env() -> "FetchMode":
    """
    Retrieves the fetch mode from the environment variable FETCH_MODE.

    Returns
    -------
    FetchMode
        The fetch mode as specified by the environment variable, or NON_DESTRUCTIVE by default.
    """
    mode_str: str = os.getenv("FETCH_MODE", "NON_DESTRUCTIVE").upper()
    try:
        return FetchMode[mode_str]
    except KeyError:
        logger.warning(f"Invalid FETCH_MODE '{mode_str}' in environment. Defaulting to DESTRUCTIVE.")
        return FetchMode.DESTRUCTIVE


class RedisIngestService(IngestServiceMeta):
    """
    Submits jobs and fetches results via Redis, supporting multiple fetch modes
    and state management with TTLs. Operates asynchronously using asyncio.to_thread
    for synchronous Redis client operations.
    """

    _concurrency_level: int = int(os.getenv("CONCURRENCY_LEVEL", "10"))
    __shared_instance: Optional["RedisIngestService"] = None

    @staticmethod
    def get_instance() -> "RedisIngestService":
        """
        Static access method implementing the Singleton pattern.

        Returns
        -------
        RedisIngestService
            The singleton instance of the RedisIngestService.
        """
        if RedisIngestService.__shared_instance is None:
            redis_host: str = os.getenv("MESSAGE_CLIENT_HOST", "localhost")
            redis_port: int = int(os.getenv("MESSAGE_CLIENT_PORT", "6379"))
            redis_task_queue: str = os.getenv("REDIS_INGEST_TASK_QUEUE", "ingest_task_queue")

            fetch_mode: "FetchMode" = get_fetch_mode_from_env()
            result_data_ttl: int = int(os.getenv("RESULT_DATA_TTL_SECONDS", "3600"))
            state_ttl: int = int(os.getenv("STATE_TTL_SECONDS", "7200"))

            cache_config: Dict[str, Any] = {
                "directory": os.getenv("FETCH_CACHE_DIR", "./.fetch_cache"),
                "ttl": int(os.getenv("FETCH_CACHE_TTL_SECONDS", "3600")),
            }
            use_ssl: bool = os.getenv("REDIS_USE_SSL", "false").lower() == "true"

            RedisIngestService.__shared_instance = RedisIngestService(
                redis_hostname=redis_host,
                redis_port=redis_port,
                redis_task_queue=redis_task_queue,
                fetch_mode=fetch_mode,
                result_data_ttl_seconds=result_data_ttl if result_data_ttl > 0 else None,
                state_ttl_seconds=state_ttl if state_ttl > 0 else None,
                cache_config=cache_config,
                use_ssl=use_ssl,
            )
            logger.debug(f"RedisIngestService configured with FetchMode: {fetch_mode.name}")
        else:
            logger.debug("Returning existing RedisIngestService Singleton instance.")
        return RedisIngestService.__shared_instance

    def __init__(
        self,
        redis_hostname: str,
        redis_port: int,
        redis_task_queue: str,
        fetch_mode: "FetchMode",
        result_data_ttl_seconds: Optional[int],
        state_ttl_seconds: Optional[int],
        cache_config: Optional[Dict[str, Any]],
        use_ssl: bool,
    ) -> None:
        """
        Initializes the service and the underlying RedisClient.

        Parameters
        ----------
        redis_hostname : str
            Redis server hostname.
        redis_port : int
            Redis server port.
        redis_task_queue : str
            The Redis queue name for tasks.
        fetch_mode : FetchMode
            The fetch mode configuration.
        result_data_ttl_seconds : int or None
            TTL for result data in seconds, or None to disable.
        state_ttl_seconds : int or None
            TTL for the job state record, or None to disable.
        cache_config : dict or None
            Configuration for caching.
        use_ssl : bool
            Whether to use SSL for the Redis connection.
        """
        self._redis_hostname: str = redis_hostname
        self._redis_port: int = redis_port
        self._redis_task_queue: str = redis_task_queue
        self._fetch_mode: "FetchMode" = fetch_mode
        self._result_data_ttl_seconds: Optional[int] = result_data_ttl_seconds
        self._state_ttl_seconds: Optional[int] = state_ttl_seconds

        self._bulk_vdb_cache_prefix: str = "vdb_bulk_upload_cache:"
        self._cache_prefix: str = "processing_cache:"
        self._state_prefix: str = "job_state:"

        self._ingest_client = RedisClient(
            host=self._redis_hostname,
            port=self._redis_port,
            max_pool_size=self._concurrency_level,
            fetch_mode=self._fetch_mode,
            cache_config=cache_config,
            message_ttl_seconds=self._result_data_ttl_seconds,
            use_ssl=use_ssl,
            max_retries=int(os.getenv("REDIS_MAX_RETRIES", "3")),
            max_backoff=int(os.getenv("REDIS_MAX_BACKOFF", "32")),
            connection_timeout=int(os.getenv("REDIS_CONNECTION_TIMEOUT", "300")),
        )
        logger.debug(
            f"RedisClient initialized for service. Host: {redis_hostname}:{redis_port}, "
            f"FetchMode: {fetch_mode.name}, ResultTTL: {result_data_ttl_seconds}, StateTTL: {state_ttl_seconds}"
        )

    async def submit_job(self, job_spec_wrapper: "MessageWrapper", trace_id: str) -> str:
        """
        Validates, prepares, and submits a job specification to the Redis task queue.
        Sets result data TTL if configured for NON_DESTRUCTIVE mode.

        Parameters
        ----------
        job_spec_wrapper : MessageWrapper
            A wrapper containing the job specification payload.
        trace_id : str
            A unique identifier for the job.

        Returns
        -------
        str
            The job trace_id.

        Raises
        ------
        ValueError
            If the payload is missing or invalid.
        JSONDecodeError, TypeError
            For payload parsing errors.
        RedisError, ConnectionError
            For Redis-related errors.
        """
        try:
            json_data = job_spec_wrapper.model_dump(mode="json").get("payload")
            if not json_data:
                raise ValueError("MessageWrapper payload is missing or empty.")
            if isinstance(json_data, str):
                job_spec = json.loads(json_data)
            elif isinstance(json_data, dict):
                job_spec = json_data
            else:
                raise TypeError(f"Unexpected payload type: {type(json_data)}")

            validate_ingest_job(job_spec)
            job_spec["job_id"] = trace_id
            tasks = job_spec.get("tasks", [])
            updated_tasks = []
            for task in tasks:
                task_prop = task.get("task_properties", {})
                if hasattr(task_prop, "model_dump") and callable(task_prop.model_dump):
                    task["task_properties"] = task_prop.model_dump(mode="json")
                elif not isinstance(task_prop, dict):
                    try:
                        task["task_properties"] = dict(task_prop)
                    except (TypeError, ValueError):
                        logger.error(f"Cannot convert task_properties to dict: {task_prop}. Skipping properties.")
                        task["task_properties"] = {}
                updated_tasks.append(task)
            job_spec["tasks"] = updated_tasks
            job_spec_json = json.dumps(job_spec)
            ttl_for_result: Optional[int] = (
                self._result_data_ttl_seconds if self._fetch_mode == FetchMode.NON_DESTRUCTIVE else None
            )
            logger.debug(
                f"Submitting job {trace_id} to queue '{self._redis_task_queue}' with result TTL: {ttl_for_result}"
            )
            await asyncio.to_thread(
                self._ingest_client.submit_message,
                channel_name=self._redis_task_queue,
                message=job_spec_json,
                ttl_seconds=ttl_for_result,
            )
            logger.debug(f"Successfully submitted job {trace_id}")
            return trace_id
        except (JSONDecodeError, TypeError, ValueError) as err:
            logger.exception(f"Data validation or parsing error for job {trace_id}: {err}")
            raise ValueError(f"Invalid job specification: {err}") from err
        except (redis.RedisError, ConnectionError) as err:
            logger.exception(f"Redis error submitting job {trace_id}: {err}")
            raise err
        except Exception as err:
            logger.exception(f"Unexpected error submitting job {trace_id}: {err}")
            raise

    async def fetch_job(self, job_id: str) -> Optional[Dict]:
        """
        Fetches the job result using the configured RedisClient fetch mode and timeout.
        Executes the synchronous client call asynchronously.

        Parameters
        ----------
        job_id : str
            The unique identifier of the job.

        Returns
        -------
        dict or None
            The job result message.

        Raises
        ------
        TimeoutError, RedisError, ConnectionError, ValueError, RuntimeError
            If the fetch operation fails.
        """
        try:
            result_channel: str = f"{job_id}"
            logger.debug(f"Attempting to fetch job result for {job_id} using mode {self._fetch_mode.name}")
            message = await asyncio.to_thread(
                self._ingest_client.fetch_message,
                channel_name=result_channel,
                timeout=10,
            )
            if message is not None:
                logger.debug(f"Successfully fetched result for job {job_id}.")
                return message
            else:
                logger.warning(f"fetch_message for {job_id} returned None unexpectedly.")
                raise TimeoutError("No data found (unexpected None response).")
        except (TimeoutError, redis.RedisError, ConnectionError, ValueError, RuntimeError) as e:
            logger.info(f"Fetch operation for job {job_id} did not complete: ({type(e).__name__}) {e}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error during async fetch_job for {job_id}: {e}")
            raise RuntimeError(f"Unexpected error fetching job {job_id}") from e

    async def set_job_state(self, job_id: str, state: str) -> None:
        """
        Sets the explicit state of a job and refreshes its TTL.

        Parameters
        ----------
        job_id : str
            The unique identifier of the job.
        state : str
            The state to be assigned to the job.

        Returns
        -------
        None
        """
        state_key: str = f"{self._state_prefix}{job_id}"
        ttl_to_set: Optional[int] = self._state_ttl_seconds
        try:
            logger.debug(f"Setting state for {job_id} to {state} with TTL {ttl_to_set}")
            await asyncio.to_thread(
                self._ingest_client.get_client().set,
                state_key,
                state,
                ex=ttl_to_set,
            )
            logger.debug(f"Successfully set state for {job_id}.")
        except (redis.RedisError, ConnectionError) as err:
            logger.error(f"Failed to set state for {state_key}: {err}")
        except Exception as err:
            logger.exception(f"Unexpected error setting state for {state_key}: {err}")

    async def get_job_state(self, job_id: str) -> Optional[str]:
        """
        Retrieves the explicit state of a job.

        Parameters
        ----------
        job_id : str
            The unique identifier of the job.

        Returns
        -------
        str or None
            The state of the job, or None if not found or upon error.
        """
        state_key: str = f"{self._state_prefix}{job_id}"
        try:
            data_bytes: Optional[bytes] = await asyncio.to_thread(self._ingest_client.get_client().get, state_key)
            if data_bytes:
                state: str = data_bytes.decode("utf-8")
                logger.debug(f"Retrieved state for {job_id}: {state}")
                return state
            else:
                logger.debug(f"No state found for {job_id} (key: {state_key})")
                return None
        except (redis.RedisError, ConnectionError) as err:
            logger.error(f"Redis error getting state for {state_key}: {err}")
            return None
        except Exception as err:
            logger.exception(f"Unexpected error getting state for {state_key}: {err}")
            return None

    async def set_processing_cache(self, job_id: str, jobs_data: List["ProcessingJob"]) -> None:
        """
        Stores processing jobs data in a simple key-value cache.

        Parameters
        ----------
        job_id : str
            The unique identifier of the job.
        jobs_data : list of ProcessingJob
            The processing job data to be cached.

        Returns
        -------
        None
        """
        cache_key: str = f"{self._cache_prefix}{job_id}"
        try:
            data_to_store: str = json.dumps([job.model_dump(mode="json") for job in jobs_data])
            await asyncio.to_thread(
                self._ingest_client.get_client().set,
                cache_key,
                data_to_store,
                ex=3600,
            )
        except Exception as err:
            logger.exception(f"Error setting cache for {cache_key}: {err}")

    async def get_processing_cache(self, job_id: str) -> List["ProcessingJob"]:
        """
        Retrieves processing jobs data from the simple key-value cache.

        Parameters
        ----------
        job_id : str
            The unique identifier of the job.

        Returns
        -------
        list of ProcessingJob
            A list of processing jobs, or an empty list if not found or upon error.
        """
        cache_key: str = f"{self._cache_prefix}{job_id}"
        try:
            data_bytes: Optional[bytes] = await asyncio.to_thread(self._ingest_client.get_client().get, cache_key)
            if data_bytes is None:
                return []
            return [ProcessingJob(**job) for job in json.loads(data_bytes)]
        except Exception as err:
            logger.exception(f"Error getting cache for {cache_key}: {err}")
            return []

    async def get_fetch_mode(self) -> "FetchMode":
        """
        Returns the configured fetch mode for the service.

        Returns
        -------
        FetchMode
            The current fetch mode.
        """
        return self._fetch_mode
