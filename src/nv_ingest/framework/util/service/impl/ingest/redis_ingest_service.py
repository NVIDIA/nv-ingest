# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
from json import JSONDecodeError
from typing import Optional, Dict, Any, List

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
        # Bound async-to-thread concurrency slightly below Redis connection pool
        self._async_operation_semaphore: Optional[asyncio.Semaphore] = None

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

    def _get_async_semaphore(self) -> asyncio.Semaphore:
        if self._async_operation_semaphore is None:
            semaphore_limit = max(1, self._concurrency_level - 2)
            self._async_operation_semaphore = asyncio.Semaphore(semaphore_limit)
        return self._async_operation_semaphore

    async def _run_bounded_to_thread(self, func, *args, **kwargs):
        async with self._get_async_semaphore():
            return await asyncio.to_thread(func, *args, **kwargs)

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
            # Determine target queue based on optional QoS hint
            queue_hint = None
            try:
                routing_opts = job_spec.get("routing_options") or {}
                tracing_opts = job_spec.get("tracing_options") or {}
                queue_hint = routing_opts.get("queue_hint") or tracing_opts.get("queue_hint")
            except Exception:
                queue_hint = None
            allowed = {"default", "immediate", "micro", "small", "medium", "large"}
            if isinstance(queue_hint, str) and queue_hint in allowed:
                if queue_hint == "default":
                    channel_name = self._redis_task_queue
                else:
                    channel_name = f"{self._redis_task_queue}_{queue_hint}"
            else:
                channel_name = self._redis_task_queue
            logger.debug(
                f"Submitting job {trace_id} to queue '{channel_name}' (hint={queue_hint}) "
                f"with result TTL: {ttl_for_result}"
            )

            logger.debug(
                f"Submitting job {trace_id} to queue '{self._redis_task_queue}' with result TTL: {ttl_for_result}"
            )
            await self._run_bounded_to_thread(
                self._ingest_client.submit_message,
                channel_name=channel_name,
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
            message = await self._run_bounded_to_thread(
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
            logger.debug(f"Fetch operation for job {job_id} did not complete: ({type(e).__name__}) {e}")
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
            await self._run_bounded_to_thread(
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
            data_bytes: Optional[bytes] = await self._run_bounded_to_thread(
                self._ingest_client.get_client().get,
                state_key,
            )
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
            await self._run_bounded_to_thread(
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
            data_bytes: Optional[bytes] = await self._run_bounded_to_thread(
                self._ingest_client.get_client().get,
                cache_key,
            )
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

    async def set_parent_job_mapping(
        self,
        parent_job_id: str,
        subjob_ids: List[str],
        metadata: Dict[str, Any],
        *,
        subjob_descriptors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Store parent-subjob mapping in Redis for V2 PDF splitting.

        Parameters
        ----------
        parent_job_id : str
            The parent job identifier
        subjob_ids : List[str]
            List of subjob identifiers
        metadata : Dict[str, Any]
            Metadata about the parent job (total_pages, original_source_id, etc.)
        subjob_descriptors : List[Dict[str, Any]], optional
            Detailed descriptors (job_id, chunk_index, start/end pages) for subjobs
        """
        parent_key = f"parent:{parent_job_id}:subjobs"
        metadata_key = f"parent:{parent_job_id}:metadata"

        try:
            # Store subjob IDs as a set (only if there are subjobs)
            if subjob_ids:
                await self._run_bounded_to_thread(
                    self._ingest_client.get_client().sadd,
                    parent_key,
                    *subjob_ids,
                )

            # Store metadata as hash (including original subjob ordering for deterministic fetches)
            metadata_to_store = dict(metadata)
            try:
                metadata_to_store["subjob_order"] = json.dumps(subjob_ids)
            except (TypeError, ValueError):
                logger.warning(
                    "Unable to serialize subjob ordering for parent %s; falling back to Redis set ordering",
                    parent_job_id,
                )
                metadata_to_store.pop("subjob_order", None)

            if subjob_descriptors:
                metadata_to_store["subjob_descriptors"] = json.dumps(subjob_descriptors)

            await self._run_bounded_to_thread(
                self._ingest_client.get_client().hset,
                metadata_key,
                mapping=metadata_to_store,
            )

            # Set TTL on both keys to match state TTL
            if self._state_ttl_seconds:
                await self._run_bounded_to_thread(
                    self._ingest_client.get_client().expire,
                    parent_key,
                    self._state_ttl_seconds,
                )
                await self._run_bounded_to_thread(
                    self._ingest_client.get_client().expire,
                    metadata_key,
                    self._state_ttl_seconds,
                )

            logger.debug(f"Stored parent job mapping for {parent_job_id} with {len(subjob_ids)} subjobs")

        except Exception as err:
            logger.exception(f"Error storing parent job mapping for {parent_job_id}: {err}")
            raise

    async def get_parent_job_info(self, parent_job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve parent job information including subjob IDs and metadata.

        Parameters
        ----------
        parent_job_id : str
            The parent job identifier

        Returns
        -------
        Dict[str, Any] or None
            Dictionary with 'subjob_ids' and 'metadata' keys, or None if not a parent job
        """
        parent_key = f"parent:{parent_job_id}:subjobs"
        metadata_key = f"parent:{parent_job_id}:metadata"

        try:
            # Check if this is a parent job (check metadata_key since non-split PDFs may not have parent_key)
            exists = await self._run_bounded_to_thread(
                self._ingest_client.get_client().exists,
                metadata_key,  # Check metadata instead of parent_key for non-split PDF support
            )

            if not exists:
                return None

            # Get subjob IDs (may be empty for non-split PDFs)
            subjob_ids_bytes = await self._run_bounded_to_thread(
                self._ingest_client.get_client().smembers,
                parent_key,
            )
            subjob_id_set = {id.decode("utf-8") for id in subjob_ids_bytes} if subjob_ids_bytes else set()

            # Get metadata
            metadata_dict = await self._run_bounded_to_thread(
                self._ingest_client.get_client().hgetall,
                metadata_key,
            )
            metadata = {k.decode("utf-8"): v.decode("utf-8") for k, v in metadata_dict.items()}

            # Convert numeric strings back to numbers
            if "total_pages" in metadata:
                metadata["total_pages"] = int(metadata["total_pages"])
            if "pages_per_chunk" in metadata:
                try:
                    metadata["pages_per_chunk"] = int(metadata["pages_per_chunk"])
                except ValueError:
                    metadata.pop("pages_per_chunk", None)

            ordered_ids: Optional[List[str]] = None
            stored_order = metadata.pop("subjob_order", None)
            if stored_order:
                try:
                    candidate_order = json.loads(stored_order)
                    if isinstance(candidate_order, list):
                        ordered_ids = [sid for sid in candidate_order if sid in subjob_id_set]
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "Failed to parse stored subjob order for parent %s: %s",
                        parent_job_id,
                        exc,
                    )

            if ordered_ids is None:
                ordered_ids = sorted(subjob_id_set)
            else:
                remaining_ids = sorted(subjob_id_set - set(ordered_ids))
                ordered_ids.extend(remaining_ids)

            subjob_descriptors: Optional[List[Dict[str, Any]]] = None
            stored_descriptors = metadata.pop("subjob_descriptors", None)
            if stored_descriptors:
                try:
                    decoded = json.loads(stored_descriptors)
                    if isinstance(decoded, list):
                        subjob_descriptors = decoded
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "Failed to parse stored subjob descriptors for parent %s: %s",
                        parent_job_id,
                        exc,
                    )

            return {
                "subjob_ids": ordered_ids,
                "metadata": metadata,
                "subjob_descriptors": subjob_descriptors or [],
            }

        except Exception as err:
            logger.error(f"Error retrieving parent job info for {parent_job_id}: {err}")
            return None
