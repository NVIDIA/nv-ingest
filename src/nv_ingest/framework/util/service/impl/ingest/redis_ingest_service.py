# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
from json import JSONDecodeError
from typing import Optional, Dict

from typing import List

from redis import RedisError

from nv_ingest.framework.schemas.framework_message_wrapper_schema import MessageWrapper
from nv_ingest.framework.schemas.framework_processing_job_schema import ProcessingJob
from nv_ingest.framework.util.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import validate_ingest_job
from nv_ingest_api.util.service_clients.client_base import FetchMode
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

logger = logging.getLogger("uvicorn")


# Helper to parse FetchMode from env var
def get_fetch_mode_from_env() -> FetchMode:
    mode_str = os.getenv("FETCH_MODE", "NON_DESTRUCTIVE").upper()
    try:
        return FetchMode[mode_str]
    except KeyError:
        logger.warning(f"Invalid FETCH_MODE '{mode_str}' in environment. Defaulting to DESTRUCTIVE.")
        return FetchMode.DESTRUCTIVE


class RedisIngestService(IngestServiceMeta):
    """
    Submits Jobs and fetches results via Redis, supporting multiple fetch modes
    and state management with TTLs. Operates asynchronously using asyncio.to_thread
    for synchronous Redis client operations.
    """

    _concurrency_level = int(os.getenv("CONCURRENCY_LEVEL", "10"))  # Ensure int
    __shared_instance = None

    @staticmethod
    def getInstance():
        """Static Access Method implementing Singleton pattern."""
        if RedisIngestService.__shared_instance is None:
            redis_host = os.getenv("MESSAGE_CLIENT_HOST", "localhost")
            redis_port = int(os.getenv("MESSAGE_CLIENT_PORT", "6379"))  # Ensure int
            redis_task_queue = os.getenv("REDIS_MORPHEUS_TASK_QUEUE", "morpheus_task_queue")

            # --- New Configuration via Environment Variables ---
            fetch_mode = get_fetch_mode_from_env()
            # TTL for the actual result data in Redis (used for NON_DESTRUCTIVE)
            # Set to 0 or None to disable TTL
            result_data_ttl = int(os.getenv("RESULT_DATA_TTL_SECONDS", "3600"))  # 1 hour default
            # TTL for the job state record itself (should be > result_data_ttl)
            # Set to 0 or None to disable TTL
            state_ttl = int(os.getenv("STATE_TTL_SECONDS", "86400"))  # 24 hours default

            # Basic Cache Config example (can be expanded)
            cache_config = {
                "directory": os.getenv("FETCH_CACHE_DIR", "./.fetch_cache"),
                "ttl": int(os.getenv("FETCH_CACHE_TTL_SECONDS", "3600")),  # 1 hour default for cache
            }
            use_ssl = os.getenv("REDIS_USE_SSL", "false").lower() == "true"

            RedisIngestService.__shared_instance = RedisIngestService(
                redis_hostname=redis_host,
                redis_port=redis_port,
                redis_task_queue=redis_task_queue,
                fetch_mode=fetch_mode,
                result_data_ttl_seconds=result_data_ttl if result_data_ttl > 0 else None,
                state_ttl_seconds=state_ttl if state_ttl > 0 else None,
                cache_config=cache_config,  # Pass cache config dict
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
        fetch_mode: FetchMode,
        result_data_ttl_seconds: Optional[int],
        state_ttl_seconds: Optional[int],
        cache_config: Optional[dict],
        use_ssl: bool,
    ):
        """Initializes the service and the underlying RedisClient."""
        self._redis_hostname = redis_hostname
        self._redis_port = redis_port
        self._redis_task_queue = redis_task_queue
        self._fetch_mode = fetch_mode
        self._result_data_ttl_seconds = result_data_ttl_seconds
        self._state_ttl_seconds = state_ttl_seconds

        # Prefixes for keys
        self._bulk_vdb_cache_prefix = "vdb_bulk_upload_cache:"
        self._cache_prefix = "processing_cache:"
        self._state_prefix = "job_state:"

        # Instantiate the updated RedisClient
        self._ingest_client = RedisClient(
            host=self._redis_hostname,
            port=self._redis_port,
            max_pool_size=self._concurrency_level,
            fetch_mode=self._fetch_mode,
            cache_config=cache_config,
            message_ttl_seconds=self._result_data_ttl_seconds,  # Pass result TTL here
            use_ssl=use_ssl,
            # Pass other relevant params like max_retries, backoff if needed via env/config
            max_retries=int(os.getenv("REDIS_MAX_RETRIES", "3")),
            max_backoff=int(os.getenv("REDIS_MAX_BACKOFF", "32")),
            connection_timeout=int(os.getenv("REDIS_CONNECTION_TIMEOUT", "300")),
        )
        logger.debug(
            f"RedisClient initialized for service. Host: {redis_hostname}:{redis_port}, "
            f"FetchMode: {fetch_mode.name}, ResultTTL: {result_data_ttl_seconds}, StateTTL: {state_ttl_seconds}"
        )

    async def submit_job(self, job_spec_wrapper: MessageWrapper, trace_id: str) -> str:
        """
        Validates, prepares, and submits a job spec to the Redis task queue.
        Sets result data TTL if configured for NON_DESTRUCTIVE mode.
        """
        try:
            # Extract, parse, and validate the job spec payload
            json_data = job_spec_wrapper.model_dump(mode="json").get("payload")
            if not json_data:
                raise ValueError("MessageWrapper payload is missing or empty.")

            # The payload might already be a dict if model_dump worked as expected,
            # or it might be a JSON string. Handle both.
            if isinstance(json_data, str):
                job_spec = json.loads(json_data)
            elif isinstance(json_data, dict):
                job_spec = json_data  # Already a dict
            else:
                raise TypeError(f"Unexpected payload type: {type(json_data)}")

            validate_ingest_job(job_spec)

            job_spec["job_id"] = trace_id
            tasks = job_spec.get("tasks", [])
            updated_tasks = []
            for task in tasks:
                task_prop = task.get("task_properties", {})
                # Ensure task_properties is a plain dict for JSON serialization
                if hasattr(task_prop, "model_dump") and callable(task_prop.model_dump):
                    task["task_properties"] = task_prop.model_dump(mode="json")
                elif not isinstance(task_prop, dict):
                    # Attempt basic dict conversion if possible, otherwise log error
                    try:
                        task["task_properties"] = dict(task_prop)
                    except (TypeError, ValueError):
                        logger.error(f"Cannot convert task_properties to dict: {task_prop}. Skipping properties.")
                        task["task_properties"] = {}  # Set empty dict on failure
                # else: it's already a dict
                updated_tasks.append(task)
            job_spec["tasks"] = updated_tasks

            job_spec_json = json.dumps(job_spec)

            # Determine TTL for the result data based on fetch mode
            ttl_for_result = self._result_data_ttl_seconds if self._fetch_mode == FetchMode.NON_DESTRUCTIVE else None

            logger.debug(
                f"Submitting job {trace_id} to queue '{self._redis_task_queue}' " f"with result TTL: {ttl_for_result}"
            )

            # Execute the synchronous submit_message in a thread
            await asyncio.to_thread(
                self._ingest_client.submit_message,
                channel_name=self._redis_task_queue,
                message=job_spec_json,
                ttl_seconds=ttl_for_result,  # Pass the calculated TTL
            )

            logger.debug(f"Successfully submitted job {trace_id}")
            return trace_id

        except (JSONDecodeError, TypeError, ValueError) as err:
            logger.exception(f"Data validation or parsing error for job {trace_id}: {err}")
            # Re-raise as ValueError for endpoint handling? Or a custom exception?
            raise ValueError(f"Invalid job specification: {err}") from err
        except (RedisError, ConnectionError) as err:
            logger.exception(f"Redis error submitting job {trace_id}: {err}")
            # Re-raise for endpoint to handle as 5xx/503
            raise err
        except Exception as err:
            logger.exception(f"Unexpected error submitting job {trace_id}: {err}")
            raise  # Re-raise unexpected errors

    async def fetch_job(self, job_id: str) -> Optional[Dict]:
        """
        Fetches the job result using the configured RedisClient fetch mode and timeout.
        Wraps the synchronous client call asynchronously.
        """
        try:
            # The channel name for results is assumed to be the job_id
            result_channel = f"{job_id}"
            logger.debug(f"Attempting to fetch job result for {job_id} using mode {self._fetch_mode.name}")

            # Execute the synchronous fetch_message in a thread
            # Use a reasonable timeout for the fetch operation itself
            message = await asyncio.to_thread(
                self._ingest_client.fetch_message,
                channel_name=result_channel,
                timeout=10,  # Example timeout for the fetch operation
            )

            # Note: The updated RedisClient.fetch_message returns Dict or raises.
            # It should not return None on success anymore.
            if message is not None:  # Defensive check
                logger.debug(f"Successfully fetched result for job {job_id}.")
                return message
            else:
                # This case should ideally not be reached if fetch_message adheres
                # to the new contract (raises TimeoutError instead of returning None).
                logger.warning(f"fetch_message for {job_id} returned None unexpectedly.")
                raise TimeoutError("No data found (unexpected None response).")

        except (TimeoutError, RedisError, ConnectionError, ValueError, RuntimeError) as e:
            # Catch specific errors raised by the updated RedisClient.fetch_message
            # Log appropriately but re-raise for the endpoint handler
            logger.warning(f"Fetch operation for job {job_id} failed: ({type(e).__name__}) {e}")
            raise e  # Re-raise the original exception for the endpoint handler
        except Exception as e:
            # Catch any other unexpected errors
            logger.exception(f"Unexpected error during async fetch_job for {job_id}: {e}")
            # Wrap in a standard error type or re-raise
            raise RuntimeError(f"Unexpected error fetching job {job_id}") from e

    # --- State Management Methods ---

    async def set_job_state(self, job_id: str, state: str):
        """
        Sets the explicit state of a job and refreshes/sets its TTL using the
        configured state TTL. Executes synchronously in a thread.
        """
        state_key = f"{self._state_prefix}{job_id}"
        ttl_to_set = self._state_ttl_seconds  # Use configured state TTL
        try:
            logger.debug(f"Setting state for {job_id} to {state} with TTL {ttl_to_set}")
            # Use asyncio.to_thread for the synchronous Redis call
            await asyncio.to_thread(
                self._ingest_client.get_client().set,  # Target function
                state_key,  # Arg 1 for set
                state,  # Arg 2 for set
                ex=ttl_to_set,  # Keyword arg for set (expiry)
            )
            logger.debug(f"Successfully set state for {job_id}.")
        except (RedisError, ConnectionError) as err:
            # Log error but don't necessarily block caller unless critical
            logger.error(f"Failed to set state for {state_key}: {err}")
            # Optionally re-raise if state setting is critical path
            # raise err
        except Exception as err:
            logger.exception(f"Unexpected error setting state for {state_key}: {err}")
            # Optionally re-raise

    async def get_job_state(self, job_id: str) -> Optional[str]:
        """
        Gets the explicit state of a job. Executes synchronously in a thread.
        Returns None if the key doesn't exist or an error occurs.
        """
        state_key = f"{self._state_prefix}{job_id}"
        try:
            # Use asyncio.to_thread for the synchronous Redis call
            data_bytes = await asyncio.to_thread(
                self._ingest_client.get_client().get, state_key  # Target function  # Arg 1 for get
            )

            if data_bytes:
                state = data_bytes.decode("utf-8")
                logger.debug(f"Retrieved state for {job_id}: {state}")
                return state
            else:
                logger.debug(f"No state found for {job_id} (key: {state_key})")
                return None
        except (RedisError, ConnectionError) as err:
            logger.error(f"Redis error getting state for {state_key}: {err}")
            return None  # Return None on Redis errors for simplicity in endpoint
        except Exception as err:
            logger.exception(f"Unexpected error getting state for {state_key}: {err}")
            return None  # Return None on other errors

    # --- Cache Methods (Optional - keep or remove if not needed) ---
    async def set_processing_cache(self, job_id: str, jobs_data: List[ProcessingJob]) -> None:
        """Store processing jobs data using simple key-value (Async)."""
        cache_key = f"{self._cache_prefix}{job_id}"
        try:
            # Assuming ProcessingJob has .model_dump()
            data_to_store = json.dumps([job.model_dump(mode="json") for job in jobs_data])
            # Use asyncio.to_thread
            await asyncio.to_thread(
                self._ingest_client.get_client().set,
                cache_key,
                data_to_store,
                ex=3600,  # Keep fixed TTL or make configurable?
            )
        except Exception as err:
            logger.exception(f"Error setting cache for {cache_key}: {err}")
            # Decide whether to raise or just log

    async def get_processing_cache(self, job_id: str) -> List[ProcessingJob]:
        """Retrieve processing jobs data using simple key-value (Async)."""
        cache_key = f"{self._cache_prefix}{job_id}"
        try:
            # Use asyncio.to_thread
            data_bytes = await asyncio.to_thread(self._ingest_client.get_client().get, cache_key)
            if data_bytes is None:
                return []
            # Assuming ProcessingJob can be reconstructed from dict
            return [ProcessingJob(**job) for job in json.loads(data_bytes)]
        except Exception as err:
            logger.exception(f"Error getting cache for {cache_key}: {err}")
            return []  # Return empty on error? Or raise?

    # --- Added Methods ---
    async def get_fetch_mode(self) -> FetchMode:
        """Returns the configured fetch mode for this service instance."""
        return self._fetch_mode
