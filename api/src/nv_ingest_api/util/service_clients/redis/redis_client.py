# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
import random
from typing import Any, Callable, Union
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import redis


from nv_ingest_api.util.service_clients.client_base import MessageBrokerClientBase, FetchMode

try:
    from diskcache import Cache

    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

# pylint: skip-file

logger = logging.getLogger(__name__)

# Default cache path and TTL (adjust as needed)
DEFAULT_CACHE_DIR = "/tmp/.fetch_cache"
DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour


class RedisClient(MessageBrokerClientBase):
    """
    A client for interfacing with Redis, providing mechanisms for sending and receiving messages
    with retry logic, connection management, configurable fetch modes, and optional local caching.

    Handles message fragmentation transparently during fetch operations.
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        max_retries: int = 3,
        max_backoff: int = 32,
        connection_timeout: int = 300,
        max_pool_size: int = 128,
        use_ssl: bool = False,
        redis_allocator: Callable[..., redis.Redis] = redis.Redis,
        fetch_mode: "FetchMode" = None,  # Replace with appropriate default if FetchMode.DESTRUCTIVE is available.
        cache_config: Optional[Dict[str, Any]] = None,
        message_ttl_seconds: Optional[int] = 600,
    ) -> None:
        """
        Initializes the Redis client with connection pooling, retry/backoff configuration,
        and optional caching for non-destructive or hybrid fetch modes.

        Parameters
        ----------
        host : str
            The Redis server hostname or IP address.
        port : int
            The Redis server port.
        db : int, optional
            The Redis logical database to use. Default is 0.
        max_retries : int, optional
            Maximum number of retries allowed for operations. Default is 3.
        max_backoff : int, optional
            Maximum backoff in seconds for retry delays. Default is 32.
        connection_timeout : int, optional
            Timeout in seconds for establishing a Redis connection. Default is 300.
        max_pool_size : int, optional
            Maximum size of the Redis connection pool. Default is 128.
        use_ssl : bool, optional
            Whether to use SSL for the connection. Default is False.
        redis_allocator : Callable[..., redis.Redis], optional
            Callable that returns a Redis client instance. Default is redis.Redis.
        fetch_mode : FetchMode, optional
            Fetch mode configuration (e.g., DESTRUCTIVE, NON_DESTRUCTIVE, CACHE_BEFORE_DELETE).
            Default should be set appropriately (e.g., FetchMode.DESTRUCTIVE).
        cache_config : dict, optional
            Configuration dictionary for local caching, e.g., {"directory": "/path/to/cache", "ttl": 7200}.
        message_ttl_seconds : int, optional
            TTL (in seconds) for messages in NON_DESTRUCTIVE mode. If not provided,
            messages may persist indefinitely.

        Returns
        -------
        None
        """
        self._host: str = host
        self._port: int = port
        self._db: int = db
        self._max_retries: int = max_retries
        self._max_backoff: int = max_backoff
        self._connection_timeout: int = connection_timeout
        self._use_ssl: bool = use_ssl  # TODO: Implement SSL specifics.
        # If no fetch_mode is provided, assume a default value.
        self._fetch_mode: "FetchMode" = fetch_mode if fetch_mode is not None else FetchMode.DESTRUCTIVE
        self._message_ttl_seconds: Optional[int] = message_ttl_seconds
        self._redis_allocator: Callable[..., redis.Redis] = redis_allocator

        if self._fetch_mode == FetchMode.NON_DESTRUCTIVE and message_ttl_seconds is None:
            logger.warning(
                "FetchMode.NON_DESTRUCTIVE selected without setting message_ttl_seconds. "
                "Messages fetched non-destructively may persist indefinitely in Redis."
            )

        # Configure Connection Pool
        pool_kwargs: Dict[str, Any] = {
            "host": self._host,
            "port": self._port,
            "db": self._db,
            "socket_connect_timeout": self._connection_timeout,
            "max_connections": max_pool_size,
        }
        if self._use_ssl:
            pool_kwargs["ssl"] = True
            pool_kwargs["ssl_cert_reqs"] = None  # Or specify requirements as needed.
            logger.debug("Redis connection configured with SSL.")

        self._pool: redis.ConnectionPool = redis.ConnectionPool(**pool_kwargs)

        # Allocate initial client
        self._client: Optional[redis.Redis] = self._redis_allocator(connection_pool=self._pool)

        # Configure Cache if mode requires it
        self._cache: Optional[Any] = None
        if self._fetch_mode == FetchMode.CACHE_BEFORE_DELETE and DISKCACHE_AVAILABLE:
            cache_dir: str = (cache_config or {}).get("directory", DEFAULT_CACHE_DIR)
            self._cache_ttl: int = (cache_config or {}).get("ttl", DEFAULT_CACHE_TTL_SECONDS)
            try:
                # TODO: make size_limit configurable
                self._cache = Cache(cache_dir, timeout=self._cache_ttl, size_limit=int(50e9))
                logger.debug(f"Fetch cache enabled: mode={self._fetch_mode}, dir={cache_dir}, ttl={self._cache_ttl}s")
            except Exception as e:
                logger.exception(f"Failed to initialize disk cache at {cache_dir}. Caching disabled. Error: {e}")
                self._fetch_mode = FetchMode.DESTRUCTIVE
                logger.warning("Falling back to FetchMode.DESTRUCTIVE due to cache init failure.")

        # Validate max_retries on init using setter
        self.max_retries = max_retries

    def _connect(self) -> None:
        """
        Attempts to reconnect to the Redis server by allocating a new client from the pool.

        Returns
        -------
        None

        Raises
        ------
        ConnectionError
            If the newly allocated client fails to respond to a ping.
        """
        logger.debug("Attempting to reconnect to Redis by re-allocating client.")
        try:
            self._client = self._redis_allocator(connection_pool=self._pool)
            if not self.ping():
                raise ConnectionError("Re-allocated client failed to ping.")
            logger.info("Successfully reconnected to Redis.")
        except Exception as e:
            logger.error(f"Failed to reconnect to Redis: {e}")
            self._client = None

    @property
    def max_retries(self) -> int:
        """
        Gets the maximum number of allowed retries for Redis operations.

        Returns
        -------
        int
            The maximum number of retries.
        """
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        """
        Sets the maximum number of allowed retries for Redis operations.

        Parameters
        ----------
        value : int
            The new maximum retries value; must be a non-negative integer.

        Raises
        ------
        ValueError
            If the value is not a non-negative integer.
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        self._max_retries = value

    def get_client(self) -> redis.Redis:
        """
        Returns a Redis client instance, attempting reconnection if the current client is invalid.

        Returns
        -------
        redis.Redis
            The active Redis client instance.

        Raises
        ------
        RuntimeError
            If no valid client can be established.
        """
        if self._client is None:
            logger.info("Redis client is None, attempting to connect.")
            try:
                self._connect()
            except Exception as connect_err:
                logger.error(f"Error during _connect attempt: {connect_err}")
                self._client = None

            if self._client is None:
                raise RuntimeError("Failed to establish or re-establish connection to Redis.")

        return self._client

    def ping(self) -> bool:
        """
        Checks if the Redis client connection is alive by issuing a PING command.

        Returns
        -------
        bool
            True if the ping is successful, False otherwise.
        """
        if self._client is None:
            logger.debug("Ping check: No client instance exists.")
            return False
        try:
            is_alive: bool = self._client.ping()
            if is_alive:
                logger.debug("Ping successful.")
                return True
            else:
                logger.warning("Ping command returned non-True value unexpectedly.")
                self._client = None
                return False
        except (OSError, AttributeError) as e:
            logger.warning(f"Ping failed, invalidating client connection: ({type(e).__name__}) {e}")
            self._client = None
            return False
        except redis.RedisError as e:
            logger.warning(f"Ping failed due to RedisError: {e}. Invalidating client.")
            self._client = None
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during ping, invalidating client: {e}")
            self._client = None
            return False

    def _check_response(
        self, channel_name: str, timeout: float
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[int]]:
        """
        Checks for a response from a Redis queue and processes it into a message and its fragmentation metadata.

        Parameters
        ----------
        channel_name : str
            The Redis channel from which to retrieve the response.
        timeout : float
            The time in seconds to wait for a response.

        Returns
        -------
        tuple of (Optional[Dict[str, Any]], Optional[int], Optional[int])
            - The decoded message as a dictionary, or None if not retrieved.
            - The fragment number (default 0 if absent), or None.
            - The total number of fragments, or None.

        Raises
        ------
        TimeoutError
            If no response is received within the specified timeout.
        ValueError
            If the message cannot be decoded from JSON.
        """
        response = self.get_client().blpop([channel_name], timeout)
        if response is None:
            raise TimeoutError("No response was received in the specified timeout period")

        if len(response) > 1 and response[1]:
            try:
                message = json.loads(response[1])
                fragment: int = message.get("fragment", 0)
                fragment_count: int = message.get("fragment_count", 1)
                return message, fragment, fragment_count
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")
                raise ValueError(f"Failed to decode message from Redis: {e}")

        return None, None, None

    def _fetch_first_or_all_fragments_destructive(
        self, channel_name: str, timeout: float
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fetches message fragments destructively using BLPOP, returning either a single message
        or a list of fragments if the message is split.

        Parameters
        ----------
        channel_name : str
            The Redis list key from which to pop the message.
        timeout : float
            The timeout in seconds for the BLPOP command.

        Returns
        -------
        dict or list of dict
            If the message is not fragmented, returns a single dictionary.
            If fragmented, returns a list of dictionaries representing each fragment.

        Raises
        ------
        TimeoutError
            If the initial BLPOP times out or if subsequent fragments are not retrieved within the allotted time.
        ValueError
            If JSON decoding fails or if fragment indices are inconsistent.
        """
        fragments: List[Dict[str, Any]] = []
        expected_count: int = 1
        first_message: Optional[Dict[str, Any]] = None
        accumulated_fetch_time: float = 0.0

        logger.debug(f"Destructive fetch: Popping first item from '{channel_name}' with timeout {timeout:.2f}s")
        start_pop_time: float = time.monotonic()
        response = self.get_client().blpop([channel_name], timeout=int(max(1, timeout)))
        fetch_duration: float = time.monotonic() - start_pop_time

        if response is None:
            logger.debug(f"BLPOP timed out on '{channel_name}', no message available.")
            raise TimeoutError("No message received within the initial timeout period")

        if len(response) > 1 and response[1]:
            message_bytes = response[1]
            try:
                first_message = json.loads(message_bytes)
                expected_count = first_message.get("fragment_count", 1)
                fragment_idx: int = first_message.get("fragment", 0)
                if expected_count == 1:
                    logger.debug(f"Fetched single (non-fragmented) message from '{channel_name}'.")
                    return first_message
                logger.info(
                    f"Fetched fragment {fragment_idx + 1}/{expected_count} from '{channel_name}'. "
                    f"Need to fetch remaining."
                )
                if fragment_idx != 0:
                    logger.error(
                        f"Expected first fragment (index 0) but got {fragment_idx} from '{channel_name}'. "
                        f"Aborting fetch."
                    )
                    raise ValueError(f"First fragment fetched was index {fragment_idx}, expected 0.")
                fragments.append(first_message)
                accumulated_fetch_time += fetch_duration

                remaining_timeout: float = max(0.1, timeout - accumulated_fetch_time)
                for i in range(1, expected_count):
                    start_frag_pop_time: float = time.monotonic()
                    frag_timeout: float = max(1, remaining_timeout / max(1, expected_count - i))
                    logger.debug(f"Popping fragment {i + 1}/{expected_count} with timeout {frag_timeout:.2f}s")
                    frag_response = self.get_client().blpop([channel_name], timeout=int(frag_timeout))
                    frag_fetch_duration: float = time.monotonic() - start_frag_pop_time
                    accumulated_fetch_time += frag_fetch_duration
                    remaining_timeout = max(0, timeout - accumulated_fetch_time)
                    if frag_response is None:
                        logger.error(f"Timeout waiting for fragment {i + 1}/{expected_count} on '{channel_name}'.")
                        raise TimeoutError(f"Timeout collecting fragments for {channel_name}")
                    if len(frag_response) > 1 and frag_response[1]:
                        frag_bytes = frag_response[1]
                        try:
                            frag_message = json.loads(frag_bytes)
                            fragments.append(frag_message)
                        except json.JSONDecodeError as e_frag:
                            logger.error(
                                f"Failed to decode fragment {i + 1} JSON from '{channel_name}': {e_frag}. "
                                f"Data: {frag_bytes[:200]}"
                            )
                            raise ValueError(f"Failed to decode message fragment {i + 1}: {e_frag}")
                    else:
                        logger.error(
                            f"Unexpected BLPOP response format for fragment {i + 1} "
                            f"on '{channel_name}': {frag_response}"
                        )
                        raise ValueError(f"Unexpected BLPOP response format for fragment {i + 1}")
                logger.debug(f"Successfully fetched all {expected_count} fragments destructively.")
                return fragments
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode first message JSON from '{channel_name}': {e}. Data: {message_bytes[:200]}"
                )
                raise ValueError(f"Failed to decode first message: {e}") from e
        else:
            logger.warning(f"BLPOP for '{channel_name}' returned unexpected response format: {response}")
            raise ValueError("Unexpected response format from BLPOP")

    def _fetch_fragments_non_destructive(self, channel_name: str, timeout: float) -> List[Dict[str, Any]]:
        """
        Fetches all message fragments non-destructively by polling the Redis list. Uses LINDEX,
        LLEN, and LRANGE to collect fragments, respecting a total timeout.

        Parameters
        ----------
        channel_name : str
            The Redis list key where fragments are stored.
        timeout : float
            The total allowed time in seconds for collecting all fragments.

        Returns
        -------
        List[Dict[str, Any]]
            A list of unique fragment dictionaries.

        Raises
        ------
        TimeoutError
            If the overall timeout is exceeded before all expected fragments are collected.
        ValueError
            If JSON decoding fails or inconsistent fragment counts are detected.
        ConnectionError
            If the Redis connection fails.
        redis.RedisError
            For other Redis-related errors.
        """
        start_time: float = time.monotonic()
        polling_delay: float = 0.1
        expected_count: Optional[int] = None
        fragments_map: Dict[int, Dict[str, Any]] = {}

        logger.debug(f"Starting non-destructive fetch for '{channel_name}' with total timeout {timeout:.2f}s.")

        while True:
            current_time: float = time.monotonic()
            elapsed_time: float = current_time - start_time
            if elapsed_time > timeout:
                logger.debug(f"Overall timeout ({timeout}s) exceeded for non-destructive fetch of '{channel_name}'.")
                if expected_count:
                    raise TimeoutError(
                        f"Timeout collecting fragments for {channel_name}. "
                        f"Collected {len(fragments_map)}/{expected_count}."
                    )
                else:
                    raise TimeoutError(f"Timeout waiting for initial fragment 0 for {channel_name}.")

            client = self.get_client()
            try:
                if expected_count is None:
                    logger.debug(f"Polling for fragment 0 on '{channel_name}'. Elapsed: {elapsed_time:.2f}s")
                    frag0_bytes: Optional[bytes] = client.lindex(channel_name, 0)
                    if frag0_bytes is not None:
                        try:
                            message = json.loads(frag0_bytes)
                            fragment_idx: int = message.get("fragment", -1)
                            current_expected: int = message.get("fragment_count", 1)
                            if fragment_idx == 0:
                                logger.debug(
                                    f"Found fragment 0 for '{channel_name}'. "
                                    f"Expecting {current_expected} total fragments."
                                )
                                expected_count = current_expected
                                if fragment_idx not in fragments_map:
                                    fragments_map[fragment_idx] = message
                                if expected_count == 1:
                                    logger.debug("Single fragment expected and found. Fetch complete.")
                                    break
                            else:
                                logger.warning(
                                    f"Expected fragment 0 but found index {fragment_idx} "
                                    f"at LINDEX 0 for '{channel_name}'. List state potentially inconsistent. "
                                    f"Will keep polling."
                                )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to decode JSON at index 0 for '{channel_name}': {e}. Data: {frag0_bytes[:200]}"
                            )
                            raise ValueError(f"Failed to decode potential fragment 0: {e}")

                if expected_count is not None and len(fragments_map) < expected_count:
                    current_len: int = client.llen(channel_name)
                    logger.debug(
                        f"Polling '{channel_name}': Current length {current_len}, "
                        f"have {len(fragments_map)}/{expected_count} fragments. Elapsed: {elapsed_time:.2f}s"
                    )
                    if current_len >= expected_count:
                        fetch_end_index: int = expected_count - 1
                        logger.debug(f"Fetching full expected range: LRANGE 0 {fetch_end_index}")
                        raw_potential_fragments: List[bytes] = client.lrange(channel_name, 0, fetch_end_index)
                        processed_count_this_pass: int = 0
                        for item_bytes in raw_potential_fragments:
                            try:
                                message = json.loads(item_bytes)
                                fragment_idx: int = message.get("fragment", -1)
                                current_expected_in_frag: int = message.get("fragment_count", 1)
                                if current_expected_in_frag != expected_count:
                                    logger.error(
                                        f"Inconsistent fragment_count in fragment {fragment_idx} for '{channel_name}' "
                                        f"({current_expected_in_frag} vs expected {expected_count})."
                                    )
                                    raise ValueError("Inconsistent fragment count detected in list")
                                if 0 <= fragment_idx < expected_count and fragment_idx not in fragments_map:
                                    fragments_map[fragment_idx] = message
                                    processed_count_this_pass += 1
                                    logger.debug(f"Processed fragment {fragment_idx + 1}/{expected_count} from LRANGE.")
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Failed to decode JSON fragment during poll for "
                                    f"'{channel_name}': {e}. Data: {item_bytes[:200]}"
                                )
                                raise ValueError(f"Failed to decode message fragment: {e}")
                        if processed_count_this_pass > 0:
                            logger.debug(f"Found {processed_count_this_pass} new fragments this pass.")
                    if len(fragments_map) == expected_count:
                        logger.debug(f"Collected all {expected_count} expected fragments for '{channel_name}'.")
                        break
                if expected_count is None or len(fragments_map) < expected_count:
                    time.sleep(polling_delay)
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Validation or decoding error during non-destructive fetch for '{channel_name}': {e}")
                raise e
            except (redis.RedisError, ConnectionError) as e:
                logger.warning(
                    f"Redis/Connection error during non-destructive poll for '{channel_name}': {e}. Propagating up."
                )
                raise e
            except Exception as e:
                logger.exception(f"Unexpected error during non-destructive poll for '{channel_name}': {e}")
                raise RuntimeError(f"Unexpected polling error: {e}") from e

        if expected_count is None or len(fragments_map) != expected_count:
            logger.error(
                f"Exited non-destructive fetch loop for '{channel_name}' but collection is incomplete. "
                f"Have {len(fragments_map)}/{expected_count}. This should not happen."
            )
            raise RuntimeError(f"Internal logic error: Incomplete fragment collection for {channel_name}")

        fragment_list: List[Dict[str, Any]] = list(fragments_map.values())
        logger.debug(f"Successfully collected {len(fragment_list)} fragments for '{channel_name}' non-destructively.")
        return fragment_list

    def _fetch_fragments_cached(self, channel_name: str, timeout: float) -> List[Dict[str, Any]]:
        """
        Attempts to retrieve cached message fragments; if unsuccessful, fetches destructively from Redis
        and writes the result to cache.

        Parameters
        ----------
        channel_name : str
            The Redis channel key to fetch the message from.
        timeout : float
            The timeout in seconds for fetching from Redis.

        Returns
        -------
        List[Dict[str, Any]]
            A list of message fragments retrieved either from cache or Redis.

        Raises
        ------
        RuntimeError
            If caching is not configured.
        NotImplementedError
            If caching of fragments is not implemented.
        """
        if not self._cache:
            logger.error("Cache is not configured or failed to initialize. Cannot use CACHE_BEFORE_DELETE mode.")
            raise RuntimeError("Cache not available for cached fetch mode.")

        cache_key: str = f"fetch_cache:{channel_name}"
        try:
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for '{channel_name}'. Returning cached data.")
                self._cache.delete(cache_key)
                # TODO: Decide on final caching design.
                raise NotImplementedError("Caching fragments is complex; cache final result instead.")
        except Exception as e:
            logger.exception(f"Error accessing cache for '{channel_name}': {e}. Proceeding to Redis fetch.")

        logger.debug(f"Cache miss for '{channel_name}'. Fetching destructively from Redis.")
        fragments = self._fetch_first_or_all_fragments_destructive(channel_name, timeout)
        try:
            self._cache.set(cache_key, fragments, expire=self._cache_ttl)
            logger.debug(f"Stored fetched fragments for '{channel_name}' in cache.")
        except Exception as e:
            logger.exception(f"Failed to write fragments for '{channel_name}' to cache: {e}")
        return fragments

    def fetch_message(
        self, channel_name: str, timeout: float = 10, override_fetch_mode: Optional["FetchMode"] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches a complete message from Redis. It handles fragmentation according to the specified
        or configured fetch mode and retries on connection errors.

        Parameters
        ----------
        channel_name : str
            The Redis channel key from which to fetch the message.
        timeout : float, optional
            The timeout in seconds for fetching the message. Default is 10 seconds.
        override_fetch_mode : FetchMode, optional
            If provided, overrides the configured fetch mode for this operation.

        Returns
        -------
        dict or None
            The final reconstructed message dictionary if successful, or None if not found.

        Raises
        ------
        TimeoutError
            If fetching times out.
        ValueError
            If non-retryable errors occur or max retries are exceeded.
        RuntimeError
            For other runtime errors.
        """
        retries: int = 0
        effective_fetch_mode: "FetchMode" = override_fetch_mode if override_fetch_mode is not None else self._fetch_mode
        log_prefix: str = f"fetch_message(mode={effective_fetch_mode.name}, channel='{channel_name}')"
        if override_fetch_mode:
            logger.debug(f"{log_prefix}: Using overridden mode.")
        else:
            logger.debug(f"{log_prefix}: Using configured mode.")

        if effective_fetch_mode == FetchMode.CACHE_BEFORE_DELETE and DISKCACHE_AVAILABLE:
            if not self._cache:
                raise RuntimeError(f"{log_prefix}: Cache not available.")

            cache_key: str = f"fetch_cache:{channel_name}"
            try:
                cached_final_result = self._cache.get(cache_key)
                if cached_final_result is not None:
                    logger.info(f"{log_prefix}: Cache hit.")
                    self._cache.delete(cache_key)
                    return cached_final_result
            except Exception as e:
                logger.exception(f"{log_prefix}: Cache read error: {e}. Trying Redis.")

        # If caller requests non-blocking behavior (timeout <= 0), attempt immediate pop.
        if timeout is not None and timeout <= 0:
            try:
                client = self.get_client()
                popped = client.lpop(channel_name)
                if popped is None:
                    return None
                try:
                    return json.loads(popped)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON from non-blocking LPOP on '{channel_name}': {e}")
                    return None
            except Exception as e:
                logger.warning(f"Non-blocking LPOP failed for '{channel_name}': {e}")
                return None

        while True:
            try:
                fetch_result: Union[Dict[str, Any], List[Dict[str, Any]]]
                if effective_fetch_mode == FetchMode.DESTRUCTIVE:
                    fetch_result = self._fetch_first_or_all_fragments_destructive(channel_name, timeout)
                elif effective_fetch_mode == FetchMode.NON_DESTRUCTIVE:
                    fetch_result = self._fetch_fragments_non_destructive(channel_name, timeout)
                elif effective_fetch_mode == FetchMode.CACHE_BEFORE_DELETE:
                    fetch_result = self._fetch_first_or_all_fragments_destructive(channel_name, timeout)
                else:
                    raise ValueError(f"{log_prefix}: Unsupported fetch mode: {effective_fetch_mode}")

                if isinstance(fetch_result, dict):
                    logger.debug(f"{log_prefix}: Received single message directly.")
                    final_message: Dict[str, Any] = fetch_result
                elif isinstance(fetch_result, list):
                    logger.debug(f"{log_prefix}: Received {len(fetch_result)} fragments, combining.")
                    final_message = self._combine_fragments(fetch_result)
                else:
                    logger.error(f"{log_prefix}: Fetch helper returned unexpected type: {type(fetch_result)}")
                    raise TypeError("Internal error: Unexpected fetch result type.")

                if effective_fetch_mode == FetchMode.CACHE_BEFORE_DELETE and self._cache:
                    cache_key = f"fetch_cache:{channel_name}"
                    try:
                        self._cache.set(cache_key, final_message, expire=self._cache_ttl)
                        logger.info(f"{log_prefix}: Stored reconstructed message in cache.")
                    except Exception as e:
                        logger.exception(f"{log_prefix}: Cache write error: {e}")
                return final_message

            except TimeoutError as e:
                logger.debug(f"{log_prefix}: Timeout during fetch operation: {e}")
                raise e

            except (redis.RedisError, ConnectionError) as e:
                retries += 1
                logger.warning(
                    f"{log_prefix}: Redis/Connection error ({type(e).__name__}): {e}. "
                    f"Attempt {retries}/{self.max_retries}"
                )
                self._client = None
                if self.max_retries > 0 and retries <= self.max_retries:
                    backoff_delay: float = min(2 ** (retries - 1), self._max_backoff)
                    jitter: float = random.uniform(0, backoff_delay * 0.2)
                    sleep_time: float = backoff_delay + jitter
                    logger.info(f"{log_prefix}: Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error(f"{log_prefix}: Max retries ({self.max_retries}) exceeded. Last error: {e}")
                    raise ValueError(f"Failed to fetch from Redis after {retries} attempts: {e}") from e

            except (ValueError, RuntimeError, TypeError, NotImplementedError) as e:
                logger.error(f"{log_prefix}: Non-retryable error during fetch: ({type(e).__name__}) {e}")
                raise e

            except Exception as e:
                logger.exception(f"{log_prefix}: Unexpected error during fetch: {e}")
                raise ValueError(f"Unexpected error during fetch: {e}") from e

    def fetch_message_from_any(self, channel_names: List[str], timeout: float = 0) -> Optional[Dict[str, Any]]:
        """
        Attempt to fetch a message from the first non-empty list among the provided channel names
        using Redis BLPOP. If the popped item represents a fragmented message, this method will
        continue popping from the same channel to reconstruct the full message.

        Parameters
        ----------
        channel_names : List[str]
            Ordered list of Redis list keys to attempt in priority order.
        timeout : float, optional
            Timeout in seconds to wait for any item across the provided lists. Redis supports
            integer-second timeouts; sub-second values will be truncated.

        Returns
        -------
        dict or None
            The reconstructed message dictionary if an item was fetched; otherwise None on timeout.
        """
        if not channel_names:
            return None

        client = self.get_client()
        blpop_timeout = int(max(0, timeout))
        try:
            res = client.blpop(channel_names, timeout=blpop_timeout)
        except (redis.RedisError, ConnectionError) as e:
            logger.debug(f"BLPOP error on {channel_names}: {e}")
            return None

        if res is None:
            return None

        list_key, first_bytes = res
        if isinstance(list_key, bytes):
            try:
                list_key = list_key.decode("utf-8")
            except Exception:
                list_key = str(list_key)
        # Decode first element
        try:
            first_msg = json.loads(first_bytes)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON popped from '{list_key}': {e}")
            return None

        expected_count: int = int(first_msg.get("fragment_count", 1))
        if expected_count <= 1:
            return first_msg

        # Collect remaining fragments from the same list key
        fragments: List[Dict[str, Any]] = [first_msg]
        accumulated = 0.0
        start_time = time.monotonic()
        for i in range(1, expected_count):
            remaining = max(0, timeout - accumulated)
            per_frag_timeout = int(max(1, remaining)) if timeout else 1
            try:
                frag_res = client.blpop([list_key], timeout=per_frag_timeout)
            except (redis.RedisError, ConnectionError) as e:
                logger.error(f"BLPOP error while collecting fragments from '{list_key}': {e}")
                return None
            if frag_res is None:
                logger.error(f"Timeout while collecting fragment {i}/{expected_count-1} from '{list_key}'")
                return None
            _, frag_key_bytes_or_val = frag_res
            # Redis returns (key, value); we don't need the key here
            frag_bytes = frag_key_bytes_or_val
            try:
                frag_msg = json.loads(frag_bytes)
                fragments.append(frag_msg)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode fragment JSON from '{list_key}': {e}")
                return None
            accumulated = time.monotonic() - start_time

        # Combine and return
        try:
            return self._combine_fragments(fragments)
        except Exception as e:
            logger.error(f"Error combining fragments from '{list_key}': {e}")
            return None

    def fetch_message_from_any_with_key(
        self, channel_names: List[str], timeout: float = 0
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Like fetch_message_from_any(), but returns the Redis list key together with the message.
        This is useful for higher-level schedulers that need to apply per-category quotas.
        """
        if not channel_names:
            return None

        client = self.get_client()
        blpop_timeout = int(max(0, timeout))
        try:
            res = client.blpop(channel_names, timeout=blpop_timeout)
        except (redis.RedisError, ConnectionError) as e:
            logger.debug(f"BLPOP error on {channel_names}: {e}")
            return None

        if res is None:
            return None

        list_key, first_bytes = res
        try:
            first_msg = json.loads(first_bytes)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON popped from '{list_key}': {e}")
            return None

        expected_count: int = int(first_msg.get("fragment_count", 1))
        if expected_count <= 1:
            return list_key, first_msg

        fragments: List[Dict[str, Any]] = [first_msg]
        accumulated = 0.0
        start_time = time.monotonic()
        for i in range(1, expected_count):
            remaining = max(0, timeout - accumulated)
            per_frag_timeout = int(max(1, remaining)) if timeout else 1
            try:
                frag_res = client.blpop([list_key], timeout=per_frag_timeout)
            except (redis.RedisError, ConnectionError) as e:
                logger.error(f"BLPOP error while collecting fragments from '{list_key}': {e}")
                return None
            if frag_res is None:
                logger.error(f"Timeout while collecting fragment {i}/{expected_count-1} from '{list_key}'")
                return None
            _, frag_bytes = frag_res
            try:
                frag_msg = json.loads(frag_bytes)
                fragments.append(frag_msg)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode fragment JSON from '{list_key}': {e}")
                return None
            accumulated = time.monotonic() - start_time

        try:
            return list_key, self._combine_fragments(fragments)
        except Exception as e:
            logger.error(f"Error combining fragments from '{list_key}': {e}")
            return None

    @staticmethod
    def _combine_fragments(fragments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combines a list of message fragments into a single message by merging shared metadata
        and concatenating the fragment data lists.

        Parameters
        ----------
        fragments : List[Dict[str, Any]]
            A list of fragment dictionaries containing at least a 'data' key and optional metadata.

        Returns
        -------
        dict
            A combined message dictionary.

        Raises
        ------
        ValueError
            If the fragments list is empty.
        """
        if not fragments:
            raise ValueError("Cannot combine empty list of fragments")

        fragments.sort(key=lambda x: x.get("fragment", 0))
        combined_message: Dict[str, Any] = {"data": []}
        first_frag: Dict[str, Any] = fragments[0]

        for key in ["status", "description", "trace", "annotations"]:
            if key in first_frag:
                combined_message[key] = first_frag[key]

        for fragment in fragments:
            fragment_data = fragment.get("data")
            if isinstance(fragment_data, list):
                combined_message["data"].extend(fragment_data)
            else:
                fragment_idx = fragment.get("fragment", "unknown")
                logger.warning(f"Fragment {fragment_idx} missing 'data' list or has wrong type. Skipping its data.")

        return combined_message

    def submit_message(
        self,
        channel_name: str,
        message: str,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Submits a message to Redis using RPUSH and optionally sets a TTL on the channel key.

        Parameters
        ----------
        channel_name : str
            The Redis list key (queue name) to which the message will be appended.
        message : str
            The message payload as a JSON string.
        ttl_seconds : int, optional
            Time-To-Live for the Redis key in seconds. If not provided, uses message_ttl_seconds.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If maximum retry attempts are exceeded.
        ConnectionError
            If there is a connection error with Redis.
        redis.RedisError
            For other non-recoverable Redis errors.
        """
        retries: int = 0

        while True:
            try:
                client: redis.Redis = self.get_client()
                pipe = client.pipeline()
                pipe.rpush(channel_name, message)
                effective_ttl: Optional[int] = ttl_seconds if ttl_seconds is not None else self._message_ttl_seconds
                if effective_ttl is not None and effective_ttl > 0:
                    pipe.expire(channel_name, effective_ttl)
                pipe.execute()
                logger.debug(
                    f"Message submitted to '{channel_name}'"
                    + (f" with TTL {effective_ttl}s." if effective_ttl else ".")
                )
                return
            except (redis.RedisError, ConnectionError) as e:
                retries += 1
                logger.warning(
                    f"Redis/Connection error submitting to '{channel_name}': {e}. Attempt {retries}/{self.max_retries}"
                )
                self._client = None
                if self.max_retries > 0 and retries <= self.max_retries:
                    backoff_delay: float = min(2 ** (retries - 1), self._max_backoff)
                    jitter: float = random.uniform(0, backoff_delay * 0.2)
                    sleep_time: float = backoff_delay + jitter
                    logger.debug(f"Retrying submit for '{channel_name}' in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error(
                        f"Max retries ({self.max_retries}) exceeded submitting to '{channel_name}'. Last error: {e}"
                    )
                    raise ValueError(f"Failed to submit to Redis after {retries} attempts: {e}") from e
            except Exception as e:
                logger.exception(f"Unexpected error during submit to '{channel_name}': {e}")
                raise ValueError(f"Unexpected error during submit: {e}") from e
