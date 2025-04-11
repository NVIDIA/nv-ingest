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
from redis.exceptions import RedisError

from diskcache import Cache

from nv_ingest_api.util.service_clients.client_base import MessageBrokerClientBase, FetchMode

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
        max_retries: int = 3,  # Changed default to 3 for robustness
        max_backoff: int = 32,
        connection_timeout: int = 300,
        max_pool_size: int = 128,
        use_ssl: bool = False,
        redis_allocator: Callable[..., redis.Redis] = redis.Redis,
        fetch_mode: FetchMode = FetchMode.DESTRUCTIVE,  # Add fetch_mode config
        cache_config: Optional[Dict[str, Any]] = None,  # Add cache config
        # Example cache_config: {"directory": "/path/to/cache", "ttl": 7200}
        message_ttl_seconds: Optional[int] = None,  # TTL for NON_DESTRUCTIVE mode items
    ):
        self._host = host
        self._port = port
        self._db = db
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._connection_timeout = connection_timeout
        self._use_ssl = use_ssl  # TODO(Devin) needs to be implemented.
        self._fetch_mode = fetch_mode
        self._message_ttl_seconds = message_ttl_seconds  # Used for NON_DESTRUCTIVE mode cleanup
        self._redis_allocator = redis_allocator

        if fetch_mode == FetchMode.NON_DESTRUCTIVE and message_ttl_seconds is None:
            logger.warning(
                "FetchMode.NON_DESTRUCTIVE selected without setting message_ttl_seconds. "
                "Messages fetched non-destructively may persist indefinitely in Redis."
            )

        # Configure Connection Pool
        pool_kwargs = {
            "host": self._host,
            "port": self._port,
            "db": self._db,
            "socket_connect_timeout": self._connection_timeout,
            "max_connections": max_pool_size,
            # Add SSL context if needed based on self._use_ssl
        }
        if self._use_ssl:
            # Basic SSL setup, might need more options (certs, etc.)
            pool_kwargs["ssl"] = True
            pool_kwargs["ssl_cert_reqs"] = None  # Or specify cert requirements
            logger.debug("Redis connection configured with SSL.")

        self._pool = redis.ConnectionPool(**pool_kwargs)

        # Allocate initial client
        self._client = self._redis_allocator(connection_pool=self._pool)

        # Configure Cache if mode requires it
        self._cache = None
        if self._fetch_mode == FetchMode.CACHE_BEFORE_DELETE:
            cache_dir = (cache_config or {}).get("directory", DEFAULT_CACHE_DIR)
            self._cache_ttl = (cache_config or {}).get("ttl", DEFAULT_CACHE_TTL_SECONDS)
            try:
                # TODO(Devin): make size_limit configurable
                self._cache = Cache(cache_dir, timeout=self._cache_ttl, size_limit=int(50e9))
                logger.debug(f"Fetch cache enabled: mode={self._fetch_mode}, dir={cache_dir}, ttl={self._cache_ttl}s")
            except Exception as e:
                logger.exception(f"Failed to initialize disk cache at {cache_dir}. Caching disabled. Error: {e}")
                # Optionally fallback or raise? For now, disable caching.
                self._fetch_mode = FetchMode.DESTRUCTIVE  # Fallback to destructive
                logger.warning("Falling back to FetchMode.DESTRUCTIVE due to cache init failure.")

        # Validate max_retries on init
        self.max_retries = max_retries  # Use the setter for validation

    def _connect(self) -> None:
        """Attempts to reconnect to the Redis server."""
        # This might be overly simplistic if allocator takes other args.
        # Assumes allocator primarily needs the pool.
        logger.debug("Attempting to reconnect to Redis by re-allocating client.")
        try:
            self._client = self._redis_allocator(connection_pool=self._pool)
            if not self.ping():  # Verify connection after re-allocation
                raise ConnectionError("Re-allocated client failed to ping.")
            logger.info("Successfully reconnected to Redis.")
        except Exception as e:
            logger.error(f"Failed to reconnect to Redis: {e}")
            self._client = None  # Ensure client is None on failure

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        self._max_retries = value

    def get_client(self) -> redis.Redis:  # Or appropriate type hint for your allocator
        """
        Returns a Redis client instance, attempting reconnection ONLY if
        the current client instance is None (invalidated).

        Raises
        ------
        RedisConnectionError
            If the client instance is None and reconnection via _connect() fails.
        """
        # Only attempt to connect if the client reference is explicitly None
        # (e.g., first call, or after a ping failure invalidated it).
        if self._client is None:
            logger.info("Redis client is None, attempting to connect.")
            try:
                # _connect should attempt to establish connection and set self._client
                self._connect()
            except Exception as connect_err:
                # Log connection error but also raise a specific error below
                logger.error(f"Error during _connect attempt: {connect_err}")
                self._client = None  # Ensure it's None after failed connect

            # After potentially connecting, check if we *actually* have a client.
            # _connect sets self._client to None if it fails.
            if self._client is None:
                # Raise an error if connection failed.
                raise RuntimeError("Failed to establish or re-establish connection to Redis.")

        # Return the client reference (either existing or newly connected).
        return self._client

    def ping(self) -> bool:
        """
        Checks if the current Redis client connection is responsive.
        Invalidates the client reference (`self._client = None`) if the ping fails.

        Returns
        -------
        bool
            True if the current client responds to a ping, False otherwise.
        """
        # If we don't even have a client reference, it's definitely not responsive.
        if self._client is None:
            logger.debug("Ping check: No client instance exists.")
            return False
        try:
            # Directly use the current client instance's ping command.
            # DO NOT call self.get_client() here.
            is_alive = self._client.ping()  # redis-py ping() raises on error

            # If ping() returns True (or doesn't raise), the connection is likely fine.
            # Note: redis-py ping() should return True or raise. Checking for explicit True.
            if is_alive:
                logger.debug("Ping successful.")
                return True
            else:
                # Defensive: Should not happen with modern redis-py, but handle if it does.
                logger.warning("Ping command returned non-True value unexpectedly.")
                self._client = None  # Invalidate on unexpected response
                return False

        except (OSError, AttributeError) as e:
            # Catch specific exceptions indicating connection failure or invalid client.
            # OSError can happen with broken pipes etc.
            # AttributeError if _client is somehow not a valid redis client object.
            logger.warning(f"Ping failed, invalidating client connection: ({type(e).__name__}) {e}")
            # Critical step: Invalidate the client reference.
            self._client = None
            return False
        except RedisError as e:
            # Catch other potential Redis errors during ping that aren't connection related
            logger.warning(f"Ping failed due to RedisError: {e}. Invalidating client.")
            self._client = None
            return False
        except Exception as e:
            # Catch any other unexpected error during ping
            logger.exception(f"Unexpected error during ping, invalidating client: {e}")
            self._client = None
            return False

    def _check_response(
        self, channel_name: str, timeout: float
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[int]]:
        """
        Checks for a response from the Redis queue and processes it into a message, fragment, and fragment count.

        Parameters
        ----------
        channel_name : str
            The name of the Redis channel from which to receive the response.
        timeout : float
            The time in seconds to wait for a response from the Redis queue before timing out.

        Returns
        -------
        Tuple[Optional[Dict[str, Any]], Optional[int], Optional[int]]
            A tuple containing:
                - message: A dictionary containing the decoded message if successful,
                    or None if no message was retrieved.
                - fragment: An integer representing the fragment number of the message,
                    or None if no fragment was found.
                - fragment_count: An integer representing the total number of message fragments,
                    or None if no fragment count was found.

        Raises
        ------
        ValueError
            If the message retrieved from Redis cannot be decoded from JSON.
        """

        response = self.get_client().blpop([channel_name], timeout)
        if response is None:
            raise TimeoutError("No response was received in the specified timeout period")

        if len(response) > 1 and response[1]:
            try:
                message = json.loads(response[1])
                fragment = message.get("fragment", 0)
                fragment_count = message.get("fragment_count", 1)

                return message, fragment, fragment_count
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")
                raise ValueError(f"Failed to decode message from Redis: {e}")

        return None, None, None

    def _fetch_first_or_all_fragments_destructive(self, channel_name: str, timeout: float) -> Union[Dict, List[Dict]]:
        """
        Internal: Fetches using BLPOP. Returns single dict if not fragmented,
        or list of fragment dicts if fragmented.
        """
        fragments = []
        expected_count = 1
        first_message = None
        accumulated_fetch_time = 0.0  # Track time spent fetching subsequent fragments

        logger.debug(f"Destructive fetch: Popping first item from '{channel_name}' with timeout {timeout:.2f}s")
        start_pop_time = time.monotonic()
        response = self.get_client().blpop([channel_name], timeout=int(max(1, timeout)))
        fetch_duration = time.monotonic() - start_pop_time

        if response is None:
            logger.debug(f"BLPOP timed out on '{channel_name}', no message available.")
            raise TimeoutError("No message received within the initial timeout period")

        if len(response) > 1 and response[1]:
            message_bytes = response[1]
            try:
                first_message = json.loads(message_bytes)
                expected_count = first_message.get("fragment_count", 1)
                fragment_idx = first_message.get("fragment", 0)  # Default to 0 if missing

                # --- Single Message Case ---
                if expected_count == 1:
                    logger.debug(f"Fetched single (non-fragmented) message from '{channel_name}'.")
                    return first_message  # Return the single dict directly

                # --- Multi-Fragment Case ---
                logger.info(
                    f"Fetched fragment {fragment_idx + 1}/{expected_count} from '{channel_name}'."
                    f" Need to fetch remaining."
                )
                if fragment_idx != 0:
                    # This shouldn't happen if producer always sends 0 first, but handle defensively
                    logger.error(
                        f"Expected first fragment (index 0) but got {fragment_idx} from '{channel_name}'. "
                        f"Aborting fetch."
                    )
                    raise ValueError(f"First fragment fetched was index {fragment_idx}, expected 0.")

                fragments.append(first_message)  # Add the first fragment
                accumulated_fetch_time += fetch_duration

                # Loop to fetch remaining fragments
                remaining_timeout = max(0.1, timeout - accumulated_fetch_time)  # Allow at least some time
                for i in range(1, expected_count):
                    start_frag_pop_time = time.monotonic()
                    frag_timeout = max(1, remaining_timeout / max(1, expected_count - i))  # Distribute remaining time
                    logger.debug(f"Popping fragment {i + 1}/{expected_count} with timeout {frag_timeout:.2f}s")

                    frag_response = self.get_client().blpop([channel_name], timeout=int(frag_timeout))
                    frag_fetch_duration = time.monotonic() - start_frag_pop_time
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
                            f"Unexpected BLPOP response format for fragment {i + 1} on"
                            f" '{channel_name}': {frag_response}"
                        )
                        raise ValueError(f"Unexpected BLPOP response format for fragment {i + 1}")

                logger.debug(f"Successfully fetched all {expected_count} fragments destructively.")
                return fragments  # Return the list of fragments

            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode first message JSON from '{channel_name}': {e}. Data: {message_bytes[:200]}"
                )
                raise ValueError(f"Failed to decode first message: {e}") from e
        else:
            # Should not happen with BLPOP returning non-None response, but handle defensively
            logger.warning(f"BLPOP for '{channel_name}' returned unexpected response format: {response}")
            raise ValueError("Unexpected response format from BLPOP")

    def _fetch_fragments_non_destructive(self, channel_name: str, timeout: float) -> List[Dict]:
        """
        Internal: Fetches all fragments non-destructively by polling. Requires TTL.

        Reads fragment 0 first using LINDEX to determine the expected count.
        Then, repeatedly polls the list length (LLEN) and reads potentially new
        fragments using LRANGE until the expected number of unique fragments are
        collected or the overall timeout expires. This is suitable for scenarios
        where fragments might be pushed asynchronously.

        Parameters
        ----------
        channel_name : str
            The Redis list key where fragments are stored.
        timeout : float
            The *total* time in seconds allowed for collecting all fragments.

        Returns
        -------
        List[Dict]
            A list containing all the unique fragment dictionaries for the message.
            The order is not guaranteed by this method itself but fragments contain indices.

        Raises
        ------
        TimeoutError
            If fragment 0 is not found within the timeout, or if all expected
            fragments are not collected before the timeout expires.
        ValueError
            If JSON decoding fails or if inconsistent `fragment_count` values
            are detected across fragments.
        ConnectionError
            If connection to Redis fails during polling.
        RedisError
            If a non-connection Redis error occurs.
        """

        start_time = time.monotonic()
        polling_delay = 0.1  # seconds between polls, adjust if needed
        expected_count: Optional[int] = None
        # Use a dictionary keyed by fragment index to store unique fragments found
        fragments_map: Dict[int, Dict] = {}

        logger.debug(f"Starting non-destructive fetch for '{channel_name}' with total timeout {timeout:.2f}s.")

        while True:  # Main polling loop
            current_time = time.monotonic()
            elapsed_time = current_time - start_time

            # Check overall timeout FIRST in each iteration
            if elapsed_time > timeout:
                logger.warning(f"Overall timeout ({timeout}s) exceeded for non-destructive fetch of '{channel_name}'.")
                if expected_count:
                    raise TimeoutError(
                        f"Timeout collecting fragments for {channel_name}. "
                        f"Collected {len(fragments_map)}/{expected_count}."
                    )
                else:
                    raise TimeoutError(f"Timeout waiting for initial fragment 0 for {channel_name}.")

            # Get a potentially fresh client connection
            client = self.get_client()

            try:
                # --- Stage 1: Find Fragment 0 (if not already found) ---
                if expected_count is None:
                    logger.debug(f"Polling for fragment 0 on '{channel_name}'. Elapsed: {elapsed_time:.2f}s")
                    # LINDEX is O(N) for index > 0, but O(1) for index 0 (usually)
                    frag0_bytes = client.lindex(channel_name, 0)

                    if frag0_bytes is not None:
                        # Found something at index 0, parse it
                        try:
                            message = json.loads(frag0_bytes)
                            fragment_idx = message.get("fragment", -1)  # Use -1 to easily detect missing key
                            current_expected = message.get("fragment_count", 1)

                            if fragment_idx == 0:
                                # Successfully found and parsed fragment 0
                                logger.debug(
                                    f"Found fragment 0 for '{channel_name}'. Expecting {current_expected} "
                                    f"total fragments."
                                )
                                expected_count = current_expected
                                # Store it, ensuring we don't add duplicates if we poll again somehow
                                if fragment_idx not in fragments_map:
                                    fragments_map[fragment_idx] = message

                                # Optimization: If only 1 fragment expected, we are done.
                                if expected_count == 1:
                                    logger.debug("Single fragment expected and found. Fetch complete.")
                                    break  # Exit the main polling loop

                                # Continue to polling stage (or next iteration) if more needed

                            else:
                                # Found data at index 0, but it wasn't fragment 0. This is unexpected.
                                logger.warning(
                                    f"Expected fragment 0 but found index {fragment_idx} "
                                    f"at LINDEX 0 for '{channel_name}'. List state potentially inconsistent. "
                                    f"Will keep polling."
                                )
                                # We don't have expected_count yet, so continue polling LINDEX 0

                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to decode JSON at index 0 for '{channel_name}': {e}. Data: {frag0_bytes[:200]}"
                            )
                            # For now, raise ValueError as it indicates bad data.
                            raise ValueError(f"Failed to decode potential fragment 0: {e}")
                    # else: frag0_bytes is None (list empty or TTL expired at index 0). Loop will poll again after
                    # sleep.

                # --- Stage 2: Poll and Fetch Remaining Fragments (if expected_count is known) ---
                if expected_count is not None and len(fragments_map) < expected_count:
                    # Check current list length - O(1) operation
                    current_len = client.llen(channel_name)
                    logger.debug(
                        f"Polling '{channel_name}': Current length {current_len}, "
                        f"have {len(fragments_map)}/{expected_count} fragments. Elapsed: {elapsed_time:.2f}s"
                    )

                    if current_len >= expected_count:
                        # Read the portion of the list where all fragments *should* be
                        # Read up to expected_count-1 index. If list has more, ignore extras for now.
                        fetch_end_index = expected_count - 1
                        logger.debug(f"Fetching full expected range: LRANGE 0 {fetch_end_index}")
                        # LRANGE is O(S+N) where S is start offset, N is elements. Can be costly for large N.
                        raw_potential_fragments = client.lrange(channel_name, 0, fetch_end_index)

                        processed_count_this_pass = 0
                        for item_bytes in raw_potential_fragments:
                            try:
                                message = json.loads(item_bytes)
                                fragment_idx = message.get("fragment", -1)
                                current_expected_in_frag = message.get("fragment_count", 1)

                                # Validate fragment count consistency
                                if current_expected_in_frag != expected_count:
                                    logger.error(
                                        f"Inconsistent fragment_count in fragment {fragment_idx} for '{channel_name}' "
                                        f"({current_expected_in_frag} vs expected {expected_count})."
                                    )
                                    raise ValueError("Inconsistent fragment count detected in list")

                                # Add to map if it's a valid index and not already seen
                                if 0 <= fragment_idx < expected_count and fragment_idx not in fragments_map:
                                    fragments_map[fragment_idx] = message
                                    processed_count_this_pass += 1
                                    logger.debug(f"Processed fragment {fragment_idx + 1}/{expected_count} from LRANGE.")

                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Failed to decode JSON fragment during poll for '{channel_name}': {e}. "
                                    f"Data: {item_bytes[:200]}"
                                )
                                raise ValueError(f"Failed to decode message fragment: {e}")

                        if processed_count_this_pass > 0:
                            logger.debug(f"Found {processed_count_this_pass} new fragments this pass.")
                        # else: No new unique fragments found in the expected range.

                    # Check completion condition after potential fetch
                    if len(fragments_map) == expected_count:
                        logger.debug(f"Collected all {expected_count} expected fragments for '{channel_name}'.")
                        break  # Exit the main polling loop

                # --- End of loop iteration: Sleep before next poll ---
                # Only sleep if we haven't completed the fetch
                if expected_count is None or len(fragments_map) < expected_count:
                    time.sleep(polling_delay)

            except (ValueError, json.JSONDecodeError) as e:
                # Non-retryable errors during parsing/validation within this attempt
                logger.error(f"Validation or decoding error during non-destructive fetch for '{channel_name}': {e}")
                raise e  # Propagate up immediately
            except (RedisError, ConnectionError) as e:
                # Let Redis/Connection errors propagate up to the main fetch_message retry loop
                logger.warning(
                    f"Redis/Connection error during non-destructive poll for '{channel_name}': {e}. Propagating up."
                )
                raise e  # Allow outer loop to handle retries/reconnects
            except Exception as e:
                # Catch unexpected errors within the polling logic
                logger.exception(f"Unexpected error during non-destructive poll for '{channel_name}': {e}")
                raise RuntimeError(f"Unexpected polling error: {e}") from e
        # --- End of main polling loop ---

        # --- Post Loop Validation ---
        # We should only exit the loop normally if all fragments are collected
        if expected_count is None or len(fragments_map) != expected_count:
            # This could happen if timeout occurred just before the final check,
            # or if there's a logic flaw. The timeout check at loop start handles
            # the timeout case by raising TimeoutError. If we reach here, it's likely logic error.
            logger.error(
                f"Exited non-destructive fetch loop for '{channel_name}' but collection is incomplete. "
                f"Have {len(fragments_map)}/{expected_count}. This should not happen."
            )
            raise RuntimeError(f"Internal logic error: Incomplete fragment collection for {channel_name}")

        # Convert map values to a list. Order is determined by the subsequent sort in _combine_fragments.
        fragment_list = list(fragments_map.values())
        logger.debug(f"Successfully collected {len(fragment_list)} fragments for '{channel_name}' non-destructively.")
        return fragment_list

    def _fetch_fragments_cached(self, channel_name: str, timeout: float) -> List[Dict]:
        """Internal: Tries cache first, then destructive fetch + cache write."""
        if not self._cache:
            logger.error("Cache is not configured or failed to initialize. Cannot use CACHE_BEFORE_DELETE mode.")
            raise RuntimeError("Cache not available for cached fetch mode.")

        cache_key = f"fetch_cache:{channel_name}"
        try:
            # 1. Check Cache
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for '{channel_name}'. Returning cached data.")
                # Decide whether to delete from cache on read. Usually yes for job results.
                self._cache.delete(cache_key)
                # The cached result should be the *reconstructed* message dict
                # We need to return it in a way fetch_message expects (list of fragments is harder here)
                # Let's adjust: Cache stores the *final reconstructed message*
                # This requires _fetch_fragments_cached to return the final dict, not fragments.
                # Refactoring needed... See fetch_message implementation below.
                # For now, let's assume we cache fragments (less ideal)
                # return cached_result # Assuming cached_result is List[Dict]
                # --- Let's redesign to cache the final result ---
                raise NotImplementedError("Caching fragments is complex; cache final result instead.")

        except Exception as e:
            logger.exception(f"Error accessing cache for '{channel_name}': {e}. Proceeding to Redis fetch.")

        # 2. Cache Miss or Error -> Fetch from Redis (Destructive)
        logger.debug(f"Cache miss for '{channel_name}'. Fetching destructively from Redis.")
        fragments = self._fetch_fragments_destructive(channel_name, timeout)

        # 3. Write to Cache (Store the fragments for now, though reconstructing first is better)
        try:
            # Store the list of fragment dicts
            self._cache.set(cache_key, fragments, expire=self._cache_ttl)
            logger.debug(f"Stored fetched fragments for '{channel_name}' in cache.")
        except Exception as e:
            logger.exception(f"Failed to write fragments for '{channel_name}' to cache: {e}")
            # Continue anyway, we have the fragments in memory

        return fragments

    def fetch_message(
        self, channel_name: str, timeout: float = 10, override_fetch_mode: Optional[FetchMode] = None
    ) -> Optional[Dict]:
        """
        Fetches a message, handling fragmentation and fetch modes correctly.

        Returns the raw message dictionary if not fragmented, or a combined
        dictionary if fragmented. Uses configured or overridden fetch mode.
        Handles retries for Redis errors.
        """
        retries = 0

        effective_fetch_mode = override_fetch_mode if override_fetch_mode is not None else self._fetch_mode
        log_prefix = f"fetch_message(mode={effective_fetch_mode.name}, channel='{channel_name}')"
        if override_fetch_mode:
            logger.info(f"{log_prefix}: Using overridden mode.")
        else:
            logger.debug(f"{log_prefix}: Using configured mode.")

        # --- Handle Cache Lookup (ONLY if effective mode is Cache) ---
        if effective_fetch_mode == FetchMode.CACHE_BEFORE_DELETE:
            if not self._cache:
                raise RuntimeError(f"{log_prefix}: Cache not available.")
            cache_key = f"fetch_cache:{channel_name}"
            try:
                cached_final_result = self._cache.get(cache_key)
                if cached_final_result is not None:
                    logger.info(f"{log_prefix}: Cache hit.")
                    self._cache.delete(cache_key)
                    return cached_final_result
            except Exception as e:
                logger.exception(f"{log_prefix}: Cache read error: {e}. Trying Redis.")
        # --- End Cache Lookup ---

        # --- Main Retry Loop (for Redis interaction) ---
        while True:
            try:
                fetch_result: Union[Dict, List[Dict]]  # Result from helper
                # Select fetch strategy based on the *effective* mode
                if effective_fetch_mode == FetchMode.DESTRUCTIVE:
                    fetch_result = self._fetch_first_or_all_fragments_destructive(channel_name, timeout)
                elif effective_fetch_mode == FetchMode.NON_DESTRUCTIVE:
                    # This helper always returns a list
                    fetch_result = self._fetch_fragments_non_destructive(channel_name, timeout)
                elif effective_fetch_mode == FetchMode.CACHE_BEFORE_DELETE:
                    # Cache lookup failed, fetch destructively
                    fetch_result = self._fetch_first_or_all_fragments_destructive(channel_name, timeout)
                else:
                    raise ValueError(f"{log_prefix}: Unsupported fetch mode: {effective_fetch_mode}")

                # --- Process the Fetch Result ---
                final_message: Dict
                if isinstance(fetch_result, dict):
                    # Single message returned directly by destructive helper
                    logger.debug(f"{log_prefix}: Received single message directly.")
                    final_message = fetch_result
                elif isinstance(fetch_result, list):
                    # List of fragments returned (either mode)
                    logger.debug(f"{log_prefix}: Received {len(fetch_result)} fragments, combining.")
                    final_message = self._combine_fragments(fetch_result)
                else:
                    # Should not happen
                    logger.error(f"{log_prefix}: Fetch helper returned unexpected type: {type(fetch_result)}")
                    raise TypeError("Internal error: Unexpected fetch result type.")

                # --- Cache Write (ONLY if effective mode is Cache) ---
                if effective_fetch_mode == FetchMode.CACHE_BEFORE_DELETE and self._cache:
                    cache_key = f"fetch_cache:{channel_name}"
                    try:
                        self._cache.set(cache_key, final_message, expire=self._cache_ttl)
                        logger.info(f"{log_prefix}: Stored reconstructed message in cache.")
                    except Exception as e:
                        logger.exception(f"{log_prefix}: Cache write error: {e}")

                return final_message  # Success

            # --- Exception Handling within Retry Loop ---
            except TimeoutError as e:
                logger.warning(f"{log_prefix}: Timeout during fetch operation: {e}")
                raise e  # Propagate TimeoutError up

            except (RedisError, ConnectionError) as e:
                retries += 1
                logger.warning(
                    f"{log_prefix}: Redis/Connection error ({type(e).__name__}): {e}. "
                    f"Attempt {retries}/{self.max_retries}"
                )
                self._client = None
                if self.max_retries > 0 and retries <= self.max_retries:
                    backoff_delay = min(2 ** (retries - 1), self._max_backoff)
                    jitter = random.uniform(0, backoff_delay * 0.2)
                    sleep_time = backoff_delay + jitter
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

    @staticmethod
    def _combine_fragments(fragments: List[Dict[str, Any]]) -> Dict:
        # ... (Use the flexible version from the previous response that doesn't require status/description) ...
        if not fragments:
            raise ValueError("Cannot combine empty list of fragments")
        fragments.sort(key=lambda x: x.get("fragment", 0))
        combined_message = {"data": []}
        first_frag = fragments[0]
        for key in ["status", "description", "trace", "annotations"]:  # Add others if needed
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
        # message is assumed JSON string here
    ) -> None:
        """
        Submits a message to Redis using RPUSH. Sets TTL if provided. Retries on failure.

        Parameters:
        -----------
        channel_name : str
            The Redis list key (queue name) to push the message onto.
        message : str
            The message payload (expected to be a JSON string).
        ttl_seconds : Optional[int], optional
            If provided, sets the Time-To-Live for the Redis key (list) in seconds
            after the push. Useful for non-destructive reads. Default is None (no TTL).

        Raises:
        -------
        ValueError
            If max retries are exceeded for Redis errors.
        ConnectionError
            If connection to Redis fails.
        RedisError
            Propagated for irrecoverable Redis errors.
        """
        retries = 0

        while True:
            try:
                # Use pipeline for atomic RPUSH + EXPIRE if TTL is needed
                client = self.get_client()
                pipe = client.pipeline()
                pipe.rpush(channel_name, message)
                effective_ttl = ttl_seconds if ttl_seconds is not None else self._message_ttl_seconds
                if effective_ttl is not None and effective_ttl > 0:
                    pipe.expire(channel_name, effective_ttl)

                pipe.execute()  # Execute transactionally

                logger.debug(
                    f"Message submitted to '{channel_name}'"
                    + (f" with TTL {effective_ttl}s." if effective_ttl else ".")
                )
                return  # Success

            except (RedisError, ConnectionError) as e:
                retries += 1
                logger.warning(
                    f"Redis/Connection error submitting to '{channel_name}': {e}. Attempt {retries}/{self.max_retries}"
                )
                self._client = None  # Force reconnect attempt

                if self.max_retries > 0 and retries <= self.max_retries:
                    backoff_delay = min(2 ** (retries - 1), self._max_backoff)
                    jitter = random.uniform(0, backoff_delay * 0.2)
                    sleep_time = backoff_delay + jitter
                    logger.debug(f"Retrying submit for '{channel_name}' in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue  # Retry
                else:
                    logger.error(
                        f"Max retries ({self.max_retries}) exceeded submitting to '{channel_name}'. Last error: {e}"
                    )
                    raise ValueError(f"Failed to submit to Redis after {retries} attempts: {e}") from e
            except Exception as e:
                logger.exception(f"Unexpected error during submit to '{channel_name}': {e}")
                raise ValueError(f"Unexpected error during submit: {e}") from e
