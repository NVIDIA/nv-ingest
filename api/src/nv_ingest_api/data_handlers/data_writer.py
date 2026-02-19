# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Dict, List, Literal, Optional, Union, Callable
from pydantic import BaseModel, Field
from concurrent.futures import Future as ConcurrentFuture, ThreadPoolExecutor

# External dependencies (optional - checked at runtime)
try:
    import fsspec
except ImportError:
    fsspec = None

try:
    import requests
except ImportError:
    requests = None

try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None

try:
    from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient
    from nv_ingest_api.data_handlers.errors import (
        DataWriterError,
        TransientError,
        PermanentError,
        ConnectionError,
        AuthenticationError,
        ConfigurationError,
        DependencyError,
    )
    from nv_ingest_api.data_handlers.backoff_strategies import (  # noqa: F401
        BackoffStrategy,  # noqa: F401
        create_backoff_strategy,  # noqa: F401
        BackoffStrategyType,  # noqa: F401
    )
    from nv_ingest_api.data_handlers.writer_strategies import get_writer_strategy
except ImportError:
    RedisClient = None
    DataWriterError = None
    TransientError = None
    PermanentError = None
    ConnectionError = None
    AuthenticationError = None
    ConfigurationError = None
    DependencyError = None

logger = logging.getLogger(__name__)


# Dependency availability checks
def _check_kafka_available() -> bool:
    """Check if kafka-python library is available."""
    try:
        import kafka  # noqa: F401

        return True
    except ImportError:
        return False


def _check_fsspec_available() -> bool:
    """Check if fsspec library is available."""
    try:
        import fsspec  # noqa: F401

        return True
    except ImportError:
        return False


def _check_redis_available() -> bool:
    """Check if Redis client is available."""
    try:
        from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient  # noqa: F401

        return True
    except ImportError:
        return False


# Cache dependency availability
_KAFKA_AVAILABLE = _check_kafka_available()
_FSSPEC_AVAILABLE = _check_fsspec_available()
_REDIS_AVAILABLE = _check_redis_available()

# Main thread executor for safe callback execution
_MAIN_THREAD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="main-callback")


def classify_error(error: Exception, destination_type: str) -> DataWriterError:
    """
    Classify an exception into appropriate error categories.

    Parameters
    ----------
    error : Exception
        The original exception
    destination_type : str
        Type of destination where error occurred

    Returns
    -------
    DataWriterError
        Classified error with appropriate category
    """
    # Preserve already-classified errors
    if isinstance(error, PermanentError):
        return error
    if isinstance(error, TransientError):
        return error

    # Handle dependency errors first - always permanent
    if isinstance(error, DependencyError):
        return PermanentError(f"Dependency not available: {error}")

    error_str = str(error).lower()
    error_type = type(error).__name__

    # Network/connection errors (transient)
    if any(
        keyword in error_str
        for keyword in [
            "connection",
            "timeout",
            "unreachable",
            "network",
            "econnrefused",
            "etimedout",
            "enotfound",
            "temporary failure",
        ]
    ) or error_type in ["ConnectionError", "TimeoutError", "OSError"]:
        return ConnectionError(f"Connection error: {error}")

    # Authentication errors (permanent)
    if any(
        keyword in error_str
        for keyword in [
            "unauthorized",
            "forbidden",
            "authentication",
            "credentials",
            "access denied",
            "invalid credentials",
        ]
    ) or error_type in ["AuthenticationError"]:
        return AuthenticationError(f"Authentication error: {error}")

    # Kafka-specific errors
    if destination_type == "kafka":
        if "topic" in error_str and ("not found" in error_str or "unknown_topic" in error_str):
            return PermanentError(f"Kafka topic error: {error}")
        elif "leader" in error_str or "partition" in error_str:
            return TransientError(f"Kafka leadership error: {error}")

    # HTTP-specific errors
    if destination_type == "http":
        if hasattr(error, "response") and error.response:
            status_code = error.response.status_code
            if status_code in [401, 403]:
                return AuthenticationError(f"HTTP auth error ({status_code}): {error}")
            elif status_code >= 400 and status_code < 500:
                return PermanentError(f"HTTP client error ({status_code}): {error}")
            elif status_code >= 500:
                return TransientError(f"HTTP server error ({status_code}): {error}")

    # File system errors
    if destination_type == "filesystem":
        if "permission denied" in error_str or "access denied" in error_str:
            return PermanentError(f"Filesystem permission error: {error}")
        elif "no space" in error_str or "disk full" in error_str:
            return PermanentError(f"Filesystem space error: {error}")

    # Default to transient for unknown errors
    return TransientError(f"Unclassified error: {error}")


# Callback type definitions
SuccessCallback = Callable[
    [
        List[str],
        Union[
            "RedisDestinationConfig", "FilesystemDestinationConfig", "HttpDestinationConfig", "KafkaDestinationConfig"
        ],
    ],
    None,
]
FailureCallback = Callable[
    [
        List[str],
        Union[
            "RedisDestinationConfig", "FilesystemDestinationConfig", "HttpDestinationConfig", "KafkaDestinationConfig"
        ],
        Exception,
    ],
    None,
]


class DestinationConfig(BaseModel):
    """Base class for destination configurations."""

    type: str
    retry_count: int = Field(default=2, ge=0, description="Number of retry attempts on failure")
    backoff_strategy: BackoffStrategyType = Field(
        default="exponential", description="Backoff strategy for retry delays"
    )

    class Config:
        extra = "forbid"


class RedisDestinationConfig(DestinationConfig):
    """Configuration for Redis message broker output."""

    type: Literal["redis"] = "redis"
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    channel: str  # Will be set from response_channel


class FilesystemDestinationConfig(DestinationConfig):
    """Configuration for filesystem output using fsspec."""

    type: Literal["filesystem"] = "filesystem"
    path: str  # URI like s3://bucket/path, file:///local/path, etc.


class HttpDestinationConfig(DestinationConfig):
    """Configuration for HTTP output."""

    type: Literal["http"] = "http"
    url: str
    method: str = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)
    auth_token: Optional[str] = None
    query_params: Dict[str, str] = Field(default_factory=dict)


class KafkaDestinationConfig(DestinationConfig):
    """Configuration for Kafka message broker output."""

    type: Literal["kafka"] = "kafka"
    bootstrap_servers: List[str]  # List of kafka brokers, e.g., ["localhost:9092"]
    topic: str  # Kafka topic to publish to
    key_serializer: Optional[str] = None  # Optional key for partitioning
    value_serializer: Literal["json", "string"] = "json"
    security_protocol: Literal["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"] = "PLAINTEXT"
    sasl_mechanism: Optional[Literal["PLAIN", "GSSAPI", "SCRAM-SHA-256", "SCRAM-SHA-512"]] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None


# Union type for all destination configs
AnyDestinationConfig = Union[
    RedisDestinationConfig, FilesystemDestinationConfig, HttpDestinationConfig, KafkaDestinationConfig
]


class IngestDataWriter:
    """
    Singleton data writer for external systems with async I/O and retry logic.

    Supports multiple destination types with pydantic-validated configurations.
    """

    _instance = None
    _initialized = False

    def __new__(cls, max_workers: int = 4):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = 4):
        # Only initialize once due to singleton pattern
        if not self._initialized:
            self._output_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="data-writer")
            self._initialized = True

    @classmethod
    def get_instance(cls, max_workers: int = 4) -> "IngestDataWriter":
        """
        Get the singleton instance of the data writer.

        Parameters
        ----------
        max_workers : int
            Number of worker threads (only used on first instantiation)

        Returns
        -------
        IngestDataWriter
            The singleton data writer instance
        """
        return cls(max_workers)

    @classmethod
    def reset_for_tests(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        This method shuts down the current instance's thread pool and clears the singleton,
        allowing tests to create fresh instances with different configurations.

        Warning: Only use this in test environments. This is not thread-safe and should
        not be used in production.
        """
        if cls._instance is not None:
            try:
                cls._instance.shutdown()
            except Exception:
                pass  # Ignore shutdown errors in tests
            cls._instance = None
            cls._initialized = False

    def write_async(
        self,
        data_payload: List[str],
        destination_config: AnyDestinationConfig,
        on_success: Optional[SuccessCallback] = None,
        on_failure: Optional[FailureCallback] = None,
        callback_executor: Optional["ThreadPoolExecutor"] = _MAIN_THREAD_EXECUTOR,
    ) -> ConcurrentFuture:
        """
        Write data payload to destination asynchronously.

        Parameters
        ----------
        data_payload : List[str]
            List of JSON string payloads to write
        destination_config : AnyDestinationConfig
            Pydantic-validated destination configuration
        on_success : Optional[SuccessCallback]
            Callback function called on successful write
            Signature: (data_payload, destination_config) -> None
        on_failure : Optional[FailureCallback]
            Callback function called on failed write
            Signature: (data_payload, destination_config, exception) -> None
        callback_executor : Optional[ThreadPoolExecutor]
            Executor to run callbacks on. Defaults to main thread executor for safety.
            Pass None to run callbacks on worker thread, or provide custom executor.

        Returns
        -------
        Future[None]
            Future that completes when the entire write operation finishes (including callback execution).
            Can be used for cancellation, awaiting, or composition with other async operations.
        """
        # Create a Future that represents the complete operation (write + callbacks)
        result_future = ConcurrentFuture()

        def on_write_complete(future):
            """Handle the completion of the write operation and callbacks."""
            try:
                # Check if our result future was cancelled
                if result_future.cancelled():
                    return

                # This will trigger callback execution
                self._handle_write_result(
                    future, data_payload, destination_config, on_success, on_failure, callback_executor
                )
                # Set the result future as completed successfully
                result_future.set_result(None)
            except Exception as e:
                # Set the result future as failed
                result_future.set_exception(e)

        # Submit the write operation and attach our completion handler
        write_future = self._output_pool.submit(self._write_sync, data_payload, destination_config)
        write_future.add_done_callback(on_write_complete)

        # Set up cancellation handling
        def cancel_callback():
            """Handle cancellation of the result future."""
            if not write_future.cancelled():
                write_future.cancel()

        result_future.add_done_callback(lambda f: cancel_callback() if f.cancelled() else None)

        return result_future

    def _write_sync(self, data_payload: List[str], destination_config: AnyDestinationConfig) -> None:
        """
        Synchronous write implementation with intelligent retry logic.

        Only retries on transient errors, fails immediately on permanent errors.

        Parameters
        ----------
        data_payload : List[str]
            List of JSON string payloads
        destination_config : AnyDestinationConfig
            Destination configuration with retry settings
        """
        # Resolve backoff strategy from string to actual strategy object
        backoff_strategy = create_backoff_strategy(destination_config.backoff_strategy)
        backoff_func = backoff_strategy.calculate_delay

        for attempt in range(destination_config.retry_count + 1):  # +1 for initial attempt
            try:
                # Get the appropriate writer strategy and write the data
                writer_strategy = get_writer_strategy(destination_config.type)
                writer_strategy.write(data_payload, destination_config)
                return  # Success, exit retry loop

            except Exception as e:
                # Classify the error
                classified_error = classify_error(e, destination_config.type)

                # Don't retry on permanent errors
                if isinstance(classified_error, PermanentError):
                    logger.error(f"Permanent error for {destination_config.type}, not retrying: {classified_error}")
                    raise classified_error from e

                # For transient errors, check if we should retry
                if attempt == destination_config.retry_count:
                    # Final attempt failed
                    logger.error(
                        f"All {destination_config.retry_count + 1} attempts failed for {destination_config.type}: "
                        f"{classified_error}"
                    )
                    raise classified_error from e
                else:
                    # Calculate backoff delay and wait
                    delay = backoff_func(attempt)
                    logger.warning(
                        f"Transient error on attempt {attempt + 1}/{destination_config.retry_count + 1} "
                        f"for {destination_config.type}, retrying in {delay:.2f}s: {classified_error}"
                    )
                    time.sleep(delay)

    def _handle_write_result(
        self,
        future,
        data_payload: List[str],
        destination_config: AnyDestinationConfig,
        on_success: Optional[SuccessCallback],
        on_failure: Optional[FailureCallback],
        callback_executor: Optional["ThreadPoolExecutor"],
    ) -> None:
        """
        Handle completion of write operations and invoke callbacks.

        Parameters
        ----------
        future : Future
            The completed future from the async write operation
        data_payload : List[str]
            The original data payload that was written
        destination_config : AnyDestinationConfig
            The destination configuration used
        on_success : Optional[SuccessCallback]
            Success callback to invoke
        on_failure : Optional[FailureCallback]
            Failure callback to invoke
        callback_executor : Optional[ThreadPoolExecutor]
            Executor to run callbacks on, if provided
        """

        def invoke_callbacks():
            """Inner function to invoke callbacks, potentially on different executor."""
            try:
                future.result()  # Raise any exception that occurred
                # Success - invoke success callback if provided
                if on_success:
                    try:
                        on_success(data_payload, destination_config)
                    except Exception as callback_error:
                        logger.error(f"Error in success callback: {callback_error}", exc_info=True)
            except Exception as e:
                # Failure - invoke failure callback if provided
                if on_failure:
                    try:
                        on_failure(data_payload, destination_config, e)
                    except Exception as callback_error:
                        logger.error(f"Error in failure callback: {callback_error}", exc_info=True)
                # Always log the original error
                logger.error(f"Data write operation failed: {e}", exc_info=True)

        # Execute callbacks on specified executor or directly
        if callback_executor is not None:
            callback_executor.submit(invoke_callbacks)
        else:
            invoke_callbacks()

    def shutdown(self):
        """Shutdown the writer and its worker pool."""
        self._output_pool.shutdown(wait=True)
        # Note: Main thread executor is not shut down as it's shared across instances
