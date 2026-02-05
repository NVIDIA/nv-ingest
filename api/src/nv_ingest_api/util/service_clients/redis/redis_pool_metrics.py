# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Instrumented Redis BlockingConnectionPool with Prometheus metrics.

Provides observability into Redis connection pool utilization including:
- Connections currently in use
- Connections available in the pool
- Maximum pool size
- Time spent waiting for connections
- Pool exhaustion timeout errors
"""

import logging
import threading
import time
from typing import Any, Optional

import redis
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics - defined once at module level
# Using labels allows multiple pools to be tracked independently
POOL_CONNECTIONS_IN_USE = Gauge(
    "redis_pool_connections_in_use",
    "Number of connections currently checked out from the pool",
    ["pool_name"],
)

POOL_CONNECTIONS_AVAILABLE = Gauge(
    "redis_pool_connections_available",
    "Number of connections available in the pool queue",
    ["pool_name"],
)

POOL_CONNECTIONS_MAX = Gauge(
    "redis_pool_connections_max",
    "Maximum number of connections allowed in the pool",
    ["pool_name"],
)

POOL_WAIT_SECONDS = Histogram(
    "redis_pool_wait_seconds",
    "Time spent waiting to acquire a connection from the pool",
    ["pool_name"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0),
)

POOL_TIMEOUT_TOTAL = Counter(
    "redis_pool_timeout_total",
    "Total number of pool exhaustion timeout errors",
    ["pool_name"],
)


class InstrumentedBlockingConnectionPool(redis.BlockingConnectionPool):
    """
    A BlockingConnectionPool subclass that exposes Prometheus metrics for
    monitoring pool utilization and connection wait times.

    Usage:
        pool = InstrumentedBlockingConnectionPool(
            pool_name="ingest",
            host="localhost",
            port=6379,
            max_connections=50,
            timeout=20,
        )
        client = redis.Redis(connection_pool=pool)
    """

    def __init__(self, pool_name: str = "default", **kwargs: Any) -> None:
        """
        Initialize the instrumented connection pool.

        Parameters
        ----------
        pool_name : str
            Name used to label metrics for this pool instance.
        **kwargs
            Arguments passed to BlockingConnectionPool (host, port, max_connections, timeout, etc.)
        """
        super().__init__(**kwargs)
        self._pool_name = pool_name
        self._in_use_count = 0
        self._lock = threading.Lock()

        # Set the max connections gauge once at init
        max_conns = kwargs.get("max_connections", 50)
        POOL_CONNECTIONS_MAX.labels(pool_name=self._pool_name).set(max_conns)

        logger.debug(f"InstrumentedBlockingConnectionPool '{pool_name}' initialized with max_connections={max_conns}")

    def get_connection(self, command_name: Optional[str] = None, *keys: Any, **options: Any) -> Any:
        """
        Get a connection from the pool, recording wait time and updating metrics.

        Parameters
        ----------
        command_name : str, optional
            The Redis command name (passed to parent).
        *keys
            Keys for the command (passed to parent).
        **options
            Options for getting the connection (passed to parent).

        Returns
        -------
        connection
            A Redis connection from the pool.

        Raises
        ------
        redis.ConnectionError
            If the pool is exhausted and timeout expires.
        """
        start_time = time.perf_counter()

        try:
            connection = super().get_connection(command_name, *keys, **options)
            wait_time = time.perf_counter() - start_time

            # Record successful acquisition
            POOL_WAIT_SECONDS.labels(pool_name=self._pool_name).observe(wait_time)

            with self._lock:
                self._in_use_count += 1
                POOL_CONNECTIONS_IN_USE.labels(pool_name=self._pool_name).set(self._in_use_count)

            # Update available count (pool.qsize() gives queue depth)
            self._update_available_count()

            return connection

        except redis.ConnectionError:
            # Pool exhaustion timeout - record and re-raise
            wait_time = time.perf_counter() - start_time
            POOL_WAIT_SECONDS.labels(pool_name=self._pool_name).observe(wait_time)
            POOL_TIMEOUT_TOTAL.labels(pool_name=self._pool_name).inc()
            logger.warning(
                f"Redis pool '{self._pool_name}' exhausted after {wait_time:.2f}s wait. "
                f"In use: {self._in_use_count}, Max: {self.max_connections}"
            )
            raise

    def release(self, connection: Any) -> None:
        """
        Release a connection back to the pool and update metrics.

        Parameters
        ----------
        connection
            The connection to release back to the pool.
        """
        super().release(connection)

        with self._lock:
            self._in_use_count = max(0, self._in_use_count - 1)
            POOL_CONNECTIONS_IN_USE.labels(pool_name=self._pool_name).set(self._in_use_count)

        self._update_available_count()

    def _update_available_count(self) -> None:
        """Update the available connections gauge based on pool queue size."""
        try:
            # BlockingConnectionPool uses a Queue internally
            available = self.pool.qsize() if hasattr(self, "pool") and self.pool else 0
            POOL_CONNECTIONS_AVAILABLE.labels(pool_name=self._pool_name).set(available)
        except Exception:
            # Don't let metrics collection failure affect pool operation
            pass

    def get_pool_stats(self) -> dict:
        """
        Get current pool statistics for debugging/logging.

        Returns
        -------
        dict
            Dictionary with in_use, available, max, and pool_name.
        """
        with self._lock:
            in_use = self._in_use_count

        try:
            available = self.pool.qsize() if hasattr(self, "pool") and self.pool else 0
        except Exception:
            available = 0

        return {
            "pool_name": self._pool_name,
            "in_use": in_use,
            "available": available,
            "max": self.max_connections,
            "utilization_pct": (in_use / self.max_connections * 100) if self.max_connections > 0 else 0,
        }
