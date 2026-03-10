# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrumented Redis BlockingConnectionPool with Prometheus metrics."""

import logging
import time
from typing import Any, Optional

import redis
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

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
        super().__init__(**kwargs)
        self._pool_name = pool_name
        POOL_CONNECTIONS_MAX.labels(pool_name=self._pool_name).set(self.max_connections)

    def _get_in_use_count(self) -> int:
        return self.max_connections - self.pool.qsize()

    def get_connection(self, command_name: Optional[str] = None, *keys: Any, **options: Any) -> Any:
        start_time = time.perf_counter()

        try:
            connection = super().get_connection(command_name, *keys, **options)
            wait_time = time.perf_counter() - start_time

            POOL_WAIT_SECONDS.labels(pool_name=self._pool_name).observe(wait_time)
            self._update_gauges()

            return connection

        except redis.ConnectionError:
            wait_time = time.perf_counter() - start_time
            POOL_WAIT_SECONDS.labels(pool_name=self._pool_name).observe(wait_time)
            POOL_TIMEOUT_TOTAL.labels(pool_name=self._pool_name).inc()
            in_use = self._get_in_use_count()
            logger.warning(
                f"Redis pool '{self._pool_name}' exhausted after {wait_time:.2f}s wait. "
                f"In use: {in_use}, Max: {self.max_connections}"
            )
            raise

    def release(self, connection: Any) -> None:
        super().release(connection)
        self._update_gauges()

    def _update_gauges(self) -> None:
        available = self.pool.qsize()
        in_use = self.max_connections - available
        POOL_CONNECTIONS_IN_USE.labels(pool_name=self._pool_name).set(in_use)
        POOL_CONNECTIONS_AVAILABLE.labels(pool_name=self._pool_name).set(available)

    def get_pool_stats(self) -> dict:
        available = self.pool.qsize()
        in_use = self.max_connections - available
        return {
            "pool_name": self._pool_name,
            "in_use": in_use,
            "available": available,
            "max": self.max_connections,
            "utilization_pct": (in_use / self.max_connections * 100) if self.max_connections > 0 else 0,
        }
