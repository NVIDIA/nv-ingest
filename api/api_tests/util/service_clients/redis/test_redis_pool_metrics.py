# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest
import redis

from nv_ingest_api.util.service_clients.redis.redis_pool_metrics import (
    InstrumentedBlockingConnectionPool,
    POOL_CONNECTIONS_IN_USE,
    POOL_CONNECTIONS_AVAILABLE,
    POOL_CONNECTIONS_MAX,
    POOL_TIMEOUT_TOTAL,
    POOL_WAIT_SECONDS,
)

POOL_NAME = "test_pool"


@pytest.fixture(autouse=True)
def _reset_metrics():
    for metric in (POOL_CONNECTIONS_IN_USE, POOL_CONNECTIONS_AVAILABLE, POOL_CONNECTIONS_MAX):
        metric.labels(pool_name=POOL_NAME).set(0)
    POOL_TIMEOUT_TOTAL.labels(pool_name=POOL_NAME)._value.set(0)
    POOL_WAIT_SECONDS.remove(POOL_NAME)
    yield


def _make_pool(max_connections=4, timeout=1):
    def fake_init(self_inner, **kwargs):
        self_inner.max_connections = kwargs.get("max_connections", 50)
        self_inner.timeout = kwargs.get("timeout", 20)
        self_inner.connection_kwargs = {}
        self_inner._connections = []
        self_inner.pool = queue.Queue(maxsize=self_inner.max_connections)
        for _ in range(self_inner.max_connections):
            self_inner.pool.put_nowait(MagicMock(name="mock_connection"))

    with patch.object(redis.BlockingConnectionPool, "__init__", fake_init):
        pool = InstrumentedBlockingConnectionPool(pool_name=POOL_NAME, max_connections=max_connections, timeout=timeout)
    return pool


class TestInit:
    def test_sets_max_gauge(self):
        pool = _make_pool(max_connections=10)
        assert POOL_CONNECTIONS_MAX.labels(pool_name=POOL_NAME)._value.get() == 10
        assert pool._pool_name == POOL_NAME


class TestGetConnection:
    def test_updates_gauges_on_acquire(self):
        pool = _make_pool(max_connections=4)

        with patch.object(redis.BlockingConnectionPool, "get_connection", side_effect=lambda *a, **kw: pool.pool.get()):
            pool.get_connection("SET")

        assert POOL_CONNECTIONS_IN_USE.labels(pool_name=POOL_NAME)._value.get() == 1
        assert POOL_CONNECTIONS_AVAILABLE.labels(pool_name=POOL_NAME)._value.get() == 3

    def test_records_wait_time(self):
        pool = _make_pool(max_connections=4)

        with patch.object(redis.BlockingConnectionPool, "get_connection", side_effect=lambda *a, **kw: pool.pool.get()):
            pool.get_connection("SET")
            pool.get_connection("SET")

        hist = POOL_WAIT_SECONDS.labels(pool_name=POOL_NAME)
        assert hist._sum.get() > 0
        # First bucket (<=1ms) should hold both observations since mock returns instantly
        assert hist._buckets[0].get() == 2

    def test_timeout_increments_counter(self):
        pool = _make_pool(max_connections=4)

        with patch.object(
            redis.BlockingConnectionPool, "get_connection", side_effect=redis.ConnectionError("pool exhausted")
        ):
            with pytest.raises(redis.ConnectionError):
                pool.get_connection("SET")

        assert POOL_TIMEOUT_TOTAL.labels(pool_name=POOL_NAME)._value.get() == 1

    def test_multiple_acquire_release_cycle(self):
        pool = _make_pool(max_connections=4)

        conns = []
        with patch.object(redis.BlockingConnectionPool, "get_connection", side_effect=lambda *a, **kw: pool.pool.get()):
            for _ in range(3):
                conns.append(pool.get_connection("SET"))

        assert POOL_CONNECTIONS_IN_USE.labels(pool_name=POOL_NAME)._value.get() == 3
        assert POOL_CONNECTIONS_AVAILABLE.labels(pool_name=POOL_NAME)._value.get() == 1

        with patch.object(redis.BlockingConnectionPool, "release", side_effect=lambda conn: pool.pool.put_nowait(conn)):
            for conn in conns:
                pool.release(conn)

        assert POOL_CONNECTIONS_IN_USE.labels(pool_name=POOL_NAME)._value.get() == 0
        assert POOL_CONNECTIONS_AVAILABLE.labels(pool_name=POOL_NAME)._value.get() == 4


class TestRelease:
    def test_updates_gauges_on_release(self):
        pool = _make_pool(max_connections=4)

        with patch.object(redis.BlockingConnectionPool, "get_connection", side_effect=lambda *a, **kw: pool.pool.get()):
            conn = pool.get_connection("SET")

        assert POOL_CONNECTIONS_IN_USE.labels(pool_name=POOL_NAME)._value.get() == 1

        with patch.object(redis.BlockingConnectionPool, "release", side_effect=lambda conn: pool.pool.put_nowait(conn)):
            pool.release(conn)

        assert POOL_CONNECTIONS_IN_USE.labels(pool_name=POOL_NAME)._value.get() == 0
        assert POOL_CONNECTIONS_AVAILABLE.labels(pool_name=POOL_NAME)._value.get() == 4


class TestGetPoolStats:
    def test_returns_correct_stats(self):
        pool = _make_pool(max_connections=4)

        with patch.object(redis.BlockingConnectionPool, "get_connection", side_effect=lambda *a, **kw: pool.pool.get()):
            pool.get_connection("SET")
            pool.get_connection("SET")

        stats = pool.get_pool_stats()
        assert stats["pool_name"] == POOL_NAME
        assert stats["in_use"] == 2
        assert stats["available"] == 2
        assert stats["max"] == 4
        assert stats["utilization_pct"] == pytest.approx(50.0)

    def test_empty_pool_stats(self):
        pool = _make_pool(max_connections=4)
        stats = pool.get_pool_stats()
        assert stats["in_use"] == 0
        assert stats["available"] == 4
        assert stats["utilization_pct"] == pytest.approx(0.0)


class TestConcurrency:
    def test_concurrent_acquire_release(self):
        pool = _make_pool(max_connections=8)

        errors = []
        barrier = threading.Barrier(4)

        def worker():
            try:
                conn = pool.get_connection("SET")
                barrier.wait(timeout=5)
                pool.release(conn)
            except Exception as e:
                errors.append(e)

        with patch.object(
            redis.BlockingConnectionPool, "get_connection", side_effect=lambda *a, **kw: pool.pool.get()
        ), patch.object(redis.BlockingConnectionPool, "release", side_effect=lambda c: pool.pool.put_nowait(c)):
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert not errors
        assert pool.pool.qsize() == 8
        assert pool.get_pool_stats()["in_use"] == 0
