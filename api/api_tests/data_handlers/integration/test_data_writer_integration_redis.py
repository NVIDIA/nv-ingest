# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import uuid

import pytest

from nv_ingest_api.data_handlers.data_writer import (
    IngestDataWriter,
    RedisDestinationConfig,
)
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

pytestmark = pytest.mark.integration_full


def _parse_redis_env(value: str):
    """Parse INGEST_INTEGRATION_TEST_REDIS env value like 'host:port' or 'host:port/db'."""
    host = value
    port = 6379
    db = 0
    if ":" in value:
        host_part, rest = value.split(":", 1)
        host = host_part
        if "/" in rest:
            port_part, db_part = rest.split("/", 1)
            port = int(port_part)
            db = int(db_part)
        else:
            port = int(rest)
    return host, port, db


def _require_redis_or_skip():
    url = os.getenv("INGEST_INTEGRATION_TEST_REDIS")
    if not url:
        pytest.skip("Skipping Redis integration tests: INGEST_INTEGRATION_TEST_REDIS not set")
    try:
        host, port, db = _parse_redis_env(url)
        client = RedisClient(host=host, port=port, db=db)
        # Basic connectivity check
        if not client.ping():
            pytest.skip(f"Skipping Redis integration tests: redis ping failed for {host}:{port}/{db}")
        return host, port, db
    except Exception as e:
        pytest.skip(f"Skipping Redis integration tests: cannot connect to Redis ({e})")


@pytest.fixture(autouse=True)
def reset_writer_singleton():
    IngestDataWriter.reset_for_tests()
    yield
    IngestDataWriter.reset_for_tests()


@pytest.fixture(scope="module")
def redis_target():
    return _require_redis_or_skip()


def _unique_channel(prefix: str = "nv_ingest_test") -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def _fetch_one(host: str, port: int, db: int, channel: str, timeout: float = 5.0):
    client = RedisClient(host=host, port=port, db=db)
    msg = client.fetch_message(channel, timeout=timeout)
    return msg


def _cleanup_channel(host: str, port: int, db: int, channel: str):
    """Best-effort deletion of the Redis list key used for tests."""
    try:
        client = RedisClient(host=host, port=port, db=db)
        client.get_client().delete(channel)
    except Exception:
        # Do not fail tests on cleanup errors
        pass


def test_redis_single_async_write_and_fetch(redis_target):
    host, port, db = redis_target
    channel = _unique_channel()

    payload = {"single": True, "n": 1}
    writer = IngestDataWriter.get_instance()
    cfg = RedisDestinationConfig(host=host, port=port, db=db, password=None, channel=channel)

    fut = writer.write_async([json.dumps(payload)], cfg, callback_executor=None)
    fut.result(timeout=10)

    try:
        msg = _fetch_one(host, port, db, channel, timeout=5)
        assert isinstance(msg, dict)
        assert msg.get("single") is True
        assert msg.get("n") == 1
    finally:
        _cleanup_channel(host, port, db, channel)


def test_redis_many_async_writes_all_complete(redis_target):
    host, port, db = redis_target
    writer = IngestDataWriter.get_instance()

    futures = []
    channels = []
    payloads = []

    for i in range(10):
        channel = _unique_channel(prefix="nv_ingest_many")
        channels.append(channel)
        payload = {"idx": i}
        payloads.append(payload)
        cfg = RedisDestinationConfig(host=host, port=port, db=db, password=None, channel=channel)
        fut = writer.write_async([json.dumps(payload)], cfg, callback_executor=None)
        futures.append(fut)

    # Wait all
    for fut in futures:
        fut.result(timeout=10)

    # Validate each
    try:
        for i, ch in enumerate(channels):
            msg = _fetch_one(host, port, db, ch, timeout=5)
            assert isinstance(msg, dict)
            assert msg.get("idx") == i
    finally:
        for ch in channels:
            _cleanup_channel(host, port, db, ch)


def test_redis_sync_write_and_fetch_two_messages(redis_target):
    host, port, db = redis_target
    writer = IngestDataWriter.get_instance()

    channel = _unique_channel(prefix="nv_ingest_sync")
    cfg = RedisDestinationConfig(host=host, port=port, db=db, password=None, channel=channel)

    payloads = [json.dumps({"a": 1}), json.dumps({"b": 2})]
    # Use synchronous path to write both
    writer._write_sync(payloads, cfg)  # noqa: SLF001 (testing internal path for integration)

    try:
        # Fetch both messages
        first = _fetch_one(host, port, db, channel, timeout=5)
        second = _fetch_one(host, port, db, channel, timeout=5)

        # Order is preserved by RPUSH/BLPOP; verify contents
        assert first == {"a": 1}
        assert second == {"b": 2}
    finally:
        _cleanup_channel(host, port, db, channel)


def _make_large_payload(size_bytes: int, key: str = "blob") -> dict:
    # Create a deterministic large string payload
    chunk = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" * ((size_bytes // 36) + 1))[:size_bytes]
    return {key: chunk, "size": len(chunk)}


def test_redis_single_async_write_large_payload(redis_target):
    host, port, db = redis_target
    channel = _unique_channel(prefix="nv_ingest_large_single")

    # ~512 KiB payload
    payload = _make_large_payload(512 * 1024)
    writer = IngestDataWriter.get_instance()
    cfg = RedisDestinationConfig(host=host, port=port, db=db, password=None, channel=channel)

    fut = writer.write_async([json.dumps(payload)], cfg, callback_executor=None)
    fut.result(timeout=20)

    try:
        msg = _fetch_one(host, port, db, channel, timeout=10)
        assert isinstance(msg, dict)
        assert msg.get("size") == payload["size"]
        assert msg.get("blob") == payload["blob"]
    finally:
        _cleanup_channel(host, port, db, channel)


def test_redis_many_async_writes_large_payloads_all_complete(redis_target):
    host, port, db = redis_target
    writer = IngestDataWriter.get_instance()

    futures = []
    channels = []
    expected = []

    # Ten payloads ranging from 128 KiB to 1.25 MiB
    for i in range(10):
        channel = _unique_channel(prefix="nv_ingest_large_many")
        channels.append(channel)
        size = 128 * 1024 + i * 128 * 1024
        payload = {"idx": i, **_make_large_payload(size)}
        expected.append(payload)
        cfg = RedisDestinationConfig(host=host, port=port, db=db, password=None, channel=channel)
        fut = writer.write_async([json.dumps(payload)], cfg, callback_executor=None)
        futures.append(fut)

    for fut in futures:
        fut.result(timeout=30)

    try:
        for i, ch in enumerate(channels):
            msg = _fetch_one(host, port, db, ch, timeout=15)
            assert isinstance(msg, dict)
            assert msg.get("idx") == i
            assert msg.get("size") == expected[i]["size"]
            assert msg.get("blob") == expected[i]["blob"]
    finally:
        for ch in channels:
            _cleanup_channel(host, port, db, ch)


def test_redis_sync_write_and_fetch_two_large_messages(redis_target):
    host, port, db = redis_target
    writer = IngestDataWriter.get_instance()

    channel = _unique_channel(prefix="nv_ingest_sync_large")
    cfg = RedisDestinationConfig(host=host, port=port, db=db, password=None, channel=channel)

    p1 = _make_large_payload(256 * 1024, key="d1")
    p2 = _make_large_payload(384 * 1024, key="d2")
    payloads = [json.dumps(p1), json.dumps(p2)]

    writer._write_sync(payloads, cfg)  # noqa: SLF001

    try:
        first = _fetch_one(host, port, db, channel, timeout=10)
        second = _fetch_one(host, port, db, channel, timeout=10)

        assert first.get("d1") == p1["d1"]
        assert first.get("size") == p1["size"]
        assert second.get("d2") == p2["d2"]
        assert second.get("size") == p2["size"]
    finally:
        _cleanup_channel(host, port, db, channel)
