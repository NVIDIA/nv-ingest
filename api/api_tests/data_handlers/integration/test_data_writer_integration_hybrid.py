# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import uuid
import tempfile
import threading

import pytest

from nv_ingest_api.data_handlers.data_writer import (
    IngestDataWriter,
    FilesystemDestinationConfig,
    RedisDestinationConfig,
    KafkaDestinationConfig,
)
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

pytestmark = pytest.mark.integration_full


# ---------- Redis env helpers ----------


def _parse_redis_env(value: str):
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
        pytest.skip("Skipping hybrid integration tests: INGEST_INTEGRATION_TEST_REDIS not set")
    try:
        host, port, db = _parse_redis_env(url)
        client = RedisClient(host=host, port=port, db=db)
        if not client.ping():
            pytest.skip(f"Skipping hybrid integration tests: redis ping failed for {host}:{port}/{db}")
        return host, port, db
    except Exception as e:
        pytest.skip(f"Skipping hybrid integration tests: cannot connect to Redis ({e})")


# ---------- Pytest fixtures ----------


@pytest.fixture(autouse=True)
def reset_writer_singleton():
    IngestDataWriter.reset_for_tests()
    yield
    IngestDataWriter.reset_for_tests()


@pytest.fixture(scope="module")
def redis_target():
    return _require_redis_or_skip()


# ---------- Utility helpers ----------


def _unique_channel(prefix: str = "nv_ingest_hybrid") -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def _cleanup_channel(host: str, port: int, db: int, channel: str):
    try:
        client = RedisClient(host=host, port=port, db=db)
        client.get_client().delete(channel)
    except Exception:
        pass


# ---------- Kafka helpers ----------


def _parse_kafka_env(value: str):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    servers = []
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Invalid bootstrap server entry: {p}")
        host, port = p.split(":", 1)
        servers.append(f"{host}:{int(port)}")
    return servers


def _require_kafka_or_skip():
    url = os.getenv("INGEST_INTEGRATION_TEST_KAFKA")
    if not url:
        pytest.skip("Skipping Kafka hybrid tests: INGEST_INTEGRATION_TEST_KAFKA not set")
    try:
        from kafka import KafkaConsumer  # type: ignore
    except Exception as e:
        pytest.skip(f"Skipping Kafka hybrid tests: kafka-python not available ({e})")
    try:
        bootstrap = _parse_kafka_env(url)
        consumer = KafkaConsumer(bootstrap_servers=bootstrap, consumer_timeout_ms=1000)
        _ = consumer.topics()
        consumer.close()
        return bootstrap
    except Exception as e:
        pytest.skip(f"Skipping Kafka hybrid tests: cannot connect to Kafka ({e})")


def _unique_topic(prefix: str = "nv_ingest_hybrid") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _consume_n(bootstrap, topic: str, n: int, timeout_s: float = 20.0):
    from kafka import KafkaConsumer  # type: ignore

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=int(timeout_s * 1000),
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    msgs = []
    try:
        for msg in consumer:
            msgs.append(msg.value)
            if len(msgs) >= n:
                break
    finally:
        consumer.close()
    if len(msgs) < n:
        raise TimeoutError(f"Expected {n} messages on topic {topic}, got {len(msgs)}")
    return msgs


def _fetch_one(host: str, port: int, db: int, channel: str, timeout: float = 10.0):
    client = RedisClient(host=host, port=port, db=db)
    return client.fetch_message(channel, timeout=timeout)


def _cleanup_file(path: str) -> None:
    try:
        tmp_root = os.path.realpath(tempfile.gettempdir())
        target = os.path.realpath(path)
        if not target.startswith(tmp_root + os.sep):
            return
        if os.path.isdir(target):
            return
        if os.path.exists(target):
            os.remove(target)
    except Exception:
        pass


# ---------- Tests ----------


def test_hybrid_success_callbacks_write_status_to_redis(redis_target):
    host, port, db = redis_target
    writer = IngestDataWriter.get_instance()

    futures = []
    channels = []
    paths = []

    def make_success_cb(channel_name: str):
        def on_success(data_payload, dest_cfg):
            # Send a success status to Redis via data_writer using Redis writer strategy
            status_msg = {"status": "success", "path": dest_cfg.path}
            status_cfg = RedisDestinationConfig(host=host, port=port, db=db, password=None, channel=channel_name)
            # Fire-and-forget; callback runs in worker thread here (callback_executor=None in write_async)
            writer.write_async([json.dumps(status_msg)], status_cfg, callback_executor=None)

        return on_success

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 writes
        for i in range(5):
            out_path = os.path.join(tmpdir, f"hybrid_{i}.json")
            paths.append(out_path)
            ch = _unique_channel(prefix="nv_ingest_hybrid_ok")
            channels.append(ch)

            fs_cfg = FilesystemDestinationConfig(path=out_path)
            payloads = [json.dumps({"i": i}), json.dumps({"ok": True})]

            fut = writer.write_async(
                payloads,
                fs_cfg,
                on_success=make_success_cb(ch),
                callback_executor=None,
            )
            futures.append(fut)

        # Wait
        for fut in futures:
            fut.result(timeout=15)

        # Validate files and redis statuses
        try:
            for i, p in enumerate(paths):
                assert os.path.exists(p)
                with open(p, "r") as f:
                    data = json.load(f)
                assert data == [{"i": i}, {"ok": True}]

            for i, ch in enumerate(channels):
                msg = _fetch_one(host, port, db, ch, timeout=10)
                assert isinstance(msg, dict)
                assert msg.get("status") == "success"
                # path must match the destination path written
                assert msg.get("path") == paths[i]
        finally:
            for p in paths:
                _cleanup_file(p)
            for ch in channels:
                _cleanup_channel(host, port, db, ch)


def test_hybrid_failure_callbacks_write_status_to_redis(redis_target):
    host, port, db = redis_target
    writer = IngestDataWriter.get_instance()

    # Force a failure by writing to a directory path instead of a file
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "dir_instead_of_file")
        os.makedirs(out_dir)

        channel = _unique_channel(prefix="nv_ingest_hybrid_fail")
        failure_called = threading.Event()

        def on_failure(data_payload, dest_cfg, exc):
            failure_called.set()
            status_msg = {"status": "failed", "path": dest_cfg.path, "error": str(exc)[:256]}
            status_cfg = RedisDestinationConfig(host=host, port=port, db=db, password=None, channel=channel)
            writer.write_async([json.dumps(status_msg)], status_cfg, callback_executor=None)

        fs_cfg = FilesystemDestinationConfig(path=out_dir)
        fut = writer.write_async([json.dumps({"x": 1})], fs_cfg, on_failure=on_failure, callback_executor=None)
        fut.result(timeout=10)

        try:
            assert failure_called.is_set()
            # Directory should still exist, and no file was created
            assert os.path.isdir(out_dir)

            msg = _fetch_one(host, port, db, channel, timeout=10)
            assert isinstance(msg, dict)
            assert msg.get("status") == "failed"
            assert msg.get("path") == out_dir
            assert "error" in msg
        finally:
            _cleanup_channel(host, port, db, channel)


def test_hybrid_success_callbacks_write_status_to_kafka():
    bootstrap = _require_kafka_or_skip()
    writer = IngestDataWriter.get_instance()

    topic = _unique_topic(prefix="nv_ingest_hybrid_ok_kafka")

    def make_success_cb():
        def on_success(data_payload, dest_cfg):
            status_msg = {"status": "success", "path": dest_cfg.path}
            kcfg = KafkaDestinationConfig(bootstrap_servers=bootstrap, topic=topic, value_serializer="json")
            # Await nested write to ensure message is visible before test assertions
            nested = writer.write_async([json.dumps(status_msg)], kcfg, callback_executor=None)
            nested.result(timeout=20)

        return on_success

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "hybrid_k_ok.json")
        fs_cfg = FilesystemDestinationConfig(path=out_path)
        payloads = [json.dumps({"ok": True})]

        fut = writer.write_async(payloads, fs_cfg, on_success=make_success_cb(), callback_executor=None)
        fut.result(timeout=15)

        try:
            assert os.path.exists(out_path)
            with open(out_path, "r") as f:
                data = json.load(f)
            assert data == [{"ok": True}]

            msgs = _consume_n(bootstrap, topic, n=1, timeout_s=30)
            assert msgs[0].get("status") == "success"
            assert msgs[0].get("path") == out_path
        finally:
            _cleanup_file(out_path)


def test_hybrid_failure_callbacks_write_status_to_kafka():
    bootstrap = _require_kafka_or_skip()
    writer = IngestDataWriter.get_instance()

    topic = _unique_topic(prefix="nv_ingest_hybrid_fail_kafka")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "dir_instead_of_file")
        os.makedirs(out_dir)

        def on_failure(data_payload, dest_cfg, exc):
            status_msg = {"status": "failed", "path": dest_cfg.path, "error": str(exc)[:256]}
            kcfg = KafkaDestinationConfig(bootstrap_servers=bootstrap, topic=topic, value_serializer="json")
            nested = writer.write_async([json.dumps(status_msg)], kcfg, callback_executor=None)
            nested.result(timeout=20)

        fs_cfg = FilesystemDestinationConfig(path=out_dir)
        fut = writer.write_async([json.dumps({"x": 1})], fs_cfg, on_failure=on_failure, callback_executor=None)
        fut.result(timeout=15)

        # Directory still exists
        assert os.path.isdir(out_dir)

        msgs = _consume_n(bootstrap, topic, n=1, timeout_s=30)
        assert msgs[0].get("status") == "failed"
        assert msgs[0].get("path") == out_dir
        assert "error" in msgs[0]


def test_hybrid_kafka_write_success_reports_to_redis(redis_target):
    host, port, db = redis_target
    bootstrap = _require_kafka_or_skip()
    writer = IngestDataWriter.get_instance()

    topic = _unique_topic(prefix="nv_ingest_kafka_to_redis")
    channel = _unique_channel(prefix="nv_ingest_kafka_status")

    success_evt = threading.Event()

    def on_success(data_payload, dest_cfg):
        success_evt.set()
        status_msg = {"status": "success", "topic": dest_cfg.topic}
        rcfg = RedisDestinationConfig(host=host, port=port, db=db, password=None, channel=channel)
        writer.write_async([json.dumps(status_msg)], rcfg, callback_executor=None)

    kcfg = KafkaDestinationConfig(bootstrap_servers=bootstrap, topic=topic, value_serializer="json")
    fut = writer.write_async([json.dumps({"msg": 1})], kcfg, on_success=on_success, callback_executor=None)
    fut.result(timeout=20)

    try:
        assert success_evt.is_set()
        # Verify Kafka received the message
        msgs = _consume_n(bootstrap, topic, n=1, timeout_s=20)
        assert msgs[0].get("msg") == 1

        # Verify Redis recorded the success status
        status = _fetch_one(host, port, db, channel, timeout=10)
        assert status.get("status") == "success"
        assert status.get("topic") == topic
    finally:
        _cleanup_channel(host, port, db, channel)
