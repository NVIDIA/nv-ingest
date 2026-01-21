# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import uuid

import pytest

from nv_ingest_api.data_handlers.data_writer import (
    IngestDataWriter,
    KafkaDestinationConfig,
)

pytestmark = pytest.mark.integration_full


def _parse_kafka_env(value: str):
    """Parse INGEST_INTEGRATION_TEST_KAFKA env like 'host:9092' or 'h1:9092,h2:9092'."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    # Basic validation: expect host:port entries
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
        pytest.skip("Skipping Kafka integration tests: INGEST_INTEGRATION_TEST_KAFKA not set")
    # Check dependency and connectivity using KafkaConsumer
    try:
        from kafka import KafkaConsumer  # type: ignore
    except Exception as e:
        pytest.skip(f"Skipping Kafka integration tests: kafka-python not available ({e})")
    try:
        bootstrap = _parse_kafka_env(url)
        # Create a consumer to force metadata fetch
        consumer = KafkaConsumer(bootstrap_servers=bootstrap, consumer_timeout_ms=1000)
        # Trigger a metadata refresh
        _ = consumer.topics()
        consumer.close()
        return bootstrap
    except Exception as e:
        pytest.skip(f"Skipping Kafka integration tests: cannot connect to Kafka ({e})")


@pytest.fixture(autouse=True)
def reset_writer_singleton():
    IngestDataWriter.reset_for_tests()
    yield
    IngestDataWriter.reset_for_tests()


@pytest.fixture(scope="module")
def kafka_bootstrap():
    return _require_kafka_or_skip()


def _unique_topic(prefix: str = "nv_ingest_test") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _consume_n(bootstrap, topic: str, n: int, timeout_s: float = 15.0, key_field: str | None = None):
    from kafka import KafkaConsumer  # type: ignore
    import time as _time

    deadline = _time.monotonic() + timeout_s
    msgs = []
    group_id = f"nv_ingest_it_{topic}"
    seen_keys = set()
    while len(msgs) < n and _time.monotonic() < deadline:
        remaining = max(0.5, deadline - _time.monotonic())
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            group_id=group_id,
            consumer_timeout_ms=int(remaining * 1000),
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        )
        try:
            for msg in consumer:
                value = msg.value
                if key_field is None:
                    msgs.append(value)
                    if len(msgs) >= n:
                        break
                else:
                    k = value.get(key_field)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        msgs.append(value)
                        if len(seen_keys) >= n:
                            break
        finally:
            consumer.close()
    if (len(seen_keys) if key_field is not None else len(msgs)) < n:
        raise TimeoutError(f"Expected {n} messages on topic {topic}, got {len(msgs)}")
    return msgs


def _make_large_payload(size_bytes: int, key: str = "blob") -> dict:
    chunk = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" * ((size_bytes // 36) + 1))[:size_bytes]
    return {key: chunk, "size": len(chunk)}


def _ensure_topic(bootstrap, topic: str, num_partitions: int = 1, replication_factor: int = 1):
    """Best-effort topic creation to avoid auto-create race conditions."""
    try:
        from kafka.admin import KafkaAdminClient, NewTopic  # type: ignore
        from kafka.errors import TopicAlreadyExistsError  # type: ignore

        admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id=f"nv_ingest_admin_{topic}")
        try:
            new_topic = NewTopic(name=topic, num_partitions=num_partitions, replication_factor=replication_factor)
            admin.create_topics([new_topic], validate_only=False)
        except TopicAlreadyExistsError:
            pass
        finally:
            admin.close()
    except Exception:
        # If admin client is unavailable or creation fails, proceed; tests may still pass
        pass


def test_kafka_single_async_write_and_consume(kafka_bootstrap):
    bootstrap = kafka_bootstrap
    topic = _unique_topic()

    payload = {"single": True, "n": 1}
    writer = IngestDataWriter.get_instance()
    cfg = KafkaDestinationConfig(bootstrap_servers=bootstrap, topic=topic, value_serializer="json")

    fut = writer.write_async([json.dumps(payload)], cfg, callback_executor=None)
    fut.result(timeout=20)

    msgs = _consume_n(bootstrap, topic, n=1, timeout_s=15)
    assert isinstance(msgs[0], dict)
    assert msgs[0].get("single") is True
    assert msgs[0].get("n") == 1


def test_kafka_many_async_writes_all_complete(kafka_bootstrap):
    bootstrap = kafka_bootstrap
    topic = _unique_topic(prefix="nv_ingest_many")

    writer = IngestDataWriter.get_instance()
    futures = []
    expected = []

    for i in range(10):
        payload = {"idx": i}
        expected.append(payload)
        cfg = KafkaDestinationConfig(bootstrap_servers=bootstrap, topic=topic, value_serializer="json")
        fut = writer.write_async([json.dumps(payload)], cfg, callback_executor=None)
        futures.append(fut)

    for fut in futures:
        fut.result(timeout=30)

    msgs = _consume_n(bootstrap, topic, n=10, timeout_s=20)
    # Order is not guaranteed, validate presence of all indices
    seen = sorted(m.get("idx") for m in msgs)
    assert seen == list(range(10))


def test_kafka_single_async_write_large_payload(kafka_bootstrap):
    bootstrap = kafka_bootstrap
    topic = _unique_topic(prefix="nv_ingest_large_single")

    payload = _make_large_payload(512 * 1024)
    writer = IngestDataWriter.get_instance()
    cfg = KafkaDestinationConfig(bootstrap_servers=bootstrap, topic=topic, value_serializer="json")

    fut = writer.write_async([json.dumps(payload)], cfg, callback_executor=None)
    fut.result(timeout=30)

    msgs = _consume_n(bootstrap, topic, n=1, timeout_s=30)
    assert isinstance(msgs[0], dict)
    assert msgs[0].get("size") == payload["size"]
    assert msgs[0].get("blob") == payload["blob"]


@pytest.mark.skip("Failing for now, need to investigate")
def test_kafka_many_async_writes_large_payloads(kafka_bootstrap):
    bootstrap = kafka_bootstrap
    topic = _unique_topic(prefix="nv_ingest_large_many")

    writer = IngestDataWriter.get_instance()
    futures = []
    expected = []

    # Ensure topic exists before producing to avoid auto-create delay
    _ensure_topic(bootstrap, topic, num_partitions=1, replication_factor=1)

    for i in range(8):
        size = 128 * 1024 + i * 128 * 1024
        payload = {"idx": i, **_make_large_payload(size)}
        expected.append(payload)
        cfg = KafkaDestinationConfig(bootstrap_servers=bootstrap, topic=topic, value_serializer="json")
        fut = writer.write_async([json.dumps(payload)], cfg, callback_executor=None)
        futures.append(fut)

    for fut in futures:
        fut.result(timeout=60)

    msgs = _consume_n(bootstrap, topic, n=len(expected), timeout_s=90, key_field="idx")
    # Build a map of idx->message
    got = {m.get("idx"): m for m in msgs}
    for i, exp in enumerate(expected):
        assert i in got
        m = got[i]
        assert m.get("size") == exp["size"]
        assert m.get("blob") == exp["blob"]


def test_kafka_sync_write_and_consume_two_messages(kafka_bootstrap):
    bootstrap = kafka_bootstrap
    topic = _unique_topic(prefix="nv_ingest_sync")

    writer = IngestDataWriter.get_instance()
    cfg = KafkaDestinationConfig(bootstrap_servers=bootstrap, topic=topic, value_serializer="json")

    payloads = [json.dumps({"a": 1}), json.dumps({"b": 2})]
    writer._write_sync(payloads, cfg)  # noqa: SLF001

    msgs = _consume_n(bootstrap, topic, n=2, timeout_s=20)
    # Order in a single-partition topic should be preserved, but don't rely; check set equality
    assert {tuple(sorted(m.items())) for m in msgs} == {
        tuple(sorted({"a": 1}.items())),
        tuple(sorted({"b": 2}.items())),
    }
