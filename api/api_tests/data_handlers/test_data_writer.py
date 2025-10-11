# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
import threading
from unittest.mock import Mock, patch

from nv_ingest_api.data_handlers.data_writer import (
    IngestDataWriter,
    classify_error,
    RedisDestinationConfig,
    FilesystemDestinationConfig,
    HttpDestinationConfig,
    KafkaDestinationConfig,
)
from nv_ingest_api.data_handlers.errors import (
    TransientError,
    PermanentError,
    ConnectionError as DWConnectionError,
    AuthenticationError,
)


class TestIngestDataWriter:
    """Black-box tests for IngestDataWriter behavior."""

    def setup_method(self):
        # Ensure clean singleton state for each test
        IngestDataWriter.reset_for_tests()

    def teardown_method(self):
        IngestDataWriter.reset_for_tests()

    def test_singleton_get_instance_and_reset(self):
        w1 = IngestDataWriter.get_instance(max_workers=1)
        w2 = IngestDataWriter.get_instance(max_workers=8)
        assert w1 is w2

        IngestDataWriter.reset_for_tests()
        w3 = IngestDataWriter.get_instance(max_workers=2)
        assert w3 is not w1

    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_write_async_success_invokes_success_callback(self, mock_get_strategy):
        writer_strategy = Mock()
        writer_strategy.write = Mock(return_value=None)
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = FilesystemDestinationConfig(path="/tmp/ok.json")
        payload = [json.dumps({"a": 1})]

        success_called = threading.Event()
        failure_called = threading.Event()

        def on_success(data, config):
            assert data == payload
            assert config is cfg
            success_called.set()

        def on_failure(data, config, exc):
            failure_called.set()

        fut = writer.write_async(payload, cfg, on_success=on_success, on_failure=on_failure, callback_executor=None)
        fut.result(timeout=2)

        assert writer_strategy.write.call_count == 1
        assert success_called.is_set()
        assert not failure_called.is_set()

    @patch("nv_ingest_api.data_handlers.data_writer.time.sleep", return_value=None)
    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_retry_on_transient_then_success(self, mock_get_strategy, _):
        # First call raises transient (e.g., built-in ConnectionError), second succeeds
        writer_strategy = Mock()
        writer_strategy.write = Mock(side_effect=[ConnectionError("timeout"), None])
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = FilesystemDestinationConfig(path="/tmp/retry.json")
        cfg.retry_count = 2  # allow retry

        success_called = threading.Event()

        fut = writer.write_async(
            [json.dumps({"x": 1})], cfg, on_success=lambda *_: success_called.set(), callback_executor=None
        )
        fut.result(timeout=2)

        assert writer_strategy.write.call_count == 2
        assert success_called.is_set()

    # --- Additional coverage for writer orchestration ---

    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_write_async_cancel_triggers_cancel_callback(self, mock_get_strategy):
        """Cancelling the result future should invoke the cancellation callback path without error."""

        # Strategy that sleeps briefly to keep write task running
        def slow_write(*_args, **_kwargs):
            time.sleep(0.05)

        writer_strategy = Mock()
        writer_strategy.write = Mock(side_effect=slow_write)
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = FilesystemDestinationConfig(path="/tmp/cancel.json")

        fut = writer.write_async([json.dumps({"c": 1})], cfg, callback_executor=None)
        # Cancel immediately; this covers the result_future cancel callback wiring
        fut.cancel()
        # Ensure we can wait without exceptions; the internal write may still finish
        try:
            fut.result(timeout=1)
        except Exception:
            pass

    def test_handle_write_result_uses_executor_submit(self):
        """Ensure _handle_write_result schedules callback on provided executor via submit()."""
        writer = IngestDataWriter.get_instance()
        cfg = FilesystemDestinationConfig(path="/tmp/exec.json")
        payload = [json.dumps({"e": 1})]

        # Build a completed future to pass into _handle_write_result
        from concurrent.futures import Future

        done_future = Future()
        done_future.set_result(None)

        # Mock executor with submit()
        exec_mock = Mock()
        writer._handle_write_result(
            done_future, payload, cfg, on_success=lambda *_: None, on_failure=None, callback_executor=exec_mock
        )
        exec_mock.submit.assert_called()

    def test_shutdown(self):
        writer = IngestDataWriter.get_instance()
        writer.shutdown()
        # No exception means path executed; new instance can be created
        w2 = IngestDataWriter.get_instance()
        assert w2 is not None

    def test_dependency_check_helpers(self):
        """Exercise _check_* availability helpers by injecting/removing modules."""
        import sys

        modname = "kafka"
        # Ensure absent
        sys.modules.pop(modname, None)
        from nv_ingest_api.data_handlers import data_writer as dw

        assert dw._check_kafka_available() in (False, True)  # Not enforcing specific env state

        # Inject dummy and verify available path
        class DummyKafka:
            pass

        sys.modules[modname] = DummyKafka()
        try:
            assert dw._check_kafka_available() is True
        finally:
            sys.modules.pop(modname, None)

    @patch("nv_ingest_api.data_handlers.data_writer.time.sleep", return_value=None)
    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_transient_retry_exhaustion_invokes_failure(self, mock_get_strategy, _):
        writer_strategy = Mock()
        writer_strategy.write = Mock(side_effect=[ConnectionError("network down"), ConnectionError("network down")])
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = FilesystemDestinationConfig(path="/tmp/fail.json")
        cfg.retry_count = 1

        failure_holder = {}
        failure_called = threading.Event()

        def on_failure(data, config, exc):
            failure_holder["exc"] = exc
            failure_called.set()

        fut = writer.write_async([json.dumps({"y": 1})], cfg, on_failure=on_failure, callback_executor=None)
        fut.result(timeout=2)

        # write attempted twice, failure callback invoked with classified error
        assert writer_strategy.write.call_count == 2
        assert failure_called.is_set()
        assert isinstance(failure_holder["exc"], TransientError)

    @patch("nv_ingest_api.data_handlers.data_writer.time.sleep", return_value=None)
    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_permanent_error_no_retry_and_failure_callback(self, mock_get_strategy, _):
        # Raise a PermanentError from strategy
        writer_strategy = Mock()
        writer_strategy.write = Mock(side_effect=PermanentError("bad request"))
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = HttpDestinationConfig(url="https://api", method="POST")
        cfg.retry_count = 3  # should still not retry

        failure_holder = {}
        failure_called = threading.Event()

        def on_failure(data, config, exc):
            failure_holder["exc"] = exc
            failure_called.set()

        fut = writer.write_async([json.dumps({"z": 1})], cfg, on_failure=on_failure, callback_executor=None)
        fut.result(timeout=2)

        assert writer_strategy.write.call_count == 1
        assert failure_called.is_set()
        assert isinstance(failure_holder["exc"], PermanentError)

    @patch("nv_ingest_api.data_handlers.data_writer.create_backoff_strategy")
    @patch("nv_ingest_api.data_handlers.data_writer.time.sleep", return_value=None)
    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_backoff_strategy_resolution_and_delay_usage(self, mock_get_strategy, _sleep, mock_create):
        # Make a stub backoff strategy that returns controlled delays
        class StubBackoff:
            def calculate_delay(self, attempt):
                return {0: 0.01, 1: 0.02}.get(attempt, 0.03)

        mock_create.return_value = StubBackoff()

        writer_strategy = Mock()
        writer_strategy.write = Mock(side_effect=[ConnectionError("timeout"), None])
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = FilesystemDestinationConfig(path="/tmp/backoff.json")
        cfg.retry_count = 2
        cfg.backoff_strategy = "fixed"  # ensure we pass through this string

        fut = writer.write_async([json.dumps({"a": 1})], cfg, callback_executor=None)
        fut.result(timeout=2)

        # create_backoff_strategy must be called with our string
        mock_create.assert_called_with("fixed")
        assert writer_strategy.write.call_count == 2
        # Ensure we attempted to sleep using attempt 0 delay first
        _sleep.assert_any_call(0.01)

    # classify_error black-box tests
    def test_classify_error_transient_connection_keywords(self):
        err = classify_error(Exception("connection reset by peer"), destination_type="filesystem")
        assert isinstance(err, DWConnectionError)

    def test_classify_error_authentication_http(self):
        class Resp:
            status_code = 401

        e = Exception("auth fail")
        e.response = Resp()
        out = classify_error(e, destination_type="http")
        assert isinstance(out, AuthenticationError)

    def test_classify_error_http_client_vs_server(self):
        class Resp:
            status_code = 404

        e1 = Exception("not found")
        e1.response = Resp()
        out1 = classify_error(e1, destination_type="http")
        assert isinstance(out1, PermanentError)

        class Resp5:
            status_code = 503

        e2 = Exception("unavailable")
        e2.response = Resp5()
        out2 = classify_error(e2, destination_type="http")
        assert isinstance(out2, TransientError)

    def test_classify_error_default_transient(self):
        out = classify_error(Exception("weird"), destination_type="redis")
        assert isinstance(out, TransientError)

    def test_classify_error_passthrough_existing_classes(self):
        """If a strategy raises PermanentError/TransientError directly, classify_error returns it unchanged."""
        perr = PermanentError("perm")
        terr = TransientError("tran")
        assert classify_error(perr, destination_type="http") is perr
        assert classify_error(terr, destination_type="kafka") is terr

    # --- Redis and Kafka specific path coverage ---

    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_redis_path_success(self, mock_get_strategy):
        """Ensure RedisDestinationConfig flows through write_async and triggers success callback."""
        writer_strategy = Mock()
        writer_strategy.write = Mock(return_value=None)
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = RedisDestinationConfig(channel="chan")
        payload = [json.dumps({"r": 1})]

        success_called = threading.Event()

        fut = writer.write_async(payload, cfg, on_success=lambda *_: success_called.set(), callback_executor=None)
        fut.result(timeout=2)

        assert writer_strategy.write.call_count == 1
        assert success_called.is_set()

    @patch("nv_ingest_api.data_handlers.data_writer.time.sleep", return_value=None)
    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_kafka_transient_then_success(self, mock_get_strategy, _):
        """Kafka path: transient error then success should retry once and succeed."""
        writer_strategy = Mock()
        writer_strategy.write = Mock(side_effect=[ConnectionError("broker down"), None])
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = KafkaDestinationConfig(bootstrap_servers=["localhost:9092"], topic="t")
        cfg.retry_count = 1

        success_called = threading.Event()

        fut = writer.write_async(
            [json.dumps({"k": 1})], cfg, on_success=lambda *_: success_called.set(), callback_executor=None
        )
        fut.result(timeout=2)

        assert writer_strategy.write.call_count == 2
        assert success_called.is_set()

    @patch("nv_ingest_api.data_handlers.data_writer.time.sleep", return_value=None)
    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_kafka_topic_error_is_permanent(self, mock_get_strategy, _):
        """Kafka 'topic not found' should be classified PermanentError with no retries."""
        writer_strategy = Mock()
        writer_strategy.write = Mock(side_effect=Exception("topic not found"))
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = KafkaDestinationConfig(bootstrap_servers=["localhost:9092"], topic="missing")
        cfg.retry_count = 3

        failure_holder = {}
        failure_called = threading.Event()

        def on_failure(data, config, exc):
            failure_holder["exc"] = exc
            failure_called.set()

        fut = writer.write_async([json.dumps({"z": 1})], cfg, on_failure=on_failure, callback_executor=None)
        fut.result(timeout=2)

        assert writer_strategy.write.call_count == 1
        assert failure_called.is_set()
        assert isinstance(failure_holder["exc"], PermanentError)

    @patch("nv_ingest_api.data_handlers.data_writer.time.sleep", return_value=None)
    @patch("nv_ingest_api.data_handlers.data_writer.get_writer_strategy")
    def test_kafka_leader_error_transient_then_success(self, mock_get_strategy, _):
        """Kafka 'leader not available' should be transient and allow retry to succeed."""
        writer_strategy = Mock()
        writer_strategy.write = Mock(side_effect=[Exception("leader not available"), None])
        mock_get_strategy.return_value = writer_strategy

        writer = IngestDataWriter.get_instance()
        cfg = KafkaDestinationConfig(bootstrap_servers=["localhost:9092"], topic="t")
        cfg.retry_count = 2

        success_called = threading.Event()

        fut = writer.write_async(
            [json.dumps({"m": 1})], cfg, on_success=lambda *_: success_called.set(), callback_executor=None
        )
        fut.result(timeout=2)

        assert writer_strategy.write.call_count == 2
        assert success_called.is_set()
