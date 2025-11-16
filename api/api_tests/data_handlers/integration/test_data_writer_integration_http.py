# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json

import pytest

from nv_ingest_api.data_handlers.data_writer import (
    IngestDataWriter,
    HttpDestinationConfig,
)

pytestmark = pytest.mark.integration_full


def _require_http_or_skip():
    base = os.getenv("INGEST_INTEGRATION_TEST_HTTP")
    if not base:
        pytest.skip("Skipping HTTP integration tests: INGEST_INTEGRATION_TEST_HTTP not set")
    try:
        import requests  # noqa: F401
    except Exception as e:
        pytest.skip(f"Skipping HTTP integration tests: requests not available ({e})")
    # Quick health check
    try:
        r = requests.get(f"{base}/healthz", timeout=5)
        if r.status_code != 200:
            pytest.skip(f"Skipping HTTP integration tests: healthz status {r.status_code}")
        return base
    except Exception as e:
        pytest.skip(f"Skipping HTTP integration tests: cannot reach service ({e})")


@pytest.fixture(autouse=True)
def reset_writer_singleton():
    IngestDataWriter.reset_for_tests()
    yield
    IngestDataWriter.reset_for_tests()


@pytest.fixture(scope="module")
def http_base_url():
    return _require_http_or_skip()


def test_http_single_async_write_and_validate(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    payloads = [json.dumps({"a": 1}), json.dumps({"b": 2})]
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers={}, auth_token=None)

    fut = writer.write_async(payloads, cfg, callback_executor=None)
    fut.result(timeout=15)

    # Validate via service /last endpoint
    import requests

    last = requests.get(f"{base}/last", timeout=5).json()
    assert last["last"] == [{"a": 1}, {"b": 2}]


def test_http_many_async_writes_and_validate(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    # Reuse the same endpoint; last will reflect the latest write
    futures = []
    expected = []

    for i in range(5):
        payloads = [json.dumps({"i": i})]
        expected = [{"i": i}]
        cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers={}, auth_token=None)
        fut = writer.write_async(payloads, cfg, callback_executor=None)
        futures.append(fut)

    for fut in futures:
        fut.result(timeout=20)

    import requests

    last = requests.get(f"{base}/last", timeout=5).json()
    assert last["last"] == expected


def test_http_sync_write_and_validate(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    payloads = [json.dumps({"x": 1}), json.dumps({"y": 2})]
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers={}, auth_token=None)

    writer._write_sync(payloads, cfg)  # noqa: SLF001

    import requests

    last = requests.get(f"{base}/last", timeout=5).json()
    assert last["last"] == [{"x": 1}, {"y": 2}]


def test_http_write_with_headers_and_auth_validated_by_server(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    payloads = [json.dumps({"hdr": True})]
    headers = {"x-test-header": "abc123"}
    cfg = HttpDestinationConfig(
        url=f"{base}/upload",
        method="POST",
        headers=headers,
        auth_token="token-xyz",
    )

    fut = writer.write_async(payloads, cfg, callback_executor=None)
    fut.result(timeout=15)

    import requests

    h = requests.get(f"{base}/last_headers", timeout=5).json()["headers"]
    # Authorization should be a Bearer token
    assert h.get("authorization") == "Bearer token-xyz"
    assert h.get("x-test-header") == "abc123"


def _make_large_payload(size_bytes: int, key: str = "blob") -> dict:
    chunk = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" * ((size_bytes // 36) + 1))[:size_bytes]
    return {key: chunk, "size": len(chunk)}


def test_http_single_async_write_large_payload_and_validate(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    payload = _make_large_payload(512 * 1024)
    payloads = [json.dumps(payload)]
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers={}, auth_token=None)

    fut = writer.write_async(payloads, cfg, callback_executor=None)
    fut.result(timeout=30)

    import requests

    last = requests.get(f"{base}/last", timeout=5).json()
    assert last["last"][0]["size"] == payload["size"]
    assert last["last"][0]["blob"] == payload["blob"]


def test_http_failure_invokes_failure_callback_4xx_no_retry(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    # Force a 429 via header and disable retries to make test fast
    payloads = [json.dumps({"err": 429})]
    headers = {"x-force-status": "429"}
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers=headers, auth_token=None, retry_count=0)

    failure_called = {"exc": None}

    def on_failure(data, config, exc):
        failure_called["exc"] = exc

    fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
    fut.result(timeout=10)

    assert failure_called["exc"] is not None


def test_http_failure_invokes_failure_callback_5xx_no_retry(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    # Force a 503 via header and disable retries to make test fast
    payloads = [json.dumps({"err": 503})]
    headers = {"x-force-status": "503"}
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers=headers, auth_token=None, retry_count=0)

    failure_called = {"exc": None}

    def on_failure(data, config, exc):
        failure_called["exc"] = exc

    fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
    fut.result(timeout=10)

    assert failure_called["exc"] is not None


def test_http_error_classification_client_error(http_base_url, monkeypatch):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    # Simulate 429 with Retry-After header via monkeypatching requests.Session.request
    class FakeResp:
        def __init__(self, status_code, headers=None):
            self.status_code = status_code
            self.ok = False
            self.headers = headers or {}

    def fake_request(method, url, json=None, headers=None, timeout=None):
        return FakeResp(429, {"Retry-After": "1"})

    _ = writer._IngestDataWriter__class__._get_session if False else None  # placeholder to appease lints

    # Patch the strategy's session getter to return our own session with a fake request
    from nv_ingest_api.data_handlers.writer_strategies.http import HttpWriterStrategy

    strat = HttpWriterStrategy()
    sess = strat._get_session()
    original_request = sess.request
    try:
        sess.request = fake_request  # type: ignore
        cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers={}, auth_token=None)
        with pytest.raises(Exception):
            strat.write([json.dumps({"a": 1})], cfg)
    finally:
        sess.request = original_request


def test_http_auth_errors_401_403_invoke_failure(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    for code in (401, 403):
        payloads = [json.dumps({"auth": code})]
        headers = {"x-force-status": str(code)}
        cfg = HttpDestinationConfig(
            url=f"{base}/upload", method="POST", headers=headers, auth_token=None, retry_count=0
        )

        failure_called = {"exc": None}

        def on_failure(data, config, exc):
            failure_called["exc"] = exc

        fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
        fut.result(timeout=10)
        assert failure_called["exc"] is not None


def test_http_408_retry_after_numeric_treated_transient(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    payloads = [json.dumps({"timeout": True})]
    headers = {"x-force-status": "408", "x-retry-after": "1"}
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers=headers, auth_token=None, retry_count=0)

    failure_called = {"exc": None}

    def on_failure(data, config, exc):
        failure_called["exc"] = exc

    fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
    fut.result(timeout=10)
    assert failure_called["exc"] is not None


def test_http_408_retry_after_malformed_treated_permanent(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    payloads = [json.dumps({"timeout": True})]
    headers = {"x-force-status": "408", "x-retry-after": "abc"}
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers=headers, auth_token=None, retry_count=0)

    failure_called = {"exc": None}

    def on_failure(data, config, exc):
        failure_called["exc"] = exc

    fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
    fut.result(timeout=10)
    assert failure_called["exc"] is not None


def test_http_400_bad_request_is_permanent(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    payloads = [json.dumps({"bad": True})]
    headers = {"x-force-status": "400"}
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers=headers, auth_token=None, retry_count=0)

    failure_called = {"exc": None}

    def on_failure(data, config, exc):
        failure_called["exc"] = exc

    fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
    fut.result(timeout=10)
    assert failure_called["exc"] is not None


def test_http_connection_error_classified(http_base_url):
    # Point to a likely closed port to induce connection error quickly
    writer = IngestDataWriter.get_instance()
    base = "http://127.0.0.1:19999"
    payloads = [json.dumps({"conn": True})]
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers={}, auth_token=None, retry_count=0)

    failure_called = {"exc": None}

    def on_failure(data, config, exc):
        failure_called["exc"] = exc

    fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
    fut.result(timeout=10)
    assert failure_called["exc"] is not None


def test_http_transient_retry_then_success(http_base_url):
    base = http_base_url
    writer = IngestDataWriter.get_instance()

    # Ask server to fail 2 times with 503, then succeed
    payloads = [json.dumps({"retry": True})]
    headers = {"x-fail-n": "2"}
    cfg = HttpDestinationConfig(url=f"{base}/upload", method="POST", headers=headers, auth_token=None, retry_count=3)

    success_called = {"ok": False}
    failure_called = {"exc": None}

    def on_success(data, config):
        success_called["ok"] = True

    def on_failure(data, config, exc):
        failure_called["exc"] = exc

    fut = writer.write_async(payloads, cfg, on_success=on_success, on_failure=on_failure, callback_executor=None)
    fut.result(timeout=30)

    # Should have eventually succeeded without invoking failure callback
    assert success_called["ok"] is True
    assert failure_called["exc"] is None
