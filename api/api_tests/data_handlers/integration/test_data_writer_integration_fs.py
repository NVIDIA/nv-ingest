# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import tempfile

import pytest
import threading
import uuid

from nv_ingest_api.data_handlers.data_writer import (
    IngestDataWriter,
    FilesystemDestinationConfig,
)

pytestmark = pytest.mark.integration_full


@pytest.fixture(autouse=True)
def reset_writer_singleton():
    """Ensure a fresh IngestDataWriter for each test."""
    IngestDataWriter.reset_for_tests()
    yield
    IngestDataWriter.reset_for_tests()


def _cleanup_file(path: str) -> None:
    try:
        # Guard: must be a single, absolute path under the system temp directory
        tmp_root = os.path.realpath(tempfile.gettempdir())
        target = os.path.realpath(path)

        # Reject if not under tmp
        if not target.startswith(tmp_root + os.sep):
            print(f"Refusing to remove non-tmp path: {target}")
            return

        # Reject glob-like inputs (safety)
        if any(ch in path for ch in ("*", "?", "[", "]")):
            print(f"Refusing to remove glob-like path: {path}")
            return

        # Only remove files, never directories
        if os.path.isdir(target):
            print(f"Refusing to remove directory: {target}")
            return

        if os.path.exists(target):
            os.remove(target)
    except Exception:
        print(f"Error while cleaning up file {path} -- it may need to be removed manually.")
        # Best-effort cleanup; do not fail tests on cleanup errors
        pass


def test_filesystem_single_write_creates_file_with_expected_contents():
    payloads = [json.dumps({"a": 1}), json.dumps({"b": 2})]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "out.json")
        cfg = FilesystemDestinationConfig(path=out_path)

        writer = IngestDataWriter.get_instance()
        fut = writer.write_async(payloads, cfg, callback_executor=None)
        # Wait for completion (write + callbacks)
        fut.result(timeout=5)

        try:
            assert os.path.exists(out_path)
            with open(out_path, "r") as f:
                data = json.load(f)
            assert data == [{"a": 1}, {"b": 2}]
        finally:
            _cleanup_file(out_path)


def test_async_success_callback_exception_is_caught():
    """Ensure that an exception in the success callback is caught and does not fail the write."""
    payloads = [json.dumps({"ok": 2})]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "cb_out_raise.json")
        cfg = FilesystemDestinationConfig(path=out_path)

        writer = IngestDataWriter.get_instance()

        success_called = threading.Event()

        def on_success(data, config):
            success_called.set()
            raise RuntimeError("callback boom")

        fut = writer.write_async(payloads, cfg, on_success=on_success, callback_executor=None)
        # Even if callback raises, the result future should complete
        fut.result(timeout=5)

        try:
            assert success_called.is_set()
            assert os.path.exists(out_path)
            with open(out_path, "r") as f:
                data = json.load(f)
            assert data == [{"ok": 2}]
        finally:
            _cleanup_file(out_path)


def test_async_failure_invokes_failure_callback_on_directory_path():
    """Attempt to write to a directory path and validate failure callback is invoked."""
    payloads = [json.dumps({"fail": True})]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory where a file is expected, causing write to fail
        out_path = os.path.join(tmpdir, "dir_instead_of_file")
        os.makedirs(out_path)
        cfg = FilesystemDestinationConfig(path=out_path)

        writer = IngestDataWriter.get_instance()

        failure_called = threading.Event()
        captured = {}

        def on_failure(data, config, exc):
            captured["exc"] = exc
            failure_called.set()

        fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
        # The result future completes regardless; failure is delivered via callback
        fut.result(timeout=5)

        assert failure_called.is_set()
        assert "exc" in captured
        # Ensure the path remains a directory and no file was created
        assert os.path.isdir(out_path)


def test_failure_callback_receives_exception_type():
    payloads = [json.dumps({"fail": True})]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "dir_instead_of_file_type")
        os.makedirs(out_path)
        cfg = FilesystemDestinationConfig(path=out_path)

        writer = IngestDataWriter.get_instance()

        captured = {"exc": None}
        evt = threading.Event()

        def on_failure(data, config, exc):
            captured["exc"] = exc
            evt.set()

        fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
        fut.result(timeout=5)

        assert evt.is_set()
        # The exception type should be something like IsADirectoryError or OSError
        assert isinstance(captured["exc"], Exception)
        assert os.path.isdir(out_path)


# --------------------
# MinIO/S3 integration
# --------------------


def _parse_minio_env(url: str):
    """Parse INGEST_INTEGRATION_TEST_MINIO of form http://host:9000/bucket into (endpoint_url, bucket)."""
    # Accept http(s)://host[:port]/bucket[/]
    if "//" not in url:
        raise ValueError("Expected URL like http://host:9000/bucket")
    scheme_host, bucket_path = url.split("//", 1)[1].split("/", 1)
    endpoint_url = url.rsplit("/", 1)[0]  # everything up to /bucket
    bucket = bucket_path.strip("/")
    return endpoint_url, bucket


def _require_minio_or_skip():
    env = os.getenv("INGEST_INTEGRATION_TEST_MINIO")
    if not env:
        pytest.skip("Skipping MinIO S3 tests: INGEST_INTEGRATION_TEST_MINIO not set")
    try:
        import fsspec  # noqa: F401
        import s3fs  # noqa: F401
    except Exception as e:
        pytest.skip(f"Skipping MinIO S3 tests: s3fs/fsspec not available ({e})")

    try:
        endpoint_url, bucket = _parse_minio_env(env)

        fs = fsspec.filesystem("s3")
        # ls may raise if bucket does not exist; we will attempt and allow empty results
        try:
            _ = fs.ls(f"s3://{bucket}")
        except FileNotFoundError:
            # Bucket may be empty but exists; s3fs may raise; proceed anyway
            pass
        return bucket
    except Exception as e:
        pytest.skip(f"Skipping MinIO S3 tests: cannot access bucket ({e})")


def _s3_cleanup(path: str):
    try:
        import fsspec

        fs = fsspec.filesystem("s3")
        if fs.exists(path):
            fs.rm(path)
    except Exception:
        # best-effort cleanup
        pass


def test_minio_s3_single_async_write_and_readback():
    bucket = _require_minio_or_skip()

    key = f"ingest-tests/{uuid.uuid4().hex}.json"
    s3_path = f"s3://{bucket}/{key}"
    payloads = [json.dumps({"s3": True}), json.dumps({"n": 1})]

    writer = IngestDataWriter.get_instance()
    cfg = FilesystemDestinationConfig(path=s3_path)

    fut = writer.write_async(payloads, cfg, callback_executor=None)
    fut.result(timeout=20)

    try:
        import fsspec

        with fsspec.open(s3_path, "r") as f:
            data = json.load(f)
        assert data == [{"s3": True}, {"n": 1}]
    finally:
        _s3_cleanup(s3_path)


def test_minio_s3_many_async_writes_and_readback():
    bucket = _require_minio_or_skip()

    writer = IngestDataWriter.get_instance()
    futures = []
    keys = []

    for i in range(5):
        key = f"ingest-tests/many/{uuid.uuid4().hex}.json"
        s3_path = f"s3://{bucket}/{key}"
        keys.append(s3_path)
        cfg = FilesystemDestinationConfig(path=s3_path)
        fut = writer.write_async([json.dumps({"idx": i})], cfg, callback_executor=None)
        futures.append(fut)

    for fut in futures:
        fut.result(timeout=30)

    try:
        import fsspec

        for i, p in enumerate(keys):
            with fsspec.open(p, "r") as f:
                data = json.load(f)
            assert data == [{"idx": i}]
    finally:
        for p in keys:
            _s3_cleanup(p)


def test_async_failure_callback_exception_is_caught():
    """If the failure callback raises, it should be caught and not crash the writer."""
    payloads = [json.dumps({"fail": True})]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "dir_instead_of_file_2")
        os.makedirs(out_path)
        cfg = FilesystemDestinationConfig(path=out_path)

        writer = IngestDataWriter.get_instance()

        failure_called = threading.Event()

        def on_failure(data, config, exc):
            failure_called.set()
            raise RuntimeError("failure callback boom")

        fut = writer.write_async(payloads, cfg, on_failure=on_failure, callback_executor=None)
        # Should not raise even though failure callback raises internally
        fut.result(timeout=5)

        assert failure_called.is_set()
        assert os.path.isdir(out_path)


def test_filesystem_many_async_writes_all_complete():
    payload = [json.dumps({"x": 42})]

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = IngestDataWriter.get_instance()
        futures = []
        paths = []

        for i in range(10):
            out_path = os.path.join(tmpdir, f"out_{i}.json")
            paths.append(out_path)
            cfg = FilesystemDestinationConfig(path=out_path)
            fut = writer.write_async(payload, cfg, callback_executor=None)
            futures.append(fut)

        # Wait for all futures
        for fut in futures:
            fut.result(timeout=10)

        # Validate all files exist and contain the expected payload array
        try:
            for p in paths:
                assert os.path.exists(p)
                with open(p, "r") as f:
                    data = json.load(f)
                assert data == [{"x": 42}]
        finally:
            for p in paths:
                _cleanup_file(p)


def test_filesystem_sync_write_direct_call():
    """Exercise the synchronous write path using the real filesystem writer strategy."""
    payloads = [json.dumps({"sync": True}), json.dumps({"n": 1})]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "sync_out.json")
        cfg = FilesystemDestinationConfig(path=out_path)

        writer = IngestDataWriter.get_instance()
        # Call the synchronous path directly to ensure it works end-to-end
        writer._write_sync(payloads, cfg)  # noqa: SLF001 (testing internal for integration coverage)

        try:
            assert os.path.exists(out_path)
            with open(out_path, "r") as f:
                data = json.load(f)
            assert data == [{"sync": True}, {"n": 1}]
        finally:
            _cleanup_file(out_path)


def test_async_write_invokes_success_callback():
    payloads = [json.dumps({"ok": 1})]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "cb_out.json")
        cfg = FilesystemDestinationConfig(path=out_path)

        writer = IngestDataWriter.get_instance()

        success_called = threading.Event()

        def on_success(data, config):
            success_called.set()

        fut = writer.write_async(payloads, cfg, on_success=on_success, callback_executor=None)
        fut.result(timeout=5)

        try:
            assert success_called.is_set()
            assert os.path.exists(out_path)
            with open(out_path, "r") as f:
                data = json.load(f)
            assert data == [{"ok": 1}]
        finally:
            _cleanup_file(out_path)
