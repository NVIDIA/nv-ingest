# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summary ingestion releases result batches without changing the stored stream."""

from __future__ import annotations

import gc
import weakref
from unittest.mock import Mock

import pandas as pd
import pytest

from nemo_retriever.common.params import IngestExecuteParams
from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.ingestor.graph_ingestor import GraphIngestionError, GraphIngestor
from nemo_retriever.operators.graph_ops.custom_operator import UDFOperator
from nemo_retriever.operators.vdb import IngestVdbOperator


class StreamingBackend:
    supports_stream_ingest = True

    def __init__(self):
        self.texts = []

    def stream_ingest(self, records):
        for record in records:
            self.texts.append(record["metadata"]["content"])


class GeneratedBatches:
    def __init__(self, *, check_release=False):
        self.check_release = check_release
        self.refs = []
        self.closed = False

    def iter_batches(self, **kwargs):
        try:
            for i in range(6):
                if self.check_release:
                    gc.collect()
                    # Iterators may still hold the current and preceding batch.
                    assert all(ref() is None for ref in self.refs[:-2])
                batch = pd.DataFrame(
                    [
                        {
                            "text": f"page-{i}",
                            "metadata": {"embedding": [1.0, 2.0]},
                            "result_only": "extraction payload",
                        },
                        {"text": "", "metadata": {}},
                    ]
                )
                self.refs.append(weakref.ref(batch))
                yield batch
        finally:
            self.closed = True


def executor_for(monkeypatch, backend, batches):
    executor = RayDataExecutor(Graph() >> IngestVdbOperator(vdb=backend))
    monkeypatch.setattr(executor, "_build_dataset", lambda *args, **kwargs: batches)
    return executor


@pytest.mark.parametrize("return_results", [True, False])
def test_streaming_summary_releases_batches_and_preserves_canonical_records(monkeypatch, return_results):
    batches = GeneratedBatches(check_release=not return_results)
    backend = StreamingBackend()
    executor = executor_for(monkeypatch, backend, batches)

    result = executor.ingest(object(), return_results=return_results)

    assert backend.texts == [f"page-{i}" for i in range(6)]
    assert batches.closed
    if return_results:
        assert len(result) == 12
        assert "result_only" in result
    else:
        assert result.to_dict("records") == [{"input_rows": 12, "submitted_records": 6}]
        gc.collect()
        assert all(ref() is None for ref in batches.refs)


def test_summary_does_not_return_success_on_finalization_failure(monkeypatch):
    class FailingBackend(StreamingBackend):
        def stream_ingest(self, records):
            super().stream_ingest(records)
            raise RuntimeError("index creation failed")

    batches = GeneratedBatches(check_release=True)
    executor = executor_for(monkeypatch, FailingBackend(), batches)
    with pytest.raises(RuntimeError, match="index creation failed"):
        executor.ingest(object(), return_results=False)
    assert batches.closed


@pytest.mark.parametrize("rejected_batch", [0, 1])
def test_summary_validates_before_submitting_each_batch(monkeypatch, rejected_batch):
    batches = GeneratedBatches()
    backend = StreamingBackend()
    executor = executor_for(monkeypatch, backend, batches)
    validated = []

    def validate_batch(frame):
        batch_index = len(validated)
        assert backend.texts == [f"page-{i}" for i in range(batch_index)]
        validated.append(frame.iloc[0]["text"])
        if batch_index == rejected_batch:
            raise ValueError("rejected stage error")

    with pytest.raises(ValueError, match="rejected stage error"):
        executor._ingest(object(), return_results=False, validate_batch=validate_batch)

    assert validated == [f"page-{i}" for i in range(rejected_batch + 1)]
    assert backend.texts == [f"page-{i}" for i in range(rejected_batch)]
    assert batches.closed


@pytest.mark.parametrize("shape", ["no_sink", "legacy_sink", "downstream"])
def test_summary_rejects_unsupported_graph_before_execution(monkeypatch, shape):
    backend = StreamingBackend()
    if shape == "legacy_sink":
        backend.supports_stream_ingest = False
    graph = Graph()
    if shape != "no_sink":
        graph = graph >> IngestVdbOperator(vdb=backend)
    if shape == "downstream":
        graph = graph >> UDFOperator(lambda batch: batch)
    executor = RayDataExecutor(graph)
    monkeypatch.setattr(executor, "_build_dataset", lambda *a, **k: pytest.fail("must reject before execution"))
    with pytest.raises(ValueError, match="terminal VDB"):
        executor.ingest(object(), return_results=False)


@pytest.mark.parametrize(
    "options",
    [
        {"return_results": False},
        {"params": {"return_results": False}},
        {"params": IngestExecuteParams(return_results=False)},
        {"params": IngestExecuteParams(return_results=True), "return_results": False},
    ],
)
def test_public_summary_option_checks_remote_errors_before_releasing_batch(monkeypatch, options):
    ingestor = (
        GraphIngestor(run_mode="batch")
        .files(["doc.pdf"])
        .extract(page_elements_invoke_url="http://invalid.test/page-elements")
        .vdb_upload()
    )

    class FakeCluster:
        def available_gpu_count(self):
            return 0

        def total_gpu_count(self):
            return 0

        def total_cpu_count(self):
            return 4

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def _ingest(self, data, *, return_results, validate_batch):
            assert return_results is False
            frame = pd.DataFrame(
                [
                    {
                        "path": "doc.pdf",
                        "text": "valid text",
                        "page_elements_v3": {
                            "error": {"stage": "page_elements_v3", "type": "RuntimeError", "message": "failed"}
                        },
                    }
                ]
            )
            validate_batch(frame)
            pytest.fail("stage error must be raised before the frame is released")

    monkeypatch.setattr(ingestor, "_ensure_batch_runtime", lambda: (None, FakeCluster()))
    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.RayDataExecutor", FakeExecutor)
    with pytest.raises(GraphIngestionError):
        ingestor.ingest(**options)


@pytest.mark.parametrize(
    "run_mode,upload,error_policy,return_failures",
    [
        ("inprocess", True, "raise", False),
        ("batch", False, "raise", False),
        ("batch", True, "collect", False),
        ("batch", True, "raise", True),
    ],
)
def test_public_summary_rejects_incompatible_options(run_mode, upload, error_policy, return_failures):
    ingestor = GraphIngestor(run_mode=run_mode, error_policy=error_policy)
    if upload:
        ingestor.vdb_upload()
    with pytest.raises(ValueError, match="return_results=False requires"):
        ingestor.ingest(return_results=False, return_failures=return_failures)


def test_empty_summary_validates_without_starting_ray_or_writing(monkeypatch):
    ingestor = GraphIngestor(run_mode="batch").texts([" "]).vdb_upload()
    backend = StreamingBackend()
    construct = Mock(return_value=backend)
    monkeypatch.setattr(ingestor, "_ensure_batch_runtime", lambda: pytest.fail("empty input must not start Ray"))
    monkeypatch.setattr("nemo_retriever.operators.vdb._construct_vdb", construct)
    monkeypatch.setattr(backend, "stream_ingest", lambda records: pytest.fail("empty input must not write to the VDB"))

    result = ingestor.ingest(return_results=False)

    construct.assert_called_once()
    assert result.to_dict("records") == [{"input_rows": 0, "submitted_records": 0}]
    assert ingestor.get_dataset() is result


@pytest.mark.parametrize("shape", ["legacy_sink", "downstream", "invalid_config"])
def test_empty_summary_rejects_invalid_pipeline(monkeypatch, shape):
    ingestor = GraphIngestor(run_mode="batch").texts([" "]).vdb_upload()
    backend = StreamingBackend()
    construct = Mock(return_value=backend)
    expected_error = "terminal VDB"
    if shape == "legacy_sink":
        backend.supports_stream_ingest = False
    elif shape == "downstream":
        ingestor.webhook(endpoint_url="http://invalid.test/webhook")
    else:
        construct.side_effect = ValueError("invalid backend configuration")
        expected_error = "invalid backend configuration"
    monkeypatch.setattr("nemo_retriever.operators.vdb._construct_vdb", construct)
    monkeypatch.setattr(ingestor, "_ensure_batch_runtime", lambda: pytest.fail("validation must happen before Ray"))

    with pytest.raises(ValueError, match=expected_error):
        ingestor.ingest(return_results=False)

    construct.assert_called_once()
