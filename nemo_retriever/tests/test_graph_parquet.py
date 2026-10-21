# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_retriever.ingestor.parquet import ParquetDatasetWriter


def test_streaming_native_rows_keeps_late_failure_fields_without_concatenation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Summary mode must not concatenate the result corpus")

    monkeypatch.setattr(pd, "concat", forbidden)
    target = tmp_path / "rows"
    batches = (
        pd.DataFrame([{"page": 1, "meta": {"error": None, "raw": "ok"}, "bytes": b"one"}]),
        pd.DataFrame(
            [
                {
                    "page": 2,
                    "meta": {
                        "error": {"type": "Cutoff", "message": "length"},
                        "raw": "partial",
                    },
                    "bytes": b"two",
                    "later": 5,
                }
            ]
        ),
    )
    result = ParquetDatasetWriter(str(target), rows_per_file=1).write_batches(iter(batches))
    assert result.iloc[0]["written_rows"] == 2
    rows = pq.read_table(target).to_pylist()
    assert rows[0]["later"] is None
    assert rows[0]["meta"]["error"] is None
    assert rows[1]["meta"]["error"] == {"type": "Cutoff", "message": "length"}
    assert rows[1]["bytes"] == b"two"
    assert json.loads((target / "_SUCCESS").read_text())["rows"] == 2


def test_failed_stream_never_publishes_a_successful_dataset(tmp_path):
    closed = []

    def source():
        try:
            yield pd.DataFrame([{"text": "first"}])
            raise RuntimeError("source failed")
        finally:
            closed.append(True)

    target = tmp_path / "rows"
    with pytest.raises(RuntimeError, match="source failed"):
        ParquetDatasetWriter(str(target)).write_batches(source())
    assert closed == [True]
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_dataset_is_not_overwritten(tmp_path):
    target = tmp_path / "rows"
    writer = ParquetDatasetWriter(str(target))
    writer.write_batches([pd.DataFrame([{"text": "keep"}])])
    with pytest.raises(FileExistsError):
        writer.write_batches([pd.DataFrame([{"text": "replace"}])])
    assert pq.read_table(target)["text"].to_pylist() == ["keep"]


def test_empty_stream_is_readable(tmp_path):
    target = tmp_path / "empty"
    result = ParquetDatasetWriter(str(target)).write_batches([])
    assert result.iloc[0]["written_rows"] == 0
    assert pq.read_table(target).num_rows == 0


def test_inprocess_graph_stream_reuses_and_closes_operators(tmp_path):
    from nemo_retriever.graph import Graph, InprocessExecutor
    from nemo_retriever.operators.abstract_operator import AbstractOperator

    class Track(AbstractOperator):
        created = 0
        closed = 0

        def __init__(self):
            super().__init__()
            type(self).created += 1

        def preprocess(self, data, **kwargs):
            return data

        def process(self, data, **kwargs):
            assert len(data) == 1
            return data.assign(text=data["bytes"].map(bytes.decode))

        def postprocess(self, data, **kwargs):
            return data

        def close(self):
            type(self).closed += 1

    files = []
    for i in range(3):
        path = tmp_path / f"{i}.txt"
        path.write_text(str(i))
        files.append(str(path))
    graph = Graph() >> Track()
    baseline = Track.created
    result = ParquetDatasetWriter(str(tmp_path / "output")).write_batches(InprocessExecutor(graph).iter_batches(files))
    assert result.iloc[0]["written_rows"] == 3
    assert Track.created - baseline == 1
    assert Track.closed == 1


def test_graph_ingestor_save_to_disk_executes_real_text_path(tmp_path, monkeypatch):
    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

        def decode(self, tokens, **kwargs):
            return " ".join(tokens)

    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer",
        lambda *a, **k: Tokenizer(),
    )
    destination = tmp_path / "text"
    result = (
        GraphIngestor(run_mode="inprocess", show_progress=False)
        .texts(["one two three", "four five"])
        .extract(split_config={"text": {"max_tokens": 2}})
        .save_to_disk(str(destination), rows_per_file=1)
        .ingest(return_results=False)
    )
    assert result.iloc[0]["written_rows"] == 3
    assert pq.read_table(destination)["text"].to_pylist() == [
        "one two",
        "three",
        "four five",
    ]


@pytest.mark.parametrize("policy", ["raise", "collect"])
def test_summary_preserves_remote_failure_policy_after_saving_all_rows(tmp_path, policy):
    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor, GraphIngestionError

    destination = tmp_path / "failures"
    ingestor = (
        GraphIngestor(run_mode="inprocess", error_policy=policy)
        .extract(
            method="nemotron_parse",
            nemotron_parse_invoke_url="http://localhost:8000/v1/chat/completions",
        )
        .save_to_disk(str(destination), rows_per_file=1)
    )
    batches = [
        pd.DataFrame(
            [
                {
                    "path": "a.pdf",
                    "page_number": n,
                    "nemotron_parse_v1_2": {"error": {"stage": "parse", "type": "Cutoff", "message": "length"}},
                }
            ]
        )
        for n in (1, 2)
    ]
    if policy == "raise":
        with pytest.raises(GraphIngestionError, match="length"):
            ingestor._write_parquet(iter(batches), return_results=False)
    else:
        assert ingestor._write_parquet(iter(batches), return_results=False).iloc[0]["written_rows"] == 2
    assert pq.read_table(destination).num_rows == 2
    assert (destination / "_SUCCESS").exists()


def _page(number, text="hello", **extra):
    return {
        "path": "a.pdf",
        "page_number": number,
        "nemotron_parse_v1_2": {
            "raw_output": f"<x_0.1><y_0.1>{text}<x_0.9><y_0.3><class_Text>",
            "finish_reason": "stop",
        },
        **extra,
    }


def test_interleaved_sorts_pages_and_retains_failures_and_passthrough(tmp_path):
    from nemo_retriever.ingestor.interleaved import INTERLEAVED_SCHEMA

    target = tmp_path / "interleaved"
    failed = _page(
        2,
        nemotron_parse_v1_2={
            "raw_output": "partial",
            "finish_reason": "length",
            "error": {"type": "Cutoff", "message": "limit"},
        },
        language="ja",
    )
    frames = [
        pd.DataFrame([_page(3, "third", language="en"), failed]),
        pd.DataFrame([_page(1, "first", language="fr")]),
    ]
    result = ParquetDatasetWriter(
        str(target), output_schema="interleaved", passthrough_columns=["language"], rows_per_file=1
    ).write_batches(iter(frames))
    assert result.iloc[0]["input_rows"] == 3
    assert result.iloc[0]["written_rows"] == 4
    table = pq.read_table(target)
    assert pa.schema([table.schema.field(f.name) for f in INTERLEAVED_SCHEMA]) == INTERLEAVED_SCHEMA
    rows = table.to_pylist()
    assert [r["position"] for r in rows] == [-1, 0, 1, 2]
    assert [r["page_number"] for r in rows] == [None, 0, 1, 2]
    assert [r["text_content"] for r in rows[1:]] == ["first", None, "third"]
    assert json.loads(rows[2]["materialize_error"])["type"] == "Cutoff"
    assert rows[2]["language"] == "ja"
    assert json.loads(rows[0]["text_content"])["failed_pages"] == 1
    assert len({r["sample_id"] for r in rows}) == 1
    assert not (target / "_projection.sqlite").exists()


def test_interleaved_never_salvages_partial_page_or_invalid_box(tmp_path):
    from nemo_retriever.ingestor.interleaved import project_page

    partial = _page(1)
    partial["nemotron_parse_v1_2"]["raw_output"] += "<x_0.2><y_0.5>cut off"
    invalid = _page(2)
    invalid["nemotron_parse_v1_2"]["raw_output"] = "<x_0.9><y_0.1>wrong<x_0.1><y_0.3><class_Text>"
    for page in (partial, invalid, _page(3, nemotron_parse_v1_2={"raw_output": "", "finish_reason": "stop"})):
        rows = project_page(page)
        assert len(rows) == 1
        assert rows[0]["materialize_error"]
        assert rows[0]["text_content"] is None


def test_interleaved_embeds_readable_picture_and_keeps_native_return(tmp_path):
    import base64
    import io
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (500, 600), "green").save(buffer, format="PNG")
    page = _page(
        1,
        "caption",
        page_image={"image_b64": base64.b64encode(buffer.getvalue()).decode(), "orig_shape_hw": [600, 500]},
    )
    page["nemotron_parse_v1_2"]["raw_output"] = "<x_0.1><y_0.1>caption<x_0.8><y_0.8><class_Picture>"
    target = tmp_path / "pictures"
    returned = ParquetDatasetWriter(str(target), output_schema="interleaved").write_batches(
        [pd.DataFrame([page])], return_results=True
    )
    assert returned.iloc[0]["nemotron_parse_v1_2"] == page["nemotron_parse_v1_2"]
    picture = pq.read_table(target).to_pylist()[1]
    assert picture["modality"] == "image"
    with Image.open(io.BytesIO(picture["binary_content"])) as img:
        img.load()
        assert min(img.size) >= 10


def test_interleaved_duplicate_identity_aborts_publication(tmp_path):
    target = tmp_path / "duplicate"
    with pytest.raises(ValueError, match="Duplicate page identity"):
        ParquetDatasetWriter(str(target), output_schema="interleaved").write_batches(
            [pd.DataFrame([_page(1), _page(1)])]
        )
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_interleaved_empty_stream_has_reserved_schema(tmp_path):
    from nemo_retriever.ingestor.interleaved import PROJECTION_SCHEMA

    target = tmp_path / "empty-interleaved"
    ParquetDatasetWriter(str(target), output_schema="interleaved").write_batches([])
    assert pq.read_table(target).schema == PROJECTION_SCHEMA


@pytest.mark.parametrize("output_schema", ["native", "interleaved"])
def test_public_batch_export_consumes_arrow_stream_without_collecting(tmp_path, monkeypatch, output_schema):
    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

    class Cluster:
        def available_gpu_count(self):
            return 0

        def total_gpu_count(self):
            return 0

        def total_cpu_count(self):
            return 4

    class Dataset:
        def iter_batches(self, **kwargs):
            assert kwargs["batch_size"] == 1
            yield pa.Table.from_pylist([_page(2, "second")])
            yield pa.Table.from_pylist([_page(1, "first")])

    class Executor:
        def __init__(self, *args, **kwargs):
            pass

        def build_dataset(self, data):
            assert data == ["a.pdf"]
            return Dataset()

        def ingest(self, data):
            pytest.fail("Disk export must use the streaming dataset")

    target = tmp_path / output_schema
    ingestor = (
        GraphIngestor(run_mode="batch")
        .files(["a.pdf"])
        .extract(method="nemotron_parse", nemotron_parse_invoke_url="http://localhost:8000/v1/chat/completions")
        .save_to_disk(str(target), rows_per_file=1, output_schema=output_schema)
    )
    monkeypatch.setattr(ingestor, "_ensure_batch_runtime", lambda: (None, Cluster()))
    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.RayDataExecutor", Executor)
    summary = ingestor.ingest(return_results=False)
    assert summary.iloc[0]["input_rows"] == 2
    assert summary.iloc[0]["written_rows"] == (2 if output_schema == "native" else 3)
    assert pq.read_table(target).num_rows == summary.iloc[0]["written_rows"]


def test_stream_closes_all_operators_when_one_cleanup_fails():
    from nemo_retriever.graph import Graph, InprocessExecutor
    from nemo_retriever.operators.abstract_operator import AbstractOperator

    closed = []

    class First(AbstractOperator):
        def preprocess(self, data, **kwargs):
            return data

        def process(self, data, **kwargs):
            return data

        def postprocess(self, data, **kwargs):
            return data

        def close(self):
            closed.append("first")

    class Second(First):
        def close(self):
            closed.append("second")
            raise RuntimeError("cleanup failed")

    stream = InprocessExecutor(Graph() >> First() >> Second()).iter_batches(pd.DataFrame([{"text": "value"}]))
    assert next(stream).iloc[0]["text"] == "value"
    with pytest.raises(RuntimeError, match="cleanup failed"):
        stream.close()
    assert closed == ["second", "first"]
