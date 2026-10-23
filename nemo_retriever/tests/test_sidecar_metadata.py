# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.params import VdbUploadParams
from nemo_retriever.operators.vdb import IngestVdbOperator
from nemo_retriever.common.vdb.sidecar_metadata import (
    apply_sidecar_metadata_to_client_batches,
    filter_hits_by_content_metadata,
    normalize_sidecar_cell_value,
    parse_hit_content_metadata,
    split_sidecar_from_vdb_kwargs,
)


class _FakeVDB:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.run_calls: list[Any] = []

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, vectors: list, **kwargs: Any) -> list:
        return [[]]

    def run(self, records: Any) -> None:
        self.run_calls.append(records)


def test_normalize_sidecar_cell_value_list_and_dict_no_raise() -> None:
    assert normalize_sidecar_cell_value([1, 2]) == [1, 2]
    assert normalize_sidecar_cell_value({"k": 1}) == {"k": 1}
    np = pytest.importorskip("numpy")
    assert (normalize_sidecar_cell_value(np.array([1, 2])) == np.array([1, 2])).all()


def test_split_sidecar_from_vdb_kwargs_round_trip() -> None:
    raw = {"uri": "./db", "meta_dataframe": "/tmp/m.csv", "meta_source_field": "source", "meta_fields": ["a", "b"]}
    clean, spec = split_sidecar_from_vdb_kwargs(dict(raw))
    assert clean == {"uri": "./db"}
    assert spec is not None
    assert spec["meta_source_field"] == "source"
    assert spec["meta_fields"] == ["a", "b"]


def test_split_sidecar_meta_fields_comma_string() -> None:
    clean, spec = split_sidecar_from_vdb_kwargs(
        {"meta_dataframe": "x.csv", "meta_source_field": "s", "meta_fields": "a, b, c"}
    )
    assert clean == {}
    assert spec["meta_fields"] == ["a", "b", "c"]


def test_split_sidecar_partial_raises() -> None:
    with pytest.raises(ValueError, match="requires all"):
        split_sidecar_from_vdb_kwargs({"meta_dataframe": "x.csv", "meta_source_field": "s"})


def test_apply_sidecar_merges_into_content_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "meta.csv"
    pd.DataFrame(
        {
            "source": ["/data/doc_one.pdf"],
            "meta_a": ["alpha"],
            "meta_b": [7],
        }
    ).to_csv(csv_path, index=False)
    meta_df = pd.read_csv(csv_path)
    batches = [
        [
            {
                "document_type": "text",
                "embedding": [0.0],
                "metadata": {
                    "content": "hello",
                    "content_metadata": {"type": "text", "page_number": 0},
                    "source_metadata": {"source_id": "/data/doc_one.pdf", "source_name": "doc_one.pdf"},
                },
            }
        ]
    ]
    out = apply_sidecar_metadata_to_client_batches(
        batches,
        meta_df=meta_df,
        meta_source_field="source",
        meta_fields=["meta_a", "meta_b"],
        join_key="auto",
    )
    cm = out[0][0]["metadata"]["content_metadata"]
    assert cm["meta_a"] == "alpha"
    assert cm["meta_b"] == 7
    assert cm["type"] == "text"


def test_ingest_vdb_operator_marks_global_batch_for_ray() -> None:
    assert IngestVdbOperator.REQUIRES_GLOBAL_BATCH is True


def test_vdb_upload_params_triplet_validation() -> None:
    with pytest.raises(ValueError, match="all be set together"):
        VdbUploadParams(meta_dataframe="x.csv", meta_source_field="s", meta_fields=None)  # type: ignore[arg-type]

    p = VdbUploadParams(
        vdb_kwargs={"uri": "lancedb"},
        meta_dataframe="m.csv",
        meta_source_field="source",
        meta_fields=["meta_a"],
    )
    kw = p.to_ingest_operator_kwargs()
    assert kw["meta_source_field"] == "source"
    assert kw["meta_fields"] == ["meta_a"]


def test_ingest_operator_passes_merged_records_to_vdb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_retriever.operators import vdb as vdb_operator_module

    csv_path = tmp_path / "meta.csv"
    pd.DataFrame({"source": ["/tmp/doc-a.pdf"], "meta_a": ["zeta"]}).to_csv(csv_path, index=False)

    class Constructed(_FakeVDB):
        pass

    def fake_get_vdb_op_cls(vdb_op: str) -> type[_FakeVDB]:
        assert vdb_op == "fake"
        return Constructed

    monkeypatch.setattr(vdb_operator_module, "get_vdb_op_cls", fake_get_vdb_op_cls)

    operator = IngestVdbOperator(
        vdb_op="fake",
        vdb_kwargs={
            "meta_dataframe": str(csv_path),
            "meta_source_field": "source",
            "meta_fields": ["meta_a"],
        },
    )
    data = [
        {
            "text": "chunk",
            "embedding": [0.1] * 2048,
            "path": "/tmp/doc-a.pdf",
            "page_number": 1,
            "metadata": {"content_metadata": {"type": "text"}},
        }
    ]
    operator.process(data)
    vdb = operator._vdb
    assert vdb.run_calls
    rec = vdb.run_calls[0][0][0]
    assert rec["metadata"]["content_metadata"].get("meta_a") == "zeta"


def test_parse_hit_and_filter() -> None:
    hit = {"metadata": '{"meta_a": "alpha", "page_number": 0}', "text": "x"}
    assert parse_hit_content_metadata(hit)["meta_a"] == "alpha"
    filt = filter_hits_by_content_metadata([hit], lambda m: m.get("meta_a") == "alpha")
    assert len(filt) == 1
