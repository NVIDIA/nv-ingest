import sys
from types import ModuleType
from types import SimpleNamespace

import pandas as pd
import pytest

from nemo_retriever.recall.core import _normalize_query_df, _search_lancedb, hit_key_and_distance, is_hit_at_k


@pytest.mark.parametrize(
    "match_mode,df,expected",
    [
        ("pdf_only", pd.DataFrame({"query": ["q1"], "expected_pdf": ["Doc_A.pdf"]}), ["Doc_A"]),
        ("pdf_page", pd.DataFrame({"query": ["q1"], "pdf": ["Doc_A.pdf"], "page": [2]}), ["Doc_A_2"]),
    ],
)
def test_normalize_query_df_modes(match_mode: str, df: pd.DataFrame, expected: list[str]) -> None:
    out = _normalize_query_df(df, match_mode=match_mode)
    assert out["golden_answer"].tolist() == expected


@pytest.mark.parametrize(
    "match_mode,golden,retrieved,k,expected",
    [
        ("pdf_only", "Doc_A", ["Doc_A_7", "Doc_B_1"], 1, True),
        ("pdf_only", "Doc_B", ["Doc_A_7", "Doc_B_1"], 1, False),
        ("pdf_page", "Doc_A_2", ["Doc_A_2", "Doc_A_9"], 1, True),
        ("pdf_page", "Doc_A_2", ["Doc_A_9", "Doc_A_-1"], 1, False),
    ],
)
def test_is_hit_at_k_modes(match_mode: str, golden: str, retrieved: list[str], k: int, expected: bool) -> None:
    assert is_hit_at_k(golden, retrieved, k, match_mode=match_mode) is expected


def test_hit_key_and_distance_supports_score() -> None:
    hit = {
        "metadata": '{"page_number": 4}',
        "source": "/tmp/Doc_A.pdf",
        "page_number": 4,
        "_score": 0.87,
    }
    key, distance = hit_key_and_distance(hit)
    assert key == "Doc_A_4"
    assert distance == 0.87


def test_hit_key_and_distance_supports_legacy_source_json() -> None:
    hit = {
        "metadata": '{"page_number": 7}',
        "source": '{"source_id": "/tmp/Legacy.pdf"}',
        "_distance": 0.12,
    }
    key, distance = hit_key_and_distance(hit)
    assert key == "Legacy_7"
    assert distance == 0.12


class _FakeQuery:
    def __init__(self, *, hybrid: bool, fail_score_select: bool = False) -> None:
        self.hybrid = hybrid
        self.fail_score_select = fail_score_select
        self.selected_fields: list[str] | None = None

    def vector(self, _vector):
        return self

    def text(self, _text: str):
        return self

    def nprobes(self, _n: int):
        return self

    def refine_factor(self, _factor: int):
        return self

    def select(self, fields: list[str]):
        self.selected_fields = list(fields)
        if self.fail_score_select and "_score" in fields:
            raise RuntimeError("score select unsupported")
        return self

    def limit(self, _k: int):
        return self

    def rerank(self, _reranker):
        return self

    def to_list(self):
        if self.hybrid:
            return [{"text": "hello", "metadata": "{}", "source": "{}", "_score": 0.9}]
        return [{"text": "hello", "metadata": "{}", "source": "{}", "_distance": 0.1}]


class _FakeTable:
    def __init__(self, *, fail_score_select: bool = False) -> None:
        self.fail_score_select = fail_score_select
        self.last_query: _FakeQuery | None = None

    def list_indices(self):
        return []

    def search(self, *_args, **kwargs):
        self.last_query = _FakeQuery(
            hybrid=kwargs.get("query_type") == "hybrid",
            fail_score_select=self.fail_score_select,
        )
        return self.last_query


def _install_fake_lancedb(monkeypatch: pytest.MonkeyPatch, table: _FakeTable) -> None:
    fake_db = SimpleNamespace(open_table=lambda _name: table)
    lancedb_mod = ModuleType("lancedb")
    lancedb_mod.connect = lambda _uri: fake_db  # type: ignore[attr-defined]
    rerankers_mod = ModuleType("lancedb.rerankers")
    rerankers_mod.RRFReranker = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lancedb", lancedb_mod)
    monkeypatch.setitem(sys.modules, "lancedb.rerankers", rerankers_mod)


def test_search_lancedb_hybrid_selects_score_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    table = _FakeTable()
    _install_fake_lancedb(monkeypatch, table)

    hits = _search_lancedb(
        lancedb_uri="/tmp/lancedb",
        table_name="nv-ingest",
        query_vectors=[[1.0, 2.0]],
        top_k=5,
        query_texts=["hello world"],
        hybrid=True,
    )

    assert hits[0][0]["_score"] == 0.9
    assert table.last_query is not None
    assert table.last_query.selected_fields == ["text", "metadata", "source", "page_number", "_score"]


def test_search_lancedb_hybrid_falls_back_without_score_select(monkeypatch: pytest.MonkeyPatch) -> None:
    table = _FakeTable(fail_score_select=True)
    _install_fake_lancedb(monkeypatch, table)

    hits = _search_lancedb(
        lancedb_uri="/tmp/lancedb",
        table_name="nv-ingest",
        query_vectors=[[1.0, 2.0]],
        top_k=5,
        query_texts=["hello world"],
        hybrid=True,
    )

    assert hits[0][0]["_score"] == 0.9
    assert table.last_query is not None
    assert table.last_query.selected_fields == ["text", "metadata", "source", "page_number"]
