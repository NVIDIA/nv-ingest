import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from nemo_retriever.examples.batch_pipeline import _ensure_lancedb_table
from nemo_retriever.utils.input_files import resolve_input_patterns


def test_ensure_lancedb_table_creates_table_when_missing(monkeypatch, tmp_path) -> None:
    created: dict[str, object] = {}

    class _FakeDb:
        def open_table(self, _name: str) -> None:
            raise RuntimeError("missing")

        def create_table(self, table_name: str, data, schema, mode: str) -> None:
            created["table_name"] = table_name
            created["schema"] = schema
            created["mode"] = mode
            created["rows"] = data.num_rows

    class _FakeLanceDb:
        def connect(self, _uri: str) -> _FakeDb:
            return _FakeDb()

    monkeypatch.setattr("nemo_retriever.examples.batch_pipeline._lancedb", lambda: _FakeLanceDb())
    monkeypatch.setattr("nemo_retriever.examples.batch_pipeline.lancedb_schema", lambda: [])

    class _FakeTable:
        def __init__(self, values: dict[str, list[object]], schema: list[object]) -> None:
            self.num_rows = len(next(iter(values.values()), []))
            self.schema = schema

    fake_pyarrow = SimpleNamespace(table=lambda values, schema: _FakeTable(values, schema))
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)

    _ensure_lancedb_table(str(tmp_path / "lancedb"), "nv-ingest")
    assert created == {"table_name": "nv-ingest", "schema": [], "mode": "create", "rows": 0}


def test_ensure_lancedb_table_noops_when_table_exists(monkeypatch, tmp_path) -> None:
    class _FakeDb:
        def __init__(self) -> None:
            self.create_called = False

        def open_table(self, _name: str) -> None:
            return None

        def create_table(self, *args, **kwargs) -> None:  # pragma: no cover - should not run
            self.create_called = True

    fake_db = _FakeDb()

    class _FakeLanceDb:
        def connect(self, _uri: str) -> _FakeDb:
            return fake_db

    monkeypatch.setattr("nemo_retriever.examples.batch_pipeline._lancedb", lambda: _FakeLanceDb())

    _ensure_lancedb_table(str(tmp_path / "lancedb"), "nv-ingest")
    assert fake_db.create_called is False


def test_resolve_input_file_patterns_recurses_for_directory_inputs(tmp_path) -> None:
    dataset_dir = tmp_path / "earnings_consulting"
    dataset_dir.mkdir()

    pdf_patterns = resolve_input_patterns(dataset_dir, "pdf")
    txt_patterns = resolve_input_patterns(dataset_dir, "txt")
    doc_patterns = resolve_input_patterns(dataset_dir, "doc")

    assert pdf_patterns == [str(dataset_dir / "**" / "*.pdf")]
    assert txt_patterns == [str(dataset_dir / "**" / "*.txt")]
    assert doc_patterns == [str(dataset_dir / "**" / "*.docx"), str(dataset_dir / "**" / "*.pptx")]
