from pathlib import Path

import pytest

from nemo_retriever.harness.recall_adapters import prepare_recall_query_file


def test_prepare_recall_query_file_none_adapter_returns_input(tmp_path: Path) -> None:
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")

    out = prepare_recall_query_file(query_csv=query_csv, recall_adapter="none", output_dir=tmp_path / "out")
    assert out == query_csv


def test_prepare_recall_query_file_financebench_json(tmp_path: Path) -> None:
    query_json = tmp_path / "financebench_train.json"
    query_json.write_text(
        '[{"question":"What is revenue?","contexts":[{"filename":"AAPL_2023.pdf"}]}]',
        encoding="utf-8",
    )

    out = prepare_recall_query_file(
        query_csv=query_json, recall_adapter="financebench_json", output_dir=tmp_path / "out"
    )
    assert out.exists()
    contents = out.read_text(encoding="utf-8")
    assert "query,expected_pdf" in contents
    assert "What is revenue?,AAPL_2023" in contents


def test_prepare_recall_query_file_rejects_unknown_adapter(tmp_path: Path) -> None:
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown recall adapter"):
        prepare_recall_query_file(query_csv=query_csv, recall_adapter="bogus", output_dir=tmp_path / "out")
