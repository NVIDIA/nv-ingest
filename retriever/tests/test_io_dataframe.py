import json
from pathlib import Path

import pandas as pd
import pytest

from retriever.io.dataframe import read_dataframe, validate_primitives_dataframe, write_dataframe


def test_validate_primitives_dataframe_requires_metadata_column() -> None:
    df = pd.DataFrame([{"text": "hello"}])
    with pytest.raises(KeyError, match="metadata"):
        validate_primitives_dataframe(df)


def test_read_dataframe_json_mapping_unwraps_extracted_records(tmp_path: Path) -> None:
    payload = {"extracted_df_records": [{"metadata": {"a": 1}, "document_type": "pdf"}]}
    path = tmp_path / "records.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    df = read_dataframe(path)
    assert len(df) == 1
    assert df.iloc[0]["document_type"] == "pdf"
    assert isinstance(df.iloc[0]["metadata"], dict)


def test_read_dataframe_jsonl_unwraps_nested_stage_payload(tmp_path: Path) -> None:
    wrapped = {"primitives": [{"metadata": {"id": 123}, "document_type": "txt"}]}
    path = tmp_path / "wrapped.jsonl"
    path.write_text(json.dumps(wrapped) + "\n", encoding="utf-8")

    df = read_dataframe(path)
    assert len(df) == 1
    assert df.iloc[0]["document_type"] == "txt"
    assert df.iloc[0]["metadata"]["id"] == 123


def test_write_and_read_dataframe_json_round_trip(tmp_path: Path) -> None:
    source = pd.DataFrame([{"metadata": {"k": "v"}, "document_type": "html", "text": "chunk"}])
    path = tmp_path / "roundtrip.json"
    write_dataframe(source, path)

    loaded = read_dataframe(path)
    assert loaded.to_dict(orient="records") == source.to_dict(orient="records")


def test_write_and_read_dataframe_jsonl_round_trip(tmp_path: Path) -> None:
    source = pd.DataFrame(
        [
            {"metadata": {"n": 1}, "document_type": "pdf"},
            {"metadata": {"n": 2}, "document_type": "pdf"},
        ]
    )
    path = tmp_path / "roundtrip.jsonl"
    write_dataframe(source, path)

    loaded = read_dataframe(path)
    assert loaded.to_dict(orient="records") == source.to_dict(orient="records")


def test_read_and_write_dataframe_reject_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported DataFrame format"):
        read_dataframe(path)

    df = pd.DataFrame([{"metadata": {}}])
    with pytest.raises(ValueError, match="Unsupported DataFrame format"):
        write_dataframe(df, path)
