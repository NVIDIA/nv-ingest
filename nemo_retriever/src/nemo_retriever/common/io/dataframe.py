# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def validate_primitives_dataframe(df: pd.DataFrame) -> None:
    if "metadata" not in df.columns:
        raise KeyError("Primitives DataFrame must include a 'metadata' column.")


def _arrow_table_to_pandas_via_pylist(table: object) -> pd.DataFrame:
    """Materialize an Arrow table without ``to_pandas`` (avoids nested/chunked bugs)."""
    if table.num_rows == 0:
        return pd.DataFrame()
    return pd.DataFrame({name: table.column(name).to_pylist() for name in table.column_names})


def read_extraction_parquet(path: Path | str) -> pd.DataFrame:
    """Load pipeline extraction Parquet (nested list/struct columns).

    PyArrow's dataset-based ``read_table`` / default ``pd.read_parquet`` can raise
    ``ArrowNotImplementedError: Nested data conversions not implemented for chunked
    array outputs``. Read with :class:`pyarrow.parquet.ParquetFile`, prefer
    ``to_pandas``, then fall back to column-wise ``to_pylist()`` (always works for
    nested types), then fastparquet / dataset reader.
    """
    path = Path(path)
    import pyarrow.parquet as pq

    try:
        table = pq.ParquetFile(path).read()
        try:
            table = table.combine_chunks()
        except Exception:
            pass
        try:
            return table.to_pandas(split_blocks=False)
        except Exception:
            return _arrow_table_to_pandas_via_pylist(table)
    except Exception:
        pass
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception:
        pass
    try:
        table = pq.ParquetFile(path).read()
        return _arrow_table_to_pandas_via_pylist(table)
    except Exception:
        pass
    return pd.read_parquet(path)


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return read_extraction_parquet(path)
    if suffix in {".jsonl", ".json"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            obj = json.loads(text)
            if isinstance(obj, dict):
                records = _unwrap_records_from_mapping(obj)
            else:
                records = obj if isinstance(obj, list) else [obj]

        # Some stage payloads are wrapped once more in jsonl.
        if (
            isinstance(records, list)
            and len(records) == 1
            and isinstance(records[0], dict)
            and not ("metadata" in records[0] and "document_type" in records[0])
        ):
            records = _unwrap_records_from_mapping(records[0])

        return pd.DataFrame(records)

    raise ValueError(f"Unsupported DataFrame format: {path} (expected .parquet, .jsonl, or .json)")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as file_handle:
            for record in df.to_dict(orient="records"):
                file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    if suffix == ".json":
        payload = df.to_dict(orient="records")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    raise ValueError(f"Unsupported DataFrame format: {path} (expected .parquet, .jsonl, or .json)")


def _unwrap_records_from_mapping(obj: dict) -> list:
    for key in ("records", "df_records", "extracted_df_records", "primitives"):
        value = obj.get(key)
        if isinstance(value, list) and (not value or isinstance(value[0], dict)):
            return value
    return [obj]
