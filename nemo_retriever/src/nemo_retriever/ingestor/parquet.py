# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded Parquet publication for graph ingestion results."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _arrow_value(value: Any) -> Any:
    """Normalize object cells; Parquet cannot represent an empty struct."""
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _arrow_value(v) for k, v in value.items()} or None
    if isinstance(value, (list, tuple)):
        return [_arrow_value(v) for v in value]
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if type(value).__module__.startswith("numpy"):
        return _arrow_value(value.tolist())
    return value


class ParquetDatasetWriter:
    """Publish a complete dataset atomically, retaining errors in native rows.

    Only bounded row blocks are converted to Arrow. If a later block introduces
    a new nullable field, earlier parts are aligned one part at a time before
    publication. No extracted corpus is concatenated unless the caller explicitly
    requests returned records. Empty dictionaries become null because Parquet
    does not support structs without fields.
    """

    def __init__(
        self,
        output_directory: str,
        *,
        compression: str | None = "snappy",
        rows_per_file: int = 1024,
        output_schema: str = "native",
        passthrough_columns: Sequence[str] = (),
    ):
        if not str(output_directory).strip():
            raise ValueError("save_to_disk requires an output_directory")
        if isinstance(rows_per_file, bool) or not isinstance(rows_per_file, int) or rows_per_file < 1:
            raise ValueError("rows_per_file must be a positive integer")
        if compression is not None and not pa.Codec.is_available(compression):
            raise ValueError(f"Unsupported Parquet compression: {compression!r}")
        if output_schema not in {"native", "interleaved"}:
            raise ValueError("output_schema must be 'native' or 'interleaved'")
        if isinstance(passthrough_columns, str) or any(not isinstance(c, str) or not c for c in passthrough_columns):
            raise ValueError("passthrough_columns must be a sequence of nonempty column names")
        if output_schema == "native" and passthrough_columns:
            raise ValueError("Native output already retains every column; passthrough_columns requires interleaved")
        if output_schema == "interleaved":
            from .interleaved import PROJECTION_SCHEMA

            collisions = set(passthrough_columns) & set(PROJECTION_SCHEMA.names)
            if collisions:
                raise ValueError(f"Passthrough columns conflict with interleaved fields: {sorted(collisions)}")
        self.output_schema = output_schema
        self.passthrough_columns = tuple(passthrough_columns)
        self.directory = Path(output_directory).expanduser().absolute()
        self.compression = compression
        self.rows_per_file = rows_per_file

    def write_batches(self, batches: Iterable[Any], *, return_results: bool = False) -> pd.DataFrame:
        """Consume the stream fully before publishing; never replace existing output."""
        if self.directory.exists():
            raise FileExistsError(f"Parquet destination already exists: {self.directory}")
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{self.directory.name}-", dir=self.directory.parent))
        frames = []
        schema = None
        parts = []
        row_count = 0
        input_count = 0
        spool = None
        iterator = iter(batches)

        def write_rows(rows):
            nonlocal schema, row_count
            if self.output_schema == "interleaved":
                from .interleaved import projection_table

                table = projection_table(rows)
            else:
                table = pa.Table.from_pylist(rows)
            schema = (
                table.schema
                if schema is None
                else pa.unify_schemas([schema, table.schema], promote_options="permissive")
            )
            name = temporary / f"part-{len(parts):06d}.parquet"
            pq.write_table(table, name, compression=self.compression)
            parts.append(name)
            row_count += len(rows)

        try:
            if self.output_schema == "interleaved":
                from .interleaved import InterleavedSpool

                spool = InterleavedSpool(temporary, self.passthrough_columns)
            for batch in iterator:
                if isinstance(batch, (pa.Table, pa.RecordBatch)):
                    frame = batch.to_pandas()
                elif isinstance(batch, pd.DataFrame):
                    frame = batch
                else:
                    raise TypeError("Parquet batches must be pandas DataFrames or Arrow tables")
                input_count += len(frame)
                if return_results:
                    frames.append(frame)
                for start in range(0, len(frame), self.rows_per_file):
                    rows = [
                        _arrow_value(row) for row in frame.iloc[start : start + self.rows_per_file].to_dict("records")
                    ]
                    if spool is None:
                        write_rows(rows)
                    else:
                        spool.add(rows)
            if spool is not None:
                block = []
                for row in spool.iter_rows():
                    block.append(row)
                    if len(block) == self.rows_per_file:
                        write_rows(block)
                        block = []
                if block:
                    write_rows(block)
                spool.close()
                spool = None
            if schema is None:
                if self.output_schema == "interleaved":
                    from .interleaved import PROJECTION_SCHEMA

                    schema = PROJECTION_SCHEMA
                else:
                    schema = pa.schema([])
                name = temporary / "part-000000.parquet"
                pq.write_table(
                    pa.Table.from_pylist([], schema=schema),
                    name,
                    compression=self.compression,
                )
                parts.append(name)
            if schema is not None:
                # Align nullable fields introduced by late failures or another document type.
                # This preserves a consistent schema for ordinary read_table/dataset readers.
                for name in parts:
                    table = pq.read_table(name)
                    if table.schema != schema:
                        aligned = pa.Table.from_pylist(table.to_pylist(), schema=schema)
                        pending = name.with_suffix(".pending")
                        pq.write_table(aligned, pending, compression=self.compression)
                        os.replace(pending, name)
                pq.write_metadata(schema, temporary / "_common_metadata")
            (temporary / "_SUCCESS").write_text(
                json.dumps(
                    {
                        "rows": row_count,
                        "input_rows": input_count,
                        "files": len(parts),
                        "format": "parquet",
                        "output_schema": self.output_schema,
                    }
                )
                + "\n"
            )
            # Refuse races as well as pre-existing outputs; only this attempt's temp dir is removed.
            if self.directory.exists():
                raise FileExistsError(f"Parquet destination appeared during ingestion: {self.directory}")
            os.rename(temporary, self.directory)
        finally:
            close = getattr(iterator, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                try:
                    if spool is not None:
                        spool.close()
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
        if return_results:
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "input_rows": input_count,
                    "written_rows": row_count,
                    "parquet_files": len(parts),
                }
            ]
        )
