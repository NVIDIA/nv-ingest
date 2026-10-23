# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch LanceDB writes and index creation (canonical under ``vdb``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import lancedb
from nemo_retriever.common.vdb.lancedb_capabilities import wait_for_column_index
from nemo_retriever.common.vdb.lancedb_schema import build_lancedb_row, infer_vector_dim, lancedb_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanceDBConfig:
    """
    Minimal config for writing embeddings into LanceDB.

    Used by parquet-to-LanceDB conversion.
    """

    uri: str = "lancedb"
    table_name: str = "nemo-retriever"
    overwrite: bool = True

    # Optional index creation (recommended for recall/search runs).
    create_index: bool = True
    index_type: str = "IVF_HNSW_SQ"
    metric: str = "l2"
    num_partitions: int = 16
    num_sub_vectors: int = 256

    hybrid: bool = False
    fts_language: str = "English"


class _MappingRow:
    def __init__(self, values: Dict[str, Any]) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _build_lancedb_rows_from_df(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform embeddings-enriched primitive rows into the shared LanceDB row schema."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_out = build_lancedb_row(
            _MappingRow(row), embedding=row.get("embedding"), provenance_page=row.get("_page_number")
        )
        if row_out is not None:
            out.append(row_out)
    return out


def create_lancedb_index(table: Any, *, cfg: LanceDBConfig, text_column: str = "text") -> None:
    """Create vector (IVF_HNSW_SQ) and optionally FTS indices on a LanceDB table."""
    covered_rows = int(table.count_rows())
    try:
        table.create_index(
            index_type=cfg.index_type,
            metric=cfg.metric,
            num_partitions=int(cfg.num_partitions),
            num_sub_vectors=int(cfg.num_sub_vectors),
            vector_column_name="vector",
        )
    except TypeError:
        table.create_index(vector_column_name="vector")

    if cfg.hybrid:
        try:
            table.create_fts_index(text_column, replace=True, language=cfg.fts_language)
        except Exception:
            logger.warning(
                "FTS index creation failed on column %r; continuing with vector-only search.",
                text_column,
                exc_info=True,
            )

    wait_for_column_index(table, "vector", covered_rows=covered_rows)
    if cfg.hybrid:
        wait_for_column_index(table, text_column, covered_rows=covered_rows)


def _write_rows_to_lancedb(rows: Sequence[Dict[str, Any]], *, cfg: LanceDBConfig) -> None:
    row_list = list(rows)
    if not row_list:
        logger.warning("No embeddings rows provided; nothing to write to LanceDB.")
        return

    dim = infer_vector_dim(row_list)
    if dim <= 0:
        raise ValueError("Failed to infer embedding dimension from rows.")

    db = lancedb.connect(uri=cfg.uri)

    schema = lancedb_schema(vector_dim=dim)

    mode = "overwrite" if cfg.overwrite else "append"
    table = db.create_table(cfg.table_name, data=row_list, schema=schema, mode=mode)

    if cfg.create_index:
        create_lancedb_index(table, cfg=cfg)


def handle_lancedb(
    rows: List[Dict[str, Any]],
    uri: str,
    table_name: str,
    hybrid: bool = False,
    mode: str = "overwrite",
) -> Dict[str, Any]:
    """Write flattened extraction rows into LanceDB and build indices.

    ``mode`` accepts ``"overwrite"`` or ``"append"`` and is mapped to the
    corresponding :class:`LanceDBConfig.overwrite` value.
    """
    mode_normalized = str(mode or "overwrite").strip().lower()
    if mode_normalized not in {"overwrite", "append"}:
        raise ValueError(f"mode must be 'overwrite' or 'append'; got {mode!r}")

    lancedb_config = LanceDBConfig(
        uri=uri,
        table_name=table_name,
        overwrite=mode_normalized == "overwrite",
        hybrid=hybrid,
    )
    db = lancedb.connect(uri=lancedb_config.uri)
    cleaned_rows = _build_lancedb_rows_from_df(rows)
    if not cleaned_rows:
        logger.warning("No embedding rows to write; skipping LanceDB index creation.")
        return {}
    _write_rows_to_lancedb(cleaned_rows, cfg=lancedb_config)
    table = db.open_table(lancedb_config.table_name)
    create_lancedb_index(table, cfg=lancedb_config)
    return {"rows_written": len(cleaned_rows)}
