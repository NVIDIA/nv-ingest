# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator that appends embedding rows to a LanceDB table.

Designed for the service-mode ingest pipeline where pages arrive in
incremental batches.  Unlike :class:`LanceDBWriterActor` (which
overwrites on init), this operator lazily creates-or-opens the target
table and appends rows so that concurrent worker processes can all
write to the same LanceDB directory (LanceDB uses file-level locking
internally).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator

logger = logging.getLogger(__name__)


class LanceDBWriteOperator(AbstractOperator, CPUOperator):
    """Append embedding rows from a DataFrame batch to LanceDB.

    The table is created on first write and appended to thereafter.
    The operator is side-effect-only: it returns the input DataFrame
    unmodified.
    """

    def __init__(
        self,
        *,
        uri: str = "/var/lib/nemo-retriever/lancedb",
        table_name: str = "nemo-retriever",
        hybrid: bool = False,
        embedding_column: str = "embedding",
        text_column: str = "text",
        include_text: bool = True,
    ) -> None:
        super().__init__()
        self._uri = uri
        self._table_name = table_name
        self._hybrid = hybrid
        self._embedding_column = embedding_column
        self._text_column = text_column
        self._include_text = include_text
        self._db: Any = None
        self._table: Any = None

    def _ensure_connected(self, vector_dim: int) -> None:
        """Lazily connect to LanceDB and create-or-open the table.

        If the existing table's schema doesn't match the expected schema,
        drop and recreate it to avoid field-mismatch errors from stale data.
        """
        if self._db is not None:
            return

        import lancedb as _ldb
        from nemo_retriever.common.vdb.lancedb_schema import lancedb_schema

        ldb_path = Path(self._uri)
        ldb_path.mkdir(parents=True, exist_ok=True)

        self._db = _ldb.connect(uri=self._uri)
        expected_schema = lancedb_schema(vector_dim)
        expected_fields = {f.name for f in expected_schema}

        try:
            table = self._db.open_table(self._table_name)
            existing_fields = {f.name for f in table.schema}
            if not expected_fields.issubset(existing_fields):
                missing = expected_fields - existing_fields
                logger.warning(
                    "LanceDB table %r schema mismatch (missing: %s) — recreating",
                    self._table_name,
                    missing,
                )
                self._db.drop_table(self._table_name)
                raise KeyError("schema mismatch")
            self._table = table
            logger.debug("Opened existing LanceDB table %r at %s", self._table_name, self._uri)
        except Exception:
            self._table = self._db.create_table(
                self._table_name,
                schema=expected_schema,
                mode="create",
            )
            logger.info("Created LanceDB table %r at %s (dim=%d)", self._table_name, self._uri, vector_dim)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        from nemo_retriever.common.vdb.lancedb_schema import build_lancedb_rows, infer_vector_dim

        rows = build_lancedb_rows(
            data,
            embedding_column=self._embedding_column,
            text_column=self._text_column,
            include_text=self._include_text,
        )

        if not rows:
            logger.debug("No embedding rows in batch — skipping LanceDB write")
            return data

        vector_dim = infer_vector_dim(rows)
        if vector_dim <= 0:
            logger.warning("Could not infer embedding dimension — skipping LanceDB write")
            return data

        self._ensure_connected(vector_dim)
        self._table.add(rows)
        logger.debug("Wrote %d rows to LanceDB table %r", len(rows), self._table_name)

        return data

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
