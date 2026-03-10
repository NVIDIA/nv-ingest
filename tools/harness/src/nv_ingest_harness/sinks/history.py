# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""SQLite history sink for tracking benchmark results over time."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nv_ingest_harness.sinks.base import Sink

DEFAULT_DB_PATH = Path(__file__).parents[3] / "history.db"

METRIC_COLUMNS = frozenset(
    {
        "result_count",
        "failure_count",
        "ingestion_time_s",
        "pages_per_second",
        "total_pages",
        "text_chunks",
        "table_chunks",
        "chart_chunks",
        "retrieval_time_s",
        "recall_at_5",
        "recall_at_5_reranker",
    }
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_name TEXT,
    dataset TEXT NOT NULL,
    git_commit TEXT,
    result_count INTEGER,
    failure_count INTEGER,
    ingestion_time_s REAL,
    pages_per_second REAL,
    total_pages INTEGER,
    text_chunks INTEGER,
    table_chunks INTEGER,
    chart_chunks INTEGER,
    retrieval_time_s REAL,
    recall_at_5 REAL,
    recall_at_5_reranker REAL,
    requirements_met INTEGER,
    raw_json TEXT
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_runs_dataset_ts ON runs(dataset, timestamp);
"""


class HistorySink(Sink):
    """SQLite sink for historical result storage."""

    name: str = "history"

    def __init__(self, sink_config: dict[str, Any]):
        self.sink_config = sink_config
        self.enabled = sink_config.get("enabled", True)
        self.db_path = sink_config.get("db_path") or os.environ.get("HARNESS_HISTORY_DB") or str(DEFAULT_DB_PATH)
        self.retention_days = sink_config.get("retention_days")
        self.session_name: str | None = None
        self.env_data: dict[str, Any] = {}
        self.conn: sqlite3.Connection | None = None

    def initialize(self, session_name: str, env_data: dict[str, Any]) -> None:
        self.session_name = session_name
        self.env_data = env_data

        if not self.enabled:
            return

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(CREATE_TABLE_SQL)
        self.conn.execute(CREATE_INDEX_SQL)
        self.conn.commit()

        if self.retention_days:
            self._prune_old_runs(self.retention_days)

        print(f"HistorySink: Connected to {self.db_path}")

    def process_result(
        self,
        result: dict[str, Any],
        entry_config: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or not self.conn:
            return

        metrics = result.get("metrics", {})
        requirements_status = result.get("requirements_status", [])
        all_requirements_met = (
            all(r.get("status") == "pass" for r in requirements_status) if requirements_status else None
        )

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_name": self.session_name,
            "dataset": result.get("dataset", "unknown"),
            "git_commit": self.env_data.get("git_commit"),
            "result_count": metrics.get("result_count"),
            "failure_count": metrics.get("failure_count"),
            "ingestion_time_s": metrics.get("ingestion_time_s"),
            "pages_per_second": metrics.get("pages_per_second"),
            "total_pages": metrics.get("total_pages"),
            "text_chunks": metrics.get("text_chunks"),
            "table_chunks": metrics.get("table_chunks"),
            "chart_chunks": metrics.get("chart_chunks"),
            "retrieval_time_s": metrics.get("retrieval_time_s"),
            "recall_at_5": metrics.get("recall_multimodal_@5_no_reranker"),
            "recall_at_5_reranker": metrics.get("recall_multimodal_@5_reranker"),
            "requirements_met": 1 if all_requirements_met else (0 if all_requirements_met is False else None),
            "raw_json": json.dumps(result),
        }

        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        sql = f"INSERT INTO runs ({columns}) VALUES ({placeholders})"

        self.conn.execute(sql, list(row.values()))
        self.conn.commit()

    def finalize(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
            print("HistorySink: Connection closed.")

    def _prune_old_runs(self, days: int) -> None:
        if not self.conn:
            return
        self.conn.execute("DELETE FROM runs WHERE timestamp < datetime('now', ?)", (f"-{days} days",))
        self.conn.commit()

    @classmethod
    def get_connection(cls, db_path: str | None = None) -> sqlite3.Connection:
        path = db_path or os.environ.get("HARNESS_HISTORY_DB") or str(DEFAULT_DB_PATH)
        return sqlite3.connect(path, timeout=30.0)

    @classmethod
    def get_recent_runs(
        cls,
        dataset: str,
        limit: int = 10,
        db_path: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = cls.get_connection(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM runs WHERE dataset = ? ORDER BY timestamp DESC LIMIT ?",
            (dataset, limit),
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    @classmethod
    def get_trend(
        cls,
        dataset: str,
        metric: str,
        days: int = 30,
        db_path: str | None = None,
    ) -> list[tuple[str, float]]:
        if metric not in METRIC_COLUMNS:
            raise ValueError(f"Invalid metric: {metric}")
        conn = cls.get_connection(db_path)
        cursor = conn.execute(
            f"SELECT timestamp, {metric} FROM runs WHERE dataset = ? "
            f"AND timestamp > datetime('now', ?) AND {metric} IS NOT NULL ORDER BY timestamp ASC",
            (dataset, f"-{days} days"),
        )
        results = cursor.fetchall()
        conn.close()
        return results
