# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BEIR Evaluator — Designer component for running BEIR evaluation against LanceDB.

Reuses the existing evaluation logic from ``nemo_retriever.recall.beir`` and
prints the standard run summary via
``nemo_retriever.tools.recall.core.print_run_summary``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated, Any, Optional

from nemo_retriever.graph.designer import Param, designer_component

logger = logging.getLogger(__name__)


@designer_component(
    name="BEIR Evaluator",
    category="Evaluation",
    compute="cpu",
    description="Runs BEIR evaluation against a LanceDB table and prints the standard run summary",
    category_color="#42d6a4",
    component_type="pipeline_evaluator",
)
class BEIREvaluatorActor:
    """Designer BEIR evaluation node against an existing LanceDB table.

    Assumes vectors were already written (for example via
    :class:`~nemo_retriever.vdb.operators.IngestVdbOperator` or ``retriever
    ingest``). After evaluation, calls ``print_run_summary`` like
    the batch pipeline.
    """

    def __init__(
        self,
        lancedb_uri: Annotated[str, Param(label="LanceDB URI", placeholder="/path/to/lancedb")] = "lancedb",
        lancedb_table: Annotated[str, Param(label="Table Name")] = "nemo-retriever",
        embedding_model: Annotated[str | None, Param(label="Embedding Model")] = None,
        beir_loader: Annotated[str, Param(label="BEIR Loader", choices=["vidore_hf"])] = "vidore_hf",
        beir_dataset_name: Annotated[
            str, Param(label="BEIR Dataset Name", placeholder="e.g. vidore_v3_computer_science")
        ] = "",
        beir_split: Annotated[str, Param(label="BEIR Split")] = "test",
        beir_query_language: Annotated[str, Param(label="Query Language", placeholder="Optional (e.g. en, fr)")] = "",
        beir_doc_id_field: Annotated[
            str,
            Param(label="Doc ID Field", choices=["pdf_basename", "pdf_page", "source_id", "path"]),
        ] = "pdf_basename",
        beir_ks: Annotated[str, Param(label="K Values", placeholder="1,3,5,10")] = "1,3,5,10",
        hybrid: Annotated[bool, Param(label="Hybrid Search")] = False,
    ) -> None:
        self.lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        self.lancedb_table = lancedb_table
        self.embedding_model = embedding_model
        self.beir_loader = beir_loader
        self.beir_dataset_name = beir_dataset_name
        self.beir_split = beir_split
        self.beir_query_language = beir_query_language or None
        self.beir_doc_id_field = beir_doc_id_field
        self.hybrid = hybrid

        self._ks: tuple[int, ...] = (
            tuple(int(k) for k in beir_ks.split(",") if k.strip()) if isinstance(beir_ks, str) else tuple(beir_ks)
        )
        if not self._ks:
            self._ks = (1, 3, 5, 10)

    def evaluate(self) -> dict[str, Any]:
        """Run the configured evaluation and print the standard run summary.

        Returns the ``summary_dict`` produced by ``print_run_summary``.
        """
        from nemo_retriever.models import resolve_embed_model
        from nemo_retriever.tools.recall.beir import BeirConfig, evaluate_lancedb_beir
        from nemo_retriever.tools.recall.core import print_run_summary

        resolved_model = resolve_embed_model(self.embedding_model) if self.embedding_model else None

        evaluation_label = "BEIR"
        evaluation_total_time = 0.0
        evaluation_metrics: dict[str, float] = {}
        evaluation_query_count: Optional[int] = None

        beir_cfg = BeirConfig(
            lancedb_uri=self.lancedb_uri,
            lancedb_table=self.lancedb_table,
            embedding_model=resolved_model,
            loader=self.beir_loader,
            dataset_name=self.beir_dataset_name,
            split=self.beir_split,
            query_language=self.beir_query_language,
            doc_id_field=self.beir_doc_id_field,
            ks=self._ks,
            hybrid=self.hybrid,
        )
        eval_start = time.perf_counter()
        beir_dataset, _raw_hits, _run, evaluation_metrics = evaluate_lancedb_beir(beir_cfg)
        evaluation_total_time = time.perf_counter() - eval_start
        evaluation_query_count = len(beir_dataset.query_ids)

        summary_dict = print_run_summary(
            processed_pages=-1,
            input_path=Path(self.lancedb_uri),
            vdb_op="lancedb",
            vdb_kwargs={"uri": self.lancedb_uri, "table_name": self.lancedb_table, "hybrid": self.hybrid},
            total_time=-1,
            ingest_only_total_time=-1,
            ray_dataset_download_total_time=-1,
            vdb_upload_total_time=-1,
            evaluation_total_time=evaluation_total_time,
            evaluation_metrics=evaluation_metrics,
            recall_total_time=0.0,
            recall_metrics={},
            evaluation_label=evaluation_label,
            evaluation_count=evaluation_query_count,
        )
        return summary_dict
