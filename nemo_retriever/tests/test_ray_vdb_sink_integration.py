# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Ray and LanceDB coverage for streaming VDB ingestion."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pyarrow as pa
import pytest

ray = pytest.importorskip("ray", minversion="2.56.1")
lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.vdb import IngestVdbOperator

_GRAPH_SCHEMA = pa.schema(
    [
        pa.field("block_id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field(
            "text_embeddings_1b_v2",
            pa.struct([pa.field("embedding", pa.list_(pa.float32()))]),
        ),
        pa.field("source_id", pa.string()),
        pa.field("page_number", pa.int64()),
        pa.field(
            "metadata",
            pa.struct(
                [
                    pa.field(
                        "content_metadata",
                        pa.struct(
                            [
                                pa.field("type", pa.string()),
                                pa.field("id", pa.string()),
                            ]
                        ),
                    )
                ]
            ),
        ),
        pa.field("result_only", pa.string()),
    ]
)


def _source_table(block_id: int) -> pa.Table:
    row_ids = range(block_id * 2, block_id * 2 + 2)
    return pa.Table.from_pylist(
        [
            {
                "block_id": block_id,
                "text": f"chunk-{row_id}",
                "text_embeddings_1b_v2": {"embedding": [float(row_id), 1.0]},
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "page_number": row_id,
                "metadata": {
                    "content_metadata": {
                        "type": "text",
                        "id": f"row-{row_id}",
                    }
                },
                "result_only": f"not-stored-{row_id}",
            }
            for row_id in row_ids
        ],
        schema=_GRAPH_SCHEMA,
    )


@pytest.mark.integration
@pytest.mark.parametrize("return_results", [True, False])
def test_ray_streams_three_blocks_into_real_lancedb_and_preserves_contract(
    tmp_path, monkeypatch, return_results
) -> None:
    """Streaming ingest avoids the global sink barrier and executes in order once."""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False)

    try:

        class _CompletedBlocks:
            def __init__(self) -> None:
                self._completed_blocks: list[int] = []

            def record_completed_block(self, block_id: int) -> None:
                self._completed_blocks.append(block_id)

            def completed_blocks(self) -> list[int]:
                return list(self._completed_blocks)

        def record_completed_block(batch: pa.Table, completed: Any) -> pa.Table:
            block_id = int(batch.column("block_id")[0].as_py())
            ray.get(completed.record_completed_block.remote(block_id))
            return batch

        completed_type = ray.remote(_CompletedBlocks)
        completed = completed_type.options(num_cpus=0).remote()
        source = ray.data.from_arrow([_source_table(i) for i in range(3)], override_num_blocks=3)
        assert source.num_blocks() == 3
        dataset = source.map_batches(
            record_completed_block,
            batch_format="pyarrow",
            batch_size=None,
            concurrency=3,
            fn_kwargs={"completed": completed},
        )

        class RequireFinalizedWrite(AbstractOperator):
            def preprocess(self, data, **kwargs):
                return data

            def process(self, data, **kwargs):
                table = lancedb.connect(str(tmp_path)).open_table("chunks")
                assert table.count_rows() == 6
                result = data.copy()
                result["after_sink"] = True
                return result

            def postprocess(self, data, **kwargs):
                return data

        graph = Graph() >> IngestVdbOperator(
            vdb_op="lancedb",
            vdb_kwargs={
                "uri": str(tmp_path),
                "table_name": "chunks",
                "vector_dim": 2,
                "overwrite": True,
                "build_index": False,
                "stream_batch_bytes": 512,
            },
        )
        if return_results:
            graph = graph >> RequireFinalizedWrite()
        executor = RayDataExecutor(graph)

        lazy = executor.build_dataset(dataset)
        assert isinstance(lazy, ray.data.Dataset)
        assert lancedb.connect(str(tmp_path)).list_tables().tables == []

        original_repartition = ray.data.Dataset.repartition

        def reject_global_repartition(self, *args, **kwargs):
            requested_blocks = kwargs.get("num_blocks", args[0] if args else None)
            if requested_blocks == 1:
                raise AssertionError("streaming ingest must not call repartition(num_blocks=1)")
            return original_repartition(self, *args, **kwargs)

        monkeypatch.setattr(ray.data.Dataset, "repartition", reject_global_repartition)

        result = executor.ingest(dataset, return_results=return_results)

        assert Counter(ray.get(completed.completed_blocks.remote())) == Counter({0: 1, 1: 1, 2: 1})

        if return_results:
            assert result["page_number"].tolist() == list(range(6))
            assert result["text"].tolist() == [f"chunk-{row_id}" for row_id in range(6)]
            assert result["result_only"].tolist() == [f"not-stored-{row_id}" for row_id in range(6)]
            assert result["after_sink"].tolist() == [True] * 6
        else:
            assert result.to_dict("records") == [{"input_rows": 6, "submitted_records": 6}]

        stored_table = lancedb.connect(str(tmp_path)).open_table("chunks")
        stored = stored_table.to_arrow().sort_by("id")
        assert stored.column_names == ["vector", "text", "metadata", "source", "id"]
        assert stored.column("id").to_pylist() == [f"row-{row_id}" for row_id in range(6)]
        assert not [name for name in stored_table.tags.list() if name.startswith("nemo_sink_")]
    finally:
        ray.shutdown()
