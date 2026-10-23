# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small synthetic filesystem integration of the real LanceDB Arrow/Rust API."""

import numpy as np

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.lancedb_comparison import compare_dense_search


def _files(root):
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_real_lancedb_pair_matches_normal_indexed_search_without_index_mutation(tmp_path):
    vectors = np.random.default_rng(20261022).normal(size=(512, 8)).astype(np.float32).tolist()
    records = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": vector,
                    "content": f"synthetic row {ordinal}",
                    "content_metadata": {"page_number": ordinal + 1, "type": "text"},
                    "source_metadata": {"source_id": "/synthetic/doc.txt", "source_name": "doc.txt"},
                },
            }
            for ordinal, vector in enumerate(vectors)
        ]
    ]
    backend = LanceDB(
        uri=str(tmp_path),
        table_name="docs",
        vector_dim=8,
        hybrid=False,
        embedding_model_name="synthetic-embedder",
        embedding_model_revision="synthetic-revision",
        num_partitions=2,
        num_sub_vectors=2,
        expiration_cleanup_enabled=False,
    )
    backend.run(records)
    assert backend.get_index_metadata("embedding_model_name") == "synthetic-embedder"
    assert backend.get_index_metadata("embedding_model_revision") == "synthetic-revision"
    before = _files(tmp_path)
    query = [list(vectors[17])]
    options = dict(top_k=17, where="text IS NOT NULL", result_fields=["id", "text", "metadata", "source"])
    normal = backend.retrieval(query, **options)
    indexed, exact, evidence = compare_dense_search(
        backend,
        query,
        expected_model_name="synthetic-embedder",
        expected_model_revision="synthetic-revision",
        **options,
    )
    assert indexed == normal
    assert len(indexed) == len(exact) == 1
    assert len(indexed[0]) == len(exact[0]) == 17
    assert exact[0][0]["text"] == "synthetic row 17"
    assert evidence["metric"] == "l2" and evidence["index_type"] == "IVF_HNSW_SQ"
    assert evidence["table_rows"] == evidence["indexed_rows"] == 512
    assert evidence["unindexed_rows"] == 0 and evidence["vector_dimension"] == 8
    assert evidence["indexed_vector_fingerprint"] == evidence["exact_vector_fingerprint"]
    assert len(evidence["vector_hashes"]) == 1
    assert backend._connect().open_table("docs").version == evidence["table_version"]
    assert _files(tmp_path) == before
