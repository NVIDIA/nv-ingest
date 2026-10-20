# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end-ish check of ``/v1/query?format=evidence`` over a real LanceDB table.

Unlike ``test_service_vectordb_app.py`` (which mocks ``VectorDBState.search``), this
drives the full server path: real ``/internal/vectordb/write`` → real LanceDB vector
search → real ``normalize_retrieval_results`` → real ``build_evidence_result``. Only
the embedding *model* is stubbed (it needs a NIM/GPU otherwise); everything that the
evidence projection depends on — schema columns, the JSON-string ``metadata``
round-trip, score/locator/fidelity derivation — runs for real.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app

_DIM = 4
_RECORD = {
    "document_type": "text",
    "embedding": [1.0, 0.0, 0.0, 0.0],
    "metadata": {
        "content": "Revenue grew 12% year over year.",
        "content_metadata": {"page_number": 12, "type": "text"},
        "source_metadata": {
            "source_id": "/data/10k_2023.pdf",
            "source_name": "10k_2023.pdf",
        },
    },
}


def test_query_evidence_format_end_to_end_over_real_lancedb(tmp_path) -> None:
    backend = LanceDB(
        uri=str(tmp_path),
        table_name="nemo_retriever",
        overwrite=False,
        build_index=False,
        vector_dim=_DIM,
    )
    app = create_vectordb_app(
        vdb=backend,
        embed_endpoint="http://embed.example/v1/embeddings",  # -> embed_mode="remote"
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )

    # Stub ONLY the embedding model; real LanceDB does the rest.
    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"records": [[_RECORD]]})
            assert write.status_code == 200, write.text
            assert write.json()["total_rows"] == 1

            resp = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert list(body) == ["results", "query_mode"]
    assert body["query_mode"] == "classic"
    assert len(body["results"]) == 1

    item = body["results"][0]
    assert set(item) == {"evidence", "coverage"}

    ev = item["evidence"][0]
    assert ev["text"] == "Revenue grew 12% year over year."
    assert ev["source"] == "10k_2023"
    assert ev["citation"] == "10k_2023 p.12"
    assert ev["locator"] == {"kind": "page", "value": 12}
    assert ev["modality"] == "text"
    assert ev["fidelity"] == "verbatim"
    # Preserve the real LanceDB distance/relevance on the legacy evidence path.
    assert isinstance(ev["score"], (int, float))

    coverage = item["coverage"]
    assert coverage["strategies_used"] == ["dense"]
    assert coverage["n_docs_seen"] == 1
    assert coverage["thin_spots"] == ["single source"]
