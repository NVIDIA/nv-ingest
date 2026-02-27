# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class retriever:
    """Simple query helper over LanceDB with configurable embedders."""

    lancedb_uri: str = "lancedb"
    lancedb_table: str = "nv-ingest"
    embedder: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    embedding_http_endpoint: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = ""
    top_k: int = 10
    vector_column_name: str = "vector"
    nprobes: int = 0
    refine_factor: int = 10
    hybrid: bool = False
    local_hf_device: Optional[str] = None
    local_hf_cache_dir: Optional[Path] = None
    local_hf_batch_size: int = 64

    def _resolve_embedding_endpoint(self) -> Optional[str]:
        http_ep = self.embedding_http_endpoint.strip() if isinstance(self.embedding_http_endpoint, str) else None
        single = self.embedding_endpoint.strip() if isinstance(self.embedding_endpoint, str) else None

        if http_ep:
            return http_ep
        if single:
            if not single.lower().startswith("http"):
                raise ValueError("gRPC endpoints are not supported; provide an HTTP NIM endpoint URL.")
            return single
        return None

    def _embed_queries_nim(
        self,
        query_texts: list[str],
        *,
        endpoint: str,
        model: str,
    ) -> list[list[float]]:
        import numpy as np
        from nv_ingest_api.util.nim import infer_microservice

        embeddings = infer_microservice(
            query_texts,
            model_name=model,
            embedding_endpoint=endpoint,
            nvidia_api_key=(self.embedding_api_key or "").strip(),
            input_type="query",
        )
        out: list[list[float]] = []
        for embedding in embeddings:
            if isinstance(embedding, np.ndarray):
                out.append(embedding.astype("float32").tolist())
            else:
                out.append(list(embedding))
        return out

    def _embed_queries_local_hf(self, query_texts: list[str], *, model_name: str) -> list[list[float]]:
        from nemo_retriever.model import is_vl_embed_model, resolve_embed_model

        model_id = resolve_embed_model(model_name)
        cache_dir = str(self.local_hf_cache_dir) if self.local_hf_cache_dir else None

        if is_vl_embed_model(model_name):
            from nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder import (
                LlamaNemotronEmbedVL1BV2Embedder,
            )

            embedder = LlamaNemotronEmbedVL1BV2Embedder(
                device=self.local_hf_device,
                hf_cache_dir=cache_dir,
                model_id=model_id,
            )
            vectors = embedder.embed_queries(query_texts, batch_size=int(self.local_hf_batch_size))
        else:
            from nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder

            embedder = LlamaNemotronEmbed1BV2Embedder(
                device=self.local_hf_device,
                hf_cache_dir=cache_dir,
                normalize=True,
                model_id=model_id,
            )
            vectors = embedder.embed(["query: " + q for q in query_texts], batch_size=int(self.local_hf_batch_size))
        return vectors.detach().to("cpu").tolist()

    def _search_lancedb(
        self,
        *,
        lancedb_uri: str,
        query_vectors: list[list[float]],
        query_texts: list[str],
    ) -> list[list[dict[str, Any]]]:
        import lancedb  # type: ignore
        import numpy as np

        db = lancedb.connect(lancedb_uri)
        table = db.open_table(self.lancedb_table)

        effective_nprobes = int(self.nprobes)
        if effective_nprobes <= 0:
            try:
                for idx in table.list_indices():
                    num_parts = getattr(idx, "num_partitions", None)
                    if num_parts and int(num_parts) > 0:
                        effective_nprobes = int(num_parts)
                        break
            except Exception:
                pass
            if effective_nprobes <= 0:
                effective_nprobes = 16

        results: list[list[dict[str, Any]]] = []
        for i, vector in enumerate(query_vectors):
            q = np.asarray(vector, dtype="float32")
            if self.hybrid:
                from lancedb.rerankers import RRFReranker  # type: ignore

                hits = (
                    table.search(query_type="hybrid")
                    .vector(q)
                    .text(query_texts[i])
                    .nprobes(effective_nprobes)
                    .refine_factor(int(self.refine_factor))
                    .select(["text", "metadata", "source"])
                    .limit(int(self.top_k))
                    .rerank(RRFReranker())
                    .to_list()
                )
            else:
                hits = (
                    table.search(q, vector_column_name=self.vector_column_name)
                    .nprobes(effective_nprobes)
                    .refine_factor(int(self.refine_factor))
                    .select(["text", "metadata", "source", "_distance"])
                    .limit(int(self.top_k))
                    .to_list()
                )
            results.append(hits)
        return results

    def query(
        self,
        query: str,
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Run retrieval for a single query string."""
        return self.queries(
            [query],
            embedder=embedder,
            lancedb_uri=lancedb_uri,
        )[0]

    def queries(
        self,
        queries: Sequence[str],
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
    ) -> list[list[dict[str, Any]]]:
        """Run retrieval for multiple query strings."""
        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        resolved_embedder = str(embedder or self.embedder)
        resolved_lancedb_uri = str(lancedb_uri or self.lancedb_uri)

        endpoint = self._resolve_embedding_endpoint()
        if endpoint is not None:
            vectors = self._embed_queries_nim(
                query_texts,
                endpoint=endpoint,
                model=resolved_embedder,
            )
        else:
            vectors = self._embed_queries_local_hf(
                query_texts,
                model_name=resolved_embedder,
            )

        return self._search_lancedb(
            lancedb_uri=resolved_lancedb_uri,
            query_vectors=vectors,
            query_texts=query_texts,
        )
