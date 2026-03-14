# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from nemo_retriever.retriever import Retriever
import json

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecallConfig:
    lancedb_uri: str
    lancedb_table: str
    embedding_model: str
    # Embedding endpoints (optional).
    #
    # If neither HTTP nor gRPC endpoint is provided (and embedding_endpoint is empty),
    # stage7 will fall back to local HuggingFace embeddings via:
    #   nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder
    embedding_http_endpoint: Optional[str] = None
    embedding_grpc_endpoint: Optional[str] = None
    # Back-compat single endpoint string (http URL or host:port for gRPC).
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = ""
    top_k: int = 10
    ks: Sequence[int] = (1, 3, 5, 10)
    # ANN search tuning (LanceDB IVF_HNSW_SQ).
    # nprobes=0 means "search all partitions" (exhaustive); refine_factor re-ranks
    # top candidates with full-precision vectors to eliminate SQ quantization error.
    nprobes: int = 0
    refine_factor: int = 10
    hybrid: bool = False
    # Local HF knobs (only used when endpoints are missing).
    local_hf_device: Optional[str] = None
    local_hf_cache_dir: Optional[str] = None
    local_hf_batch_size: int = 64
    # Gold/retrieval comparison mode:
    # - pdf_page: compare on "{pdf}_{page}" keys
    # - pdf_only: compare on "{pdf}" document keys
    match_mode: str = "pdf_page"
    reranker: Optional[str] = None
    reranker_endpoint: Optional[str] = None
    reranker_api_key: str = ""
    reranker_batch_size: int = 32


def _normalize_pdf_name(value: str) -> str:
    return str(value).replace(".pdf", "")


def _normalize_query_df(df: pd.DataFrame, *, match_mode: str) -> pd.DataFrame:
    """
    Normalize a query CSV into:
      - query (string)
      - golden_answer (string key that should match LanceDB `pdf_page`)

    Supported inputs by match mode:
      - pdf_page:
        - query,pdf_page
        - query,pdf,page (or query,pdf,gt_page)
      - pdf_only:
        - query,expected_pdf
        - query,pdf
    """
    if match_mode not in {"pdf_page", "pdf_only"}:
        raise ValueError(f"Unsupported recall match mode: {match_mode}")

    df = df.copy()

    if "query" not in df.columns:
        raise KeyError("Query CSV must contain a 'query' column.")

    if match_mode == "pdf_only":
        if "expected_pdf" in df.columns:
            df["golden_answer"] = df["expected_pdf"].astype(str).apply(_normalize_pdf_name)
            return df
        if "pdf" in df.columns:
            df["golden_answer"] = df["pdf"].astype(str).apply(_normalize_pdf_name)
            return df
        raise KeyError(
            "For pdf_only mode, query data must contain ['query','expected_pdf'] or ['query','pdf'] columns."
        )

    if "gt_page" in df.columns and "page" not in df.columns:
        df = df.rename(columns={"gt_page": "page"})

    if "pdf_page" in df.columns:
        df["golden_answer"] = df["pdf_page"].astype(str)
        return df

    required = {"pdf", "page"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "Query CSV must contain either columns ['query','pdf_page'] or ['query','pdf','page'] "
            f"(missing: {sorted(missing)})"
        )

    df["pdf"] = df["pdf"].astype(str).str.replace(".pdf", "", regex=False)
    df["page"] = df["page"].astype(str)
    df["golden_answer"] = df.apply(lambda x: f"{x.pdf}_{x.page}", axis=1)
    return df


def _resolve_embedding_endpoint(cfg: RecallConfig) -> Tuple[Optional[str], Optional[bool]]:
    """
    Resolve which embedding endpoint to use.

    Returns (endpoint, use_grpc) where:
      - endpoint is either an http(s) URL or a host:port string for gRPC
      - use_grpc is True for gRPC, False for HTTP, None when no endpoint is configured
    """
    http_ep = (cfg.embedding_http_endpoint or "").strip() if isinstance(cfg.embedding_http_endpoint, str) else None
    grpc_ep = (cfg.embedding_grpc_endpoint or "").strip() if isinstance(cfg.embedding_grpc_endpoint, str) else None
    single = (cfg.embedding_endpoint or "").strip() if isinstance(cfg.embedding_endpoint, str) else None

    if http_ep:
        return http_ep, False
    if grpc_ep:
        return grpc_ep, True
    if single:
        # Infer protocol: if a URL scheme is present, treat as HTTP; otherwise gRPC.
        return single, (not single.lower().startswith("http"))

    return None, None


def _embed_queries_nim(
    queries: List[str],
    *,
    endpoint: str,
    model: str,
    api_key: str,
    grpc: bool,
) -> List[List[float]]:
    from nv_ingest_api.util.nim import infer_microservice

    # `infer_microservice` returns a list of embeddings.
    embeddings = infer_microservice(
        queries,
        model_name=model,
        embedding_endpoint=endpoint,
        nvidia_api_key=(api_key or "").strip(),
        grpc=bool(grpc),
        input_type="query",
    )
    # Some backends return numpy arrays; normalize to list-of-list floats.
    out: List[List[float]] = []
    for e in embeddings:
        if isinstance(e, np.ndarray):
            out.append(e.astype("float32").tolist())
        else:
            out.append(list(e))
    return out


def _embed_queries_local_hf(
    queries: List[str],
    *,
    device: Optional[str],
    cache_dir: Optional[str],
    batch_size: int,
    model_name: Optional[str] = None,
) -> List[List[float]]:
    from nemo_retriever.model import create_local_embedder, is_vl_embed_model

    embedder = create_local_embedder(model_name, device=device, hf_cache_dir=cache_dir)

    if is_vl_embed_model(model_name):
        vecs = embedder.embed_queries(queries, batch_size=int(batch_size))
    else:
        vecs = embedder.embed(["query: " + q for q in queries], batch_size=int(batch_size))
    return vecs.detach().to("cpu").tolist()


def _hits_to_keys(raw_hits: List[List[Dict[str, Any]]]) -> List[List[str]]:
    retrieved_keys: List[List[str]] = []
    for hits in raw_hits:
        keys: List[str] = []
        for h in hits:
            page_number = h["page_number"]
            source = h["source"]
            # Prefer explicit `pdf_page` column; fall back to derived form.
            if page_number is not None and source:
                filename = Path(source).stem
                keys.append(f"{filename}_{str(page_number)}")
            else:
                logger.warning(
                    "Skipping hit with missing page_number or source_id: metadata=%s source=%s",
                    h.get("metadata", ""),
                    h.get("source", ""),
                )
        retrieved_keys.append([k for k in keys if k])
    return retrieved_keys


def _extract_doc_from_pdf_page(key: str) -> str:
    parts = str(key).rsplit("_", 1)
    if len(parts) != 2:
        return str(key)
    return parts[0]


def _is_hit(golden_key: str, retrieved: List[str], k: int, *, match_mode: str) -> bool:
    """Check if a golden key is found in the top-k retrieved keys.

    Handles filenames with underscores via ``rsplit`` and also accepts
    whole-document keys (page ``-1``).
    """
    if match_mode == "pdf_only":
        gold_doc = _normalize_pdf_name(str(golden_key))
        top_docs = [_extract_doc_from_pdf_page(r) for r in retrieved[:k]]
        return gold_doc in top_docs

    parts = golden_key.rsplit("_", 1)
    if len(parts) != 2:
        return golden_key in retrieved[:k]
    filename, page = parts
    specific_page = f"{filename}_{page}"
    entire_document = f"{filename}_-1"
    top = retrieved[:k]
    return specific_page in top or entire_document in top


def is_hit_at_k(golden_key: str, retrieved: Sequence[str], k: int, *, match_mode: str) -> bool:
    """Public wrapper for top-k hit checks across match modes."""
    return _is_hit(str(golden_key), list(retrieved), int(k), match_mode=str(match_mode))


def gold_to_doc_page(golden_key: str) -> tuple[str, str]:
    """Split a golden key like ``"docname_page"`` into ``(doc, page)``."""
    s = str(golden_key)
    if "_" not in s:
        return s, ""
    doc, page = s.rsplit("_", 1)
    return doc, page


def hit_key_and_distance(hit: dict) -> tuple[str | None, float | None]:
    """Extract ``(pdf_page key, distance)`` from a single LanceDB hit dict.

    Supports both ``_distance`` and ``_score`` fields for compatibility across
    LanceDB query types (vector vs hybrid).
    """
    try:
        res = json.loads(hit.get("metadata", "{}"))
        source = json.loads(hit.get("source", "{}"))
    except Exception:
        return None, None

    source_id = source.get("source_id")
    page_number = res.get("page_number")
    if not source_id or page_number is None:
        return None, float(hit.get("_distance")) if "_distance" in hit else None

    key = f"{Path(str(source_id)).stem}_{page_number}"
    dist = float(hit["_distance"]) if "_distance" in hit else float(hit["_score"]) if "_score" in hit else None
    return key, dist


def _recall_at_k(gold: List[str], retrieved: List[List[str]], k: int, *, match_mode: str) -> float:
    hits = sum(is_hit_at_k(g, r, k, match_mode=match_mode) for g, r in zip(gold, retrieved))
    return hits / max(1, len(gold))


def retrieve_and_score(
    query_csv: Path,
    *,
    cfg: RecallConfig,
    limit: Optional[int] = None,
    vector_column_name: str = "vector",
) -> Tuple[pd.DataFrame, List[str], List[List[Dict[str, Any]]], List[List[str]], Dict[str, float]]:
    """
    Run embeddings + LanceDB retrieval for a query CSV.

    Returns:
      - normalized query DataFrame
      - gold keys
      - raw LanceDB hits
      - retrieved keys (pdf_page-like)
      - metrics dict (recall@k)
    """
    df_query = _normalize_query_df(pd.read_csv(query_csv), match_mode=str(cfg.match_mode))
    if limit is not None:
        df_query = df_query.head(int(limit)).copy()

    queries = df_query["query"].astype(str).tolist()
    gold = df_query["golden_answer"].astype(str).tolist()
    endpoint, use_grpc = _resolve_embedding_endpoint(cfg)
    retriever = Retriever(
        lancedb_uri=cfg.lancedb_uri,
        lancedb_table=cfg.lancedb_table,
        embedder=cfg.embedding_model or "nvidia/llama-nemotron-embed-1b-v2",
        embedding_http_endpoint=cfg.embedding_http_endpoint,
        embedding_api_key=cfg.embedding_api_key,
        top_k=cfg.top_k,
        nprobes=cfg.nprobes,
        refine_factor=cfg.refine_factor,
        hybrid=bool(cfg.hybrid),
        local_hf_device=cfg.local_hf_device,
        local_hf_cache_dir=cfg.local_hf_cache_dir,
        local_hf_batch_size=cfg.local_hf_batch_size,
        reranker=cfg.reranker,
        reranker_endpoint=cfg.reranker_endpoint,
        reranker_api_key=cfg.reranker_api_key,
        reranker_batch_size=cfg.reranker_batch_size,
    )
    start = time.time()
    raw_hits = retriever.queries(queries)
    end_queries = time.time() - start
    print(
        f"Retrieval time for {len(queries)} ",
        f"queries: {end_queries:.2f} seconds ",
        f"(average {len(queries)/end_queries:.2f} queries/second)",
    )

    retrieved_keys = _hits_to_keys(raw_hits)
    metrics = {
        f"recall@{k}": _recall_at_k(gold, retrieved_keys, int(k), match_mode=str(cfg.match_mode)) for k in cfg.ks
    }
    return df_query, gold, raw_hits, retrieved_keys, metrics


def evaluate_recall(
    query_csv: Path,
    *,
    cfg: RecallConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    df_query, gold, raw_hits, retrieved_keys, metrics = retrieve_and_score(
        query_csv,
        cfg=cfg,
        limit=None,
        vector_column_name="vector",
    )

    # Build per-query analysis DataFrame
    rows = []
    for i, (q, g, r) in enumerate(zip(df_query["query"].astype(str).tolist(), gold, retrieved_keys)):
        row = {"query_id": i, "query": q, "golden_answer": g, "top_retrieved": r[: cfg.top_k]}
        for k in cfg.ks:
            k = int(k)
            row[f"hit@{k}"] = is_hit_at_k(g, r, k, match_mode=str(cfg.match_mode))
            if str(cfg.match_mode) == "pdf_only":
                top_docs = [_extract_doc_from_pdf_page(key) for key in r[: cfg.top_k]]
                try:
                    row[f"rank@{k}"] = top_docs.index(_normalize_pdf_name(str(g))) + 1
                except ValueError:
                    row[f"rank@{k}"] = None
            else:
                row[f"rank@{k}"] = (r[: cfg.top_k].index(g) + 1) if (g in r[: cfg.top_k]) else None
        rows.append(row)
    results_df = pd.DataFrame(rows)

    saved: Dict[str, str] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = output_dir / f"recall_results_{ts}.csv"
        results_df.to_csv(out, index=False)
        saved["results_csv"] = str(out)

    return {
        "n_queries": int(len(df_query)),
        "top_k": int(cfg.top_k),
        "metrics": metrics,
        "saved": saved,
    }
