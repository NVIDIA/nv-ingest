from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.util.nim import infer_microservice


@dataclass(frozen=True)
class RecallConfig:
    lancedb_uri: str
    lancedb_table: str
    embedding_model: str
    # Embedding endpoints (optional).
    #
    # If neither HTTP nor gRPC endpoint is provided (and embedding_endpoint is empty),
    # stage7 will fall back to local HuggingFace embeddings via:
    #   retriever.model.local.llama_nemotron_embed_1b_v2_embedder
    embedding_http_endpoint: Optional[str] = None
    embedding_grpc_endpoint: Optional[str] = None
    # Back-compat single endpoint string (http URL or host:port for gRPC).
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = ""
    top_k: int = 10
    ks: Sequence[int] = (1, 3, 5, 10)
    # Local HF knobs (only used when endpoints are missing).
    local_hf_device: Optional[str] = None
    local_hf_cache_dir: Optional[str] = None
    local_hf_batch_size: int = 64


def _normalize_query_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a query CSV into:
      - query (string)
      - golden_answer (string key that should match LanceDB `pdf_page`)

    Supported inputs:
      - query,pdf_page
      - query,pdf,page (or query,pdf,gt_page)
    """
    df = df.copy()
    if "gt_page" in df.columns and "page" not in df.columns:
        df = df.rename(columns={"gt_page": "page"})

    if "query" not in df.columns:
        raise KeyError("Query CSV must contain a 'query' column.")

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
    # `infer_microservice` returns a list of embeddings.
    embeddings = infer_microservice(
        ["query: " + q for q in queries],
        model_name=model,
        embedding_endpoint=endpoint,
        nvidia_api_key=(api_key or "").strip(),
        grpc=bool(grpc),
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
) -> List[List[float]]:
    # Lazy import: only load torch/HF when needed.
    from retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder

    embedder = LlamaNemotronEmbed1BV2Embedder(device=device, hf_cache_dir=cache_dir, normalize=True)
    vecs = embedder.embed(["query: " + q for q in queries], batch_size=int(batch_size))
    # Ensure list-of-list floats.
    return vecs.detach().to("cpu").tolist()


def _search_lancedb(
    *,
    lancedb_uri: str,
    table_name: str,
    query_vectors: List[List[float]],
    top_k: int,
    vector_column_name: str = "vector",
) -> List[List[Dict[str, Any]]]:
    import lancedb  # type: ignore

    db = lancedb.connect(lancedb_uri)
    table = db.open_table(table_name)

    results: List[List[Dict[str, Any]]] = []
    for v in query_vectors:
        q = np.asarray(v, dtype="float32")
        hits = (
            table.search(q, vector_column_name=vector_column_name)
            .select(["text", "metadata", "source", "_distance"])
            .limit(top_k)
            .to_list()
        )
        results.append(hits)
    return results


def _hits_to_keys(raw_hits: List[List[Dict[str, Any]]]) -> List[List[str]]:
    retrieved_keys: List[List[str]] = []
    for hits in raw_hits:
        keys: List[str] = []
        for h in hits:
            res = json.loads(h['metadata'])
            source = json.loads(h['source'])
            # Prefer explicit `pdf_page` column; fall back to derived form.
            if res.get("page_number") and source.get("source_id"):
                filename = Path(source["source_id"]).stem
                keys.append(filename + "_" + str(res["page_number"]))
            else:
                breakpoint()
                print(f"Big problem. Find me in source code and fix me! {res}")
        retrieved_keys.append([k for k in keys if k])
    return retrieved_keys


def _recall_at_k(gold: List[str], retrieved: List[List[str]], k: int) -> float:
    hits = 0
    for g, r in zip(gold, retrieved):
        filename = g.split("_")[0]
        page = g.split("_")[1]

        specific_page = f"{filename}_{page}"
        entire_document = f"{filename}_-1" # This indicates that the text was retrieved from the entire document at index time and therefore the exact page is unknown but it still is a hit just with the missing metadata in the VDB

        if specific_page in (r[:k] if r else []):
            hits += 1
        elif entire_document in (r[:k] if r else []):
            print(f"Entire document hit! {g} {r}")
            hits += 1
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
    df_query = _normalize_query_df(pd.read_csv(query_csv))
    if limit is not None:
        df_query = df_query.head(int(limit)).copy()

    queries = df_query["query"].astype(str).tolist()
    gold = df_query["golden_answer"].astype(str).tolist()

    endpoint, use_grpc = _resolve_embedding_endpoint(cfg)
    if endpoint is not None and use_grpc is not None:
        vectors = _embed_queries_nim(
            queries,
            endpoint=endpoint,
            model=cfg.embedding_model,
            api_key=cfg.embedding_api_key,
            grpc=bool(use_grpc),
        )
    else:
        vectors = _embed_queries_local_hf(
            queries,
            device=cfg.local_hf_device,
            cache_dir=cfg.local_hf_cache_dir,
            batch_size=int(cfg.local_hf_batch_size),
        )
    raw_hits = _search_lancedb(
        lancedb_uri=cfg.lancedb_uri,
        table_name=cfg.lancedb_table,
        query_vectors=vectors,
        top_k=int(cfg.top_k),
        vector_column_name=vector_column_name,
    )
    retrieved_keys = _hits_to_keys(raw_hits)
    metrics = {f"recall@{k}": _recall_at_k(gold, retrieved_keys, int(k)) for k in cfg.ks}
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
            row[f"hit@{k}"] = g in r[:k]
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

