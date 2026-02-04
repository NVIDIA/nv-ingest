from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.util.nim import infer_microservice


@dataclass(frozen=True)
class RecallConfig:
    lancedb_uri: str
    lancedb_table: str
    embedding_endpoint: str
    embedding_model: str
    embedding_api_key: str = ""
    top_k: int = 10
    ks: Sequence[int] = (1, 3, 5, 10)


def _normalize_query_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gt_page" in df.columns and "page" not in df.columns:
        df = df.rename(columns={"gt_page": "page"})
    required = {"query", "pdf", "page"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Query CSV must contain columns {sorted(required)} (missing: {sorted(missing)})")
    df["pdf"] = df["pdf"].astype(str).str.replace(".pdf", "", regex=False)
    df["page"] = df["page"].astype(str)
    df["golden_answer"] = df.apply(lambda x: f"{x.pdf}_{x.page}", axis=1)
    return df


def _embed_queries_nim(
    queries: List[str],
    *,
    endpoint: str,
    model: str,
    api_key: str,
) -> List[List[float]]:
    # `infer_microservice` returns a list of embeddings.
    embeddings = infer_microservice(
        ["query: " + q for q in queries],
        model_name=model,
        embedding_endpoint=endpoint,
        nvidia_api_key=(api_key or "").strip(),
        grpc="http" not in endpoint,
    )
    # Some backends return numpy arrays; normalize to list-of-list floats.
    out: List[List[float]] = []
    for e in embeddings:
        if isinstance(e, np.ndarray):
            out.append(e.astype("float32").tolist())
        else:
            out.append(list(e))
    return out


def _search_lancedb(
    *,
    lancedb_uri: str,
    table_name: str,
    query_vectors: List[List[float]],
    top_k: int,
) -> List[List[Dict[str, Any]]]:
    import lancedb  # type: ignore

    db = lancedb.connect(lancedb_uri)
    table = db.open_table(table_name)

    results: List[List[Dict[str, Any]]] = []
    for v in query_vectors:
        q = np.asarray(v, dtype="float32")
        hits = (
            table.search(q, vector_column_name="vector")
            .select(["pdf_page", "pdf_basename", "page_number", "source_id", "path", "_distance"])
            .limit(top_k)
            .to_list()
        )
        results.append(hits)
    return results


def _recall_at_k(gold: List[str], retrieved: List[List[str]], k: int) -> float:
    hits = 0
    for g, r in zip(gold, retrieved):
        if g in (r[:k] if r else []):
            hits += 1
    return hits / max(1, len(gold))


def evaluate_recall(
    query_csv: Path,
    *,
    cfg: RecallConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    df_query = _normalize_query_df(pd.read_csv(query_csv))
    queries = df_query["query"].astype(str).tolist()
    gold = df_query["golden_answer"].tolist()

    vectors = _embed_queries_nim(
        queries,
        endpoint=cfg.embedding_endpoint,
        model=cfg.embedding_model,
        api_key=cfg.embedding_api_key,
    )
    raw_hits = _search_lancedb(
        lancedb_uri=cfg.lancedb_uri,
        table_name=cfg.lancedb_table,
        query_vectors=vectors,
        top_k=cfg.top_k,
    )

    retrieved_keys: List[List[str]] = []
    for hits in raw_hits:
        keys: List[str] = []
        for h in hits:
            # Prefer explicit `pdf_page` column; fall back to derived form.
            if h.get("pdf_page"):
                keys.append(str(h["pdf_page"]))
            else:
                pdf = h.get("pdf_basename") or (Path(h["path"]).name if h.get("path") else None)
                page = h.get("page_number")
                keys.append(f"{pdf}_{page}" if (pdf is not None and page is not None) else "")
        retrieved_keys.append([k for k in keys if k])

    metrics = {f"recall@{k}": _recall_at_k(gold, retrieved_keys, int(k)) for k in cfg.ks}

    # Build per-query analysis DataFrame
    rows = []
    for i, (q, g, r) in enumerate(zip(queries, gold, retrieved_keys)):
        row = {"query_id": i, "query": q, "golden_answer": g, "top_retrieved": r[: cfg.top_k]}
        for k in cfg.ks:
            k = int(k)
            row[f"hit@{k}"] = g in r[:k]
            row[f"rank@{k}"] = (r[: cfg.top_k].index(g) + 1) if (g in r[: cfg.top_k]) else None
        rows.append(row)
    results_df = pd.DataFrame(rows)

    saved = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = output_dir / f"recall_results_{ts}.csv"
        results_df.to_csv(out, index=False)
        saved["results_csv"] = str(out)

    return {
        "n_queries": len(df_query),
        "top_k": cfg.top_k,
        "metrics": metrics,
        "saved": saved,
    }

