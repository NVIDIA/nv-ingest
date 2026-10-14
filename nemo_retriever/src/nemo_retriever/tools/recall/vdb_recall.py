# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import pandas as pd  # noqa: F401
from rich.console import Console

from nemo_retriever.tools.recall.core import (  # noqa: F401
    RecallConfig,
    evaluate_recall,
    retrieve_and_score,
    _normalize_query_df,
)

app = typer.Typer(help="Embed query CSV rows, search LanceDB, print hits, and compute recall@k.")
console = Console()


def _resolve_query_csv(path: Path) -> Path:
    if path.exists():
        return path
    # Convenience fallback for common repo layout.
    for candidate in (Path("bo767_query_gt.csv"), Path("data/bo767_query_gt.csv")):
        if candidate.exists():
            return candidate
    raise typer.BadParameter(
        f"Query CSV not found at '{path}'. Tried also: 'bo767_query_gt.csv', 'data/bo767_query_gt.csv'."
    )


def _extract_hits(result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits = []
    for res in result:
        hits.append(
            {
                "text": res.get("text", {}),
                "metadata": res.get("metadata", {}),
                "source": res.get("source", {}),
                "distance": res.get("_distance", {}),
            }
        )
    return hits


def _coerce_endpoint_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("none", "null"):
        return None
    return s


def _normalize_local_query_embed_backend(value: str) -> str:
    raw = (value or "vllm").strip().lower()
    if raw not in ("hf", "vllm"):
        raise typer.BadParameter("local query embed backend must be 'hf' or 'vllm', " f"got {value!r}")
    return raw


def _resolve_query_embedding_endpoint(
    *,
    embedding_endpoint: Optional[str],
    embedding_http_endpoint: Optional[str],
    embedding_grpc_endpoint: Optional[str],
) -> tuple[Optional[str], Optional[bool]]:
    http_ep = _coerce_endpoint_str(embedding_http_endpoint)
    grpc_ep = _coerce_endpoint_str(embedding_grpc_endpoint)
    single = _coerce_endpoint_str(embedding_endpoint)

    if http_ep:
        return http_ep, False
    if grpc_ep:
        return grpc_ep, True
    if single:
        return single, not single.lower().startswith("http")
    return None, None


@app.command("recall-with-main")
def recall_with_main(
    query_csv: Path = typer.Option(
        Path("bo767_query_gt.csv"),
        "--query-csv",
        help="Query ground-truth CSV (expects columns: query,pdf_page OR query,pdf,page).",
    ),
    top_k: int = typer.Option(10, "--top-k", min=1, help="Top-k to print per query."),
    embedding_endpoint: Optional[str] = typer.Option(
        None,
        "--embedding-endpoint",
        help=(
            "Embedding endpoint (http(s) URL or host:port for gRPC). "
            "If omitted, you may specify --embedding-http-endpoint/--embedding-grpc-endpoint instead; "
            "if no endpoints are provided, falls back to local embeddings (vLLM by default)."
        ),
    ),
    embedding_http_endpoint: Optional[str] = typer.Option(
        None,
        "--embedding-http-endpoint",
        help="HTTP embedding endpoint URL (e.g. 'http://localhost:8012/v1').",
    ),
    embedding_grpc_endpoint: Optional[str] = typer.Option(
        None,
        "--embedding-grpc-endpoint",
        help="gRPC embedding endpoint (e.g. 'localhost:8013').",
    ),
    embedding_model: Optional[str] = typer.Option(
        None,
        "--embedding-model",
        help="Embedding model override. By default, use the model recorded in the index.",
    ),
    embedding_api_key: Optional[str] = typer.Option(None, "--embedding-api-key", help="Embedding API key (optional)."),
    lancedb_uri: str = typer.Option("lancedb", "--lancedb-uri", help="LanceDB database URI (directory path)."),
    table_name: str = typer.Option("nemo-retriever", "--table-name", help="LanceDB table name."),
    vector_column_name: str = typer.Option("vector", "--vector-column", help="Vector column name in the table."),
    local_hf_device: Optional[str] = typer.Option(
        None,
        "--local-hf-device",
        help="Device for local HF embeddings when --local-query-embed-backend is hf (e.g. 'cuda', 'cpu', 'cuda:0').",
    ),
    local_hf_cache_dir: Optional[Path] = typer.Option(
        None,
        "--local-hf-cache-dir",
        file_okay=False,
        dir_okay=True,
        help="Optional HuggingFace cache directory for local embeddings.",
    ),
    local_hf_batch_size: int = typer.Option(
        64,
        "--local-hf-batch-size",
        min=1,
        help="Batch size for local HF embedding inference.",
    ),
    local_query_embed_backend: str = typer.Option(
        "vllm",
        "--local-query-embed-backend",
        help=(
            "When no remote embedding endpoint is set: 'vllm' (default) uses the local vLLM path; "
            "'hf' uses HuggingFace."
        ),
    ),
) -> None:
    query_csv = _resolve_query_csv(Path(query_csv))
    _lqeb = _normalize_local_query_embed_backend(local_query_embed_backend)

    metrics_ks = (1, 5, 10)
    search_k = max(int(top_k), max(metrics_ks))
    query_embedding_endpoint, query_embedding_use_grpc = _resolve_query_embedding_endpoint(
        embedding_endpoint=embedding_endpoint,
        embedding_http_endpoint=embedding_http_endpoint,
        embedding_grpc_endpoint=embedding_grpc_endpoint,
    )

    cfg = RecallConfig(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(lancedb_uri),
            "table_name": str(table_name),
            "vector_column_name": str(vector_column_name),
        },
        query_embedder=embedding_model,
        embedding_endpoint=query_embedding_endpoint,
        embedding_api_key=(embedding_api_key or ""),
        embedding_use_grpc=query_embedding_use_grpc,
        top_k=int(search_k),
        ks=metrics_ks,
        local_hf_device=_coerce_endpoint_str(local_hf_device),
        local_hf_cache_dir=(str(local_hf_cache_dir) if local_hf_cache_dir is not None else None),
        local_hf_batch_size=int(local_hf_batch_size),
        local_query_embed_backend=_lqeb,
    )

    print("Reading and normalizing query CSV...")
    df_query, gold, raw_hits, retrieved_keys, metrics = retrieve_and_score(
        query_csv=query_csv,
        cfg=cfg,
        limit=None,
    )

    queries = df_query["query"].astype(str).tolist()
    has_separate_cols = "pdf" in df_query.columns and "page" in df_query.columns
    if has_separate_cols:
        gold_pdfs = df_query["pdf"].astype(str).tolist()
        gold_pages = df_query["page"].astype(str).tolist()

    print(f"Retrieval completed for {len(queries)} queries.")

    for idx, (query, g_key, hits) in enumerate(zip(queries, gold, raw_hits)):
        print(f"\nQuery: {query}")
        g_pdf, g_page = g_key.rsplit("_", 1)
        print(f"\tGold PDF: {g_pdf} - Gold Page: {g_page}")
        if has_separate_cols:
            print(f"\tOther PDF: {gold_pdfs[idx]} - Other Page: {gold_pages[idx]}")
        print(f"\tTop-k Results:")  # noqa: F541

        for i, h in enumerate(hits[:top_k]):
            meta = json.loads(h["metadata"]) if isinstance(h["metadata"], str) else h["metadata"]
            src = json.loads(h["source"]) if isinstance(h["source"], str) else h["source"]
            pdf_name = Path(src["source_id"]).stem
            page_number = meta.get("page_number", -1)
            print(f"\tResult {i}: {pdf_name} - {page_number}")

            if page_number == -1:
                print(f"Seeing page numbers -1 for query: {query}")

    print(f"\n\nRecall@1: {metrics['recall@1']}")
    print(f"Recall@5: {metrics['recall@5']}")
    print(f"Recall@10: {metrics['recall@10']}")


@app.command("run")
def run(
    query_csv: Path = typer.Option(
        Path("bo767_query_gt.csv"),
        "--query-csv",
        help="Query ground-truth CSV (expects columns: query,pdf_page OR query,pdf,page).",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Optionally limit number of queries."),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Top-k to print per query."),
    lancedb_uri: str = typer.Option("lancedb", "--lancedb-uri", help="LanceDB database URI (directory path)."),
    table_name: str = typer.Option("nemo-retriever", "--table-name", help="LanceDB table name."),
    vector_column_name: str = typer.Option("vector", "--vector-column", help="Vector column name in the table."),
    embedding_endpoint: Optional[str] = typer.Option(
        None,
        "--embedding-endpoint",
        help=(
            "Embedding endpoint (http(s) URL or host:port for gRPC). "
            "If omitted, you may specify --embedding-http-endpoint/--embedding-grpc-endpoint instead; "
            "if no endpoints are provided, recall falls back to local embeddings (vLLM by default)."
        ),
    ),
    embedding_http_endpoint: Optional[str] = typer.Option(
        None,
        "--embedding-http-endpoint",
        help="HTTP embedding endpoint URL (e.g. 'http://localhost:8012/v1').",
    ),
    embedding_grpc_endpoint: Optional[str] = typer.Option(
        None,
        "--embedding-grpc-endpoint",
        help="gRPC embedding endpoint (e.g. 'localhost:8013').",
    ),
    embedding_model: Optional[str] = typer.Option(
        None,
        "--embedding-model",
        help="Embedding model override. By default, use the model recorded in the index.",
    ),
    embedding_api_key: Optional[str] = typer.Option(None, "--embedding-api-key", help="Embedding API key (optional)."),
    local_hf_device: Optional[str] = typer.Option(
        None,
        "--local-hf-device",
        help="Device for local HF embeddings when --local-query-embed-backend is hf (e.g. 'cuda', 'cpu', 'cuda:0').",
    ),
    local_hf_cache_dir: Optional[Path] = typer.Option(
        None,
        "--local-hf-cache-dir",
        file_okay=False,
        dir_okay=True,
        help="Optional HuggingFace cache directory for local embeddings.",
    ),
    local_hf_batch_size: int = typer.Option(
        64,
        "--local-hf-batch-size",
        min=1,
        help="Batch size for local HF embedding inference.",
    ),
    local_query_embed_backend: str = typer.Option(
        "vllm",
        "--local-query-embed-backend",
        help=(
            "When no remote embedding endpoint is set: 'vllm' (default) uses the local vLLM path; "
            "'hf' uses HuggingFace."
        ),
    ),
    print_hits: bool = typer.Option(True, "--print-hits/--no-print-hits", help="Print top-k hits per query."),
) -> None:
    """
    Reads a query CSV, embeds each query, searches LanceDB, prints top-k results, and prints recall@1/@5/@10.

    Note: recall@10 requires retrieving at least 10 results per query; we will query LanceDB with
    search_k = max(top_k, 10) but only print the first `--top-k`.
    """
    query_csv = _resolve_query_csv(Path(query_csv))
    _lqeb = _normalize_local_query_embed_backend(local_query_embed_backend)

    metrics_ks = (1, 5, 10)
    search_k = max(int(top_k), max(metrics_ks))
    query_embedding_endpoint, query_embedding_use_grpc = _resolve_query_embedding_endpoint(
        embedding_endpoint=embedding_endpoint,
        embedding_http_endpoint=embedding_http_endpoint,
        embedding_grpc_endpoint=embedding_grpc_endpoint,
    )

    cfg = RecallConfig(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(lancedb_uri),
            "table_name": str(table_name),
            "vector_column_name": str(vector_column_name),
        },
        query_embedder=embedding_model,
        embedding_endpoint=query_embedding_endpoint,
        embedding_api_key=(embedding_api_key or ""),
        embedding_use_grpc=query_embedding_use_grpc,
        top_k=int(search_k),
        ks=metrics_ks,
        local_hf_device=_coerce_endpoint_str(local_hf_device),
        local_hf_cache_dir=(str(local_hf_cache_dir) if local_hf_cache_dir is not None else None),
        local_hf_batch_size=int(local_hf_batch_size),
        local_query_embed_backend=_lqeb,
    )

    df_query, gold, raw_hits, retrieved_keys, metrics = retrieve_and_score(
        query_csv=query_csv,
        cfg=cfg,
        limit=limit,
    )

    if print_hits:
        # Pretty-print top-k results per query.
        for q, g, hits in zip(
            df_query["query"].astype(str).tolist(),
            gold,
            raw_hits,
        ):
            flat = _extract_hits(hits)
            console.print(f"[bold cyan]Query[/bold cyan] {q}")
            console.print(f"[bold]Gold[/bold]  {g}")
            console.print(f"[bold]Hits[/bold]  {flat[: int(top_k)]}")
            console.print("")

    console.print("[bold green]Recall metrics[/bold green]")
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
