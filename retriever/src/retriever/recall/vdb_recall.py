from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
import pandas as pd
from rich.console import Console

from .core import RecallConfig, evaluate_recall, retrieve_and_score, _normalize_query_df

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
                "pdf_page": res.get("entity", {}).get("content_metadata", {}).get("page_number"),
                "pdf_basename": Path(res.get("entity", {}).get("source", {}).get("source_id")).name.split(".")[0],
                "page_number": res.get("entity", {}).get("content_metadata", {}).get("page_number"),
                "source_id": Path(res.get("entity", {}).get("source", {}).get("source_id")).name.split(".")[0],
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


def _resolve_endpoints(
    *,
    embedding_endpoint: Optional[str],
    embedding_http_endpoint: Optional[str],
    embedding_grpc_endpoint: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve endpoint options with precedence:
      1) explicit http/grpc options
      2) single --embedding-endpoint (auto-routed by scheme)
    """
    http_ep = _coerce_endpoint_str(embedding_http_endpoint)
    grpc_ep = _coerce_endpoint_str(embedding_grpc_endpoint)
    single = _coerce_endpoint_str(embedding_endpoint)

    if http_ep or grpc_ep:
        return http_ep, grpc_ep
    if single:
        if single.lower().startswith("http"):
            return single, None
        return None, single
    return None, None


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
    table_name: str = typer.Option("nv-ingest", "--table-name", help="LanceDB table name."),
    vector_column_name: str = typer.Option("vector", "--vector-column", help="Vector column name in the table."),
    embedding_endpoint: Optional[str] = typer.Option(
        None,
        "--embedding-endpoint",
        help=(
            "Embedding endpoint (http(s) URL or host:port for gRPC). "
            "If omitted, you may specify --embedding-http-endpoint/--embedding-grpc-endpoint instead; "
            "if no endpoints are provided, stage7 falls back to local HF embeddings."
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
    embedding_model: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embedding-model",
        help="Embedding model name.",
    ),
    embedding_api_key: Optional[str] = typer.Option(None, "--embedding-api-key", help="Embedding API key (optional)."),
    local_hf_device: Optional[str] = typer.Option(
        None,
        "--local-hf-device",
        help="Device for local HF embeddings when endpoints are missing (e.g. 'cuda', 'cpu', 'cuda:0').",
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
    print_hits: bool = typer.Option(True, "--print-hits/--no-print-hits", help="Print top-k hits per query."),
) -> None:
    """
    Reads a query CSV, embeds each query, searches LanceDB, prints top-k results, and prints recall@1/@5/@10.

    Note: recall@10 requires retrieving at least 10 results per query; we will query LanceDB with
    search_k = max(top_k, 10) but only print the first `--top-k`.
    """
    query_csv = _resolve_query_csv(Path(query_csv))

    metrics_ks = (1, 5, 10)
    search_k = max(int(top_k), max(metrics_ks))

    http_ep, grpc_ep = _resolve_endpoints(
        embedding_endpoint=embedding_endpoint,
        embedding_http_endpoint=embedding_http_endpoint,
        embedding_grpc_endpoint=embedding_grpc_endpoint,
    )
    cfg = RecallConfig(
        lancedb_uri=str(lancedb_uri),
        lancedb_table=str(table_name),
        embedding_http_endpoint=http_ep,
        embedding_grpc_endpoint=grpc_ep,
        embedding_endpoint=_coerce_endpoint_str(embedding_endpoint),
        embedding_model=str(embedding_model),
        embedding_api_key=(embedding_api_key or ""),
        top_k=int(search_k),
        ks=metrics_ks,
        local_hf_device=_coerce_endpoint_str(local_hf_device),
        local_hf_cache_dir=(str(local_hf_cache_dir) if local_hf_cache_dir is not None else None),
        local_hf_batch_size=int(local_hf_batch_size),
    )

    df_query, gold, raw_hits, retrieved_keys, metrics = retrieve_and_score(
        query_csv=query_csv,
        cfg=cfg,
        limit=limit,
        vector_column_name=str(vector_column_name),
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
