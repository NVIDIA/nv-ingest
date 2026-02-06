from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .core import RecallConfig, retrieve_and_score

app = typer.Typer(help="Embed query CSV rows, search LanceDB, print hits, and compute recall@k.")


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
    embedding_endpoint: str = typer.Option("http://embedding:8000/v1", "--embedding-endpoint", help="Embedding endpoint."),
    embedding_model: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embedding-model",
        help="Embedding model name.",
    ),
    embedding_api_key: Optional[str] = typer.Option(None, "--embedding-api-key", help="Embedding API key (optional)."),
) -> None:
    """
    Reads a query CSV, embeds each query, searches LanceDB, prints top-k results, and prints recall@1/@5/@10.

    Note: recall@10 requires retrieving at least 10 results per query; we will query LanceDB with
    search_k = max(top_k, 10) but only print the first `--top-k`.
    """
    query_csv = _resolve_query_csv(Path(query_csv))

    metrics_ks = (1, 5, 10)
    search_k = max(int(top_k), max(metrics_ks))
    cfg = RecallConfig(
        lancedb_uri=str(lancedb_uri),
        lancedb_table=str(table_name),
        embedding_endpoint=str(embedding_endpoint),
        embedding_model=str(embedding_model),
        embedding_api_key=(embedding_api_key or ""),
        top_k=int(search_k),
        ks=metrics_ks,
    )

    df_query, gold, raw_hits, retrieved_keys, metrics = retrieve_and_score(
        query_csv,
        cfg=cfg,
        limit=limit,
        vector_column_name=str(vector_column_name),
    )

    # Print per-query top-k
    queries = df_query["query"].astype(str).tolist()
    for i, (q, g, hits, keys) in enumerate(zip(queries, gold, raw_hits, retrieved_keys)):
        typer.echo("")
        typer.echo(f"[{i}] query={q}")
        typer.echo(f"  gold={g}")
        shown = keys[: int(top_k)]
        if not shown:
            typer.echo("  hits=(none)")
            continue
        typer.echo(f"  hits_top{top_k}=")
        for rank, (k, h) in enumerate(zip(shown, hits[: int(top_k)]), start=1):
            dist = h.get("_distance")
            dist_s = f"{float(dist):.6f}" if dist is not None else "n/a"
            typer.echo(f"    {rank:02d}. {k}  distance={dist_s}")

    # Summary metrics
    typer.echo("")
    typer.echo(
        " ".join(
            [
                f"queries={len(df_query)}",
                f"lancedb_uri={lancedb_uri}",
                f"table={table_name}",
                f"top_k_print={top_k}",
                f"top_k_search={search_k}",
                f"recall@1={metrics['recall@1']:.4f}",
                f"recall@5={metrics['recall@5']:.4f}",
                f"recall@10={metrics['recall@10']:.4f}",
            ]
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

