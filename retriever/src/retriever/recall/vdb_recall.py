from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from nv_ingest_client.util.vdb.lancedb import LanceDB
import typer
import pandas as pd

from .core import RecallConfig, _normalize_query_df

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
    embedding_endpoint: str = typer.Option(
        "http://embedding:8000/v1", "--embedding-endpoint", help="Embedding endpoint."
    ),
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

    # df_query, gold, raw_hits, retrieved_keys, metrics = retrieve_and_score(
    #     query_csv,
    #     cfg=cfg,
    #     limit=limit,
    #     vector_column_name=str(vector_column_name),
    # )
    df_query = _normalize_query_df(pd.read_csv(query_csv))
    queries = df_query["query"].astype(str).tolist()
    gold = df_query["golden_answer"].astype(str).tolist()

    df_query = pd.read_csv(query_csv)
    queries = df_query["query"].astype(str).tolist()

    lancedb = LanceDB(uri=cfg.lancedb_uri, table_name=cfg.lancedb_table)
    results = lancedb.retrieval(queries)

    recall_1_hits = 0
    recall_5_hits = 0
    recall_10_hits = 0

    for q, g, result in zip(queries, gold, results):
        hits = _extract_hits(result)
        print(len(hits))
        gold_doc = g.split("_")[0]
        gold_page = g.split("_")[1]
        print(q, g, hits)

        for i, hit in enumerate(hits):
            if hit["pdf_basename"] == gold_doc and (
                (hit["page_number"] == -1) or (hit["page_number"] == int(gold_page))
            ):
                if i < 1:
                    recall_10_hits += 1
                    recall_5_hits += 1
                    recall_1_hits += 1
                    continue
                if i < 5:
                    recall_10_hits += 1
                    recall_5_hits += 1
                    continue
                if i < 10:
                    recall_10_hits += 1
                    continue

    print(f"Recall @1: {recall_1_hits / len(queries)}")
    print(f"Recall @5: {recall_5_hits / len(queries)}")
    print(f"Recall @10: {recall_10_hits / len(queries)}")

    # # Summary metrics
    # typer.echo("")
    # typer.echo(
    #     " ".join(
    #         [
    #             f"queries={len(df_query)}",
    #             f"lancedb_uri={lancedb_uri}",
    #             f"table={table_name}",
    #             f"top_k_print={top_k}",
    #             f"top_k_search={search_k}",
    #             f"recall@1={metrics['recall@1']:.4f}",
    #             f"recall@5={metrics['recall@5']:.4f}",
    #             f"recall@10={metrics['recall@10']:.4f}",
    #         ]
    #     )
    # )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
