"""
Batch ingestion pipeline with optional recall evaluation.
Run with: uv run python -m retriever.examples.batch_pipeline <input-dir>
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import lancedb
import ray
import typer
from retriever import create_ingestor
from retriever.recall.core import RecallConfig, retrieve_and_score

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


def _gold_to_doc_page(golden_key: str) -> tuple[str, str]:
    s = str(golden_key)
    if "_" not in s:
        return s, ""
    doc, page = s.rsplit("_", 1)
    return doc, page


def _is_hit_at_k(golden_key: str, retrieved_keys: list[str], k: int) -> bool:
    doc, page = _gold_to_doc_page(golden_key)
    specific_page = f"{doc}_{page}"
    entire_document = f"{doc}_-1"
    top = (retrieved_keys or [])[: int(k)]
    return (specific_page in top) or (entire_document in top)


def _hit_key_and_distance(hit: dict) -> tuple[str | None, float | None]:
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
    dist = float(hit.get("_distance")) if "_distance" in hit else None
    return key, dist


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDFs to ingest.",
        path_type=Path,
        exists=True,
    ),
    ray_address: Optional[str] = typer.Option(
        None,
        "--ray-address",
        help="URL or address of a running Ray cluster (e.g. 'auto' or 'ray://host:10001'). Omit for in-process Ray.",
    ),
    start_ray: bool = typer.Option(
        False,
        "--start-ray",
        help="Start a Ray head node (ray start --head) and connect to it. Dashboard at http://127.0.0.1:8265. Ignores --ray-address.",
    ),
    query_csv: Path = typer.Option(
        "bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help="Path to query CSV for recall evaluation. Default: bo767_query_gt.csv (current directory). Recall is skipped if the file does not exist.",
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help="Do not print per-query retrieval details (query, gold, hits). Only the missed-gold summary and recall metrics are printed.",
    ),
) -> None:
    os.environ.setdefault("NEMOTRON_OCR_MODEL_DIR", str(Path.cwd() / "nemotron-ocr-v1"))

    # Resolve Ray: start a head node, connect to given address, or run in-process
    if start_ray:
        subprocess.run(["ray", "start", "--head"], check=True, env=os.environ)
        ray_address = "auto"
    # else: use ray_address as-is (None → in-process, or URL to existing cluster)

    input_dir = Path(input_dir)
    pdf_glob = str(input_dir / "*.pdf")

    ingestor = create_ingestor(run_mode="batch", ray_address=ray_address)
    ingestor = (
        ingestor.files(pdf_glob)
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=False,
        )
        .embed(model_name="nemo_retriever_v1")
        .vdb_upload(lancedb_uri=LANCEDB_URI, table_name=LANCEDB_TABLE, overwrite=True, create_index=True)
    )

    print("Running extraction...")
    ingestor.ingest()
    print("Extraction complete.")

    ray.shutdown()

    # ---------------------------------------------------------------------------
    # Recall calculation (optional)
    # ---------------------------------------------------------------------------
    query_csv = Path(query_csv)
    if not query_csv.exists():
        print(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
        return

    db = lancedb.connect(f"./{LANCEDB_URI}")
    table = db.open_table(LANCEDB_TABLE)
    unique_basenames = table.to_pandas()["pdf_basename"].unique()
    print(f"Unique basenames: {unique_basenames}")

    cfg = RecallConfig(
        lancedb_uri=str(LANCEDB_URI),
        lancedb_table=str(LANCEDB_TABLE),
        embedding_model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        top_k=10,
        ks=(1, 5, 10),
    )

    _df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)

    if not no_recall_details:
        print("\nPer-query retrieval details:")
    missed_gold: list[tuple[str, str]] = []
    for i, (q, g, hits) in enumerate(
        zip(
            _df_query["query"].astype(str).tolist(),
            _gold,
            _raw_hits,
        )
    ):
        doc, page = _gold_to_doc_page(g)

        scored_hits: list[tuple[str, float | None]] = []
        for h in hits:
            key, dist = _hit_key_and_distance(h)
            if key:
                scored_hits.append((key, dist))

        top_keys = [k for (k, _d) in scored_hits]
        hit = _is_hit_at_k(g, top_keys, cfg.top_k)

        if not no_recall_details:
            print(f"\nQuery {i}: {q}")
            print(f"  Gold: {g}  (file: {doc}.pdf, page: {page})")
            print(f"  Hit@{cfg.top_k}: {hit}")
            print("  Top hits:")
            if not scored_hits:
                print("    (no hits)")
            else:
                for rank, (key, dist) in enumerate(scored_hits[: int(cfg.top_k)], start=1):
                    if dist is None:
                        print(f"    {rank:02d}. {key}")
                    else:
                        print(f"    {rank:02d}. {key}  distance={dist:.6f}")

        if not hit:
            missed_gold.append((f"{doc}.pdf", str(page)))

    missed_unique = sorted(set(missed_gold), key=lambda x: (x[0], x[1]))
    print("\nMissed gold (unique pdf/page):")
    if not missed_unique:
        print("  (none)")
    else:
        for pdf, page in missed_unique:
            print(f"  {pdf} page {page}")
    print(f"\nTotal missed: {len(missed_unique)} / {len(_gold)}")

    print("\nRecall metrics (matching retriever.recall.core):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    app()
