"""
In-process ingestion pipeline (no Ray) with optional recall evaluation.
Run with: uv run python -m retriever.examples.inprocess_pipeline <input-dir>
"""
import json
import os
from pathlib import Path

import lancedb
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
        help="Directory containing PDFs or .txt files to ingest.",
        path_type=Path,
        exists=True,
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input format: 'pdf' or 'txt'. Use 'txt' for a directory of .txt files (tokenizer-based chunking).",
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
    if input_type == "txt":
        pass  # No NEMOTRON_OCR_MODEL_DIR needed for .txt
    else:
        os.environ.setdefault("NEMOTRON_OCR_MODEL_DIR", str(Path.cwd() / "nemotron-ocr-v1"))

    input_dir = Path(input_dir)
    if input_type == "txt":
        glob_pattern = str(input_dir / "*.txt")
        ingestor = create_ingestor(run_mode="inprocess")
        ingestor = (
            ingestor.files(glob_pattern)
            .extract_txt(max_tokens=512, overlap_tokens=0)
            .embed(model_name="nemo_retriever_v1")
            .vdb_upload(lancedb_uri=LANCEDB_URI, table_name=LANCEDB_TABLE, overwrite=False, create_index=True)
        )
    else:
        glob_pattern = str(input_dir / "*.pdf")
        ingestor = create_ingestor(run_mode="inprocess")
        ingestor = (
            ingestor.files(glob_pattern)
            .extract(
                method="pdfium",
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_infographics=False,
            )
            .embed(model_name="nemo_retriever_v1")
            .vdb_upload(lancedb_uri=LANCEDB_URI, table_name=LANCEDB_TABLE, overwrite=False, create_index=True)
        )

    print("Running extraction...")
    ingestor.ingest(show_progress=True)
    print("Extraction complete.")

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
            ext = ".txt" if input_type == "txt" else ".pdf"
            print(f"\nQuery {i}: {q}")
            print(f"  Gold: {g}  (file: {doc}{ext}, page: {page})")
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
            ext = ".txt" if input_type == "txt" else ".pdf"
            missed_gold.append((f"{doc}{ext}", str(page)))

    missed_unique = sorted(set(missed_gold), key=lambda x: (x[0], x[1]))
    print("\nMissed gold (unique doc/page):")
    if not missed_unique:
        print("  (none)")
    else:
        for doc_page, page in missed_unique:
            print(f"  {doc_page} page {page}")
    print(f"\nTotal missed: {len(missed_unique)} / {len(_gold)}")

    print("\nRecall metrics (matching retriever.recall.core):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    app()
