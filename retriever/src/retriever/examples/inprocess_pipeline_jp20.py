from retriever import create_ingestor
import json
from pathlib import Path
from retriever.recall.core import RecallConfig, retrieve_and_score

ingestor = create_ingestor(run_mode="inprocess")

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"

ingestor = (ingestor
    .files("/home/jdyer/datasets/jp20/*.pdf")
    .extract(
        method="pdfium",
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
    .embed(model_name="nemo_retriever_v1")
    .vdb_upload(lancedb_uri=LANCEDB_URI, table_name=LANCEDB_TABLE, overwrite=False, create_index=True)
    .save_to_disk(output_directory="/home/jdyer/datasets/jp20_results_inprocess")
)

print("Running extraction...")
ingestor.ingest(show_progress=True)
print("Extraction complete.")

import lancedb
db = lancedb.connect("./lancedb")
table = db.open_table("nv-ingest")
unique_basenames = table.to_pandas()["pdf_basename"].unique()
print(f"Unique basenames: {unique_basenames}")

query_csv = Path("jp20_query_gt.csv")
cfg = RecallConfig(
    lancedb_uri=str(LANCEDB_URI),
    lancedb_table=str(LANCEDB_TABLE),
    # Recall script falls back to local HF embeddings when endpoints are unset.
    embedding_model="nvidia/llama-3.2-nv-embedqa-1b-v2",
    top_k=10,
    ks=(1, 5, 10),
)

_df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)


def _gold_to_doc_page(golden_key: str) -> tuple[str, str]:
    # Keys are expected to look like "<pdf_stem>_<page>" where pdf_stem may contain underscores.
    s = str(golden_key)
    if "_" not in s:
        return s, ""
    doc, page = s.rsplit("_", 1)
    return doc, page


def _is_hit_at_k(golden_key: str, retrieved_keys: list[str], k: int) -> bool:
    """
    Match retriever.recall.core._recall_at_k semantics:
      - exact page hit: "<doc>_<page>"
      - "entire document" hit: "<doc>_-1"
    """
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


print("\nPer-query retrieval details:")
missed_gold: list[tuple[str, str]] = []  # (pdf filename, page)
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


print("Recall metrics (matching retriever.recall.core):")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")