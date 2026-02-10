from retriever import create_ingestor

ingestor = create_ingestor(run_mode="inprocess")

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"

ingestor = (ingestor
    .files("/home/jdyer/datasets/jd1/*.pdf")
    .extract(
        method="pdfium",
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
    .embed(model_name="nemo_retriever_v1")
    .vdb_upload(lancedb_uri=LANCEDB_URI, table_name=LANCEDB_TABLE, overwrite=True)
    .save_to_disk(output_directory="/home/jdyer/datasets/jd1_results_inprocess")
)

print("Running extraction...")
ingestor.ingest(show_progress=True)
print("Extraction complete.")

# Now lets perform recall testing
from pathlib import Path
from retriever.recall.core import RecallConfig, retrieve_and_score

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
print("Recall metrics (matching retriever.recall.core):")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
