from retriever import create_ingestor

ingestor = create_ingestor(run_mode="batch")

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
    # .embed(model_name="nemo_retriever_v1")
    # .vdb_upload(lancedb_uri=LANCEDB_URI, table_name=LANCEDB_TABLE, overwrite=False, create_index=True)
    # .save_to_disk(output_directory="/home/jdyer/datasets/jd5_results_inprocess")
)

print("Running extraction...")
result = ingestor.ingest()
breakpoint()
print("Extraction complete.")