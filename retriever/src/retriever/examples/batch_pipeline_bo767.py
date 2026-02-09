from retriever import create_ingestor

ingestor = create_ingestor(run_mode="batch")

ingestor = (ingestor
    .files("/home/jdyer/datasets/bo767/*.pdf")
    .extract(
        method="pdfium",
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
)

assert len(ingestor._input_documents) == 767, f"Expected 767 documents, got {len(ingestor._input_documents)}"