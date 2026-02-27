from nemo_retriever import ingestor, retriever

ingestor = ingestor()
retriever = retriever()

ingestor.files("data/*.pdf")
ingestor.extract(method="pdfium")
ingestor.embed(model_name="nemo_retriever_v1")
ingestor.vdb_upload(lancedb_uri="lancedb", table_name="nv-ingest")

retriever.query("What is the main idea of the document?")
