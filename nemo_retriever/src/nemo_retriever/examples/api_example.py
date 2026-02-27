from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.retriever import retriever

ingestor = create_ingestor(run_mode="batch")
retriever = retriever()

ingestor.files("data/*.pdf")
ingestor.extract(method="pdfium")
ingestor.embed(model_name="nemo_retriever_v1")
ingestor.vdb_upload(lancedb_uri="lancedb", table_name="nv-ingest")

hits = retriever.query("Who was the assistant secretary of the city of Houston, Texas in the year, 2015?")
print(hits)


from nemo_retriever.retriever import retriever

retriever = retriever()  # Class to hold HF model state for embedder

hits = retriever.query("Who was the assistant secretary of the city of Houston, Texas in the year, 2015?")
print(hits)
