import logging

from datetime import timedelta
from functools import partial
from urllib.parse import urlparse

import lancedb
import pyarrow as pa

from nv_ingest_client.util.transport import infer_microservice
from nv_ingest_client.util.vdb.adt_vdb import VDB
from nv_ingest_client.util.vdb.milvus import nv_rerank

logger = logging.getLogger(__name__)


def _get_text_for_element(element):
    """
    Extract searchable text from an element based on document_type.

    Matches Milvus behavior: for images, use caption/OCR text instead of raw content.
    This prevents base64-encoded images from being stored in the text field.
    """
    doc_type = element.get("document_type")
    metadata = element.get("metadata", {})

    if doc_type == "text":
        return metadata.get("content")
    elif doc_type == "structured":
        # Tables, charts, infographics
        table_meta = metadata.get("table_metadata", {})
        return table_meta.get("table_content")
    elif doc_type == "image":
        # Use caption/OCR text, not raw base64 image data
        image_meta = metadata.get("image_metadata", {})
        content_meta = metadata.get("content_metadata", {})
        if content_meta.get("subtype") == "page_image":
            return image_meta.get("text")
        else:
            return image_meta.get("caption")
    elif doc_type == "audio":
        audio_meta = metadata.get("audio_metadata", {})
        return audio_meta.get("audio_transcript")
    else:
        # Fallback for unknown types
        return metadata.get("content")


def create_lancedb_results(results):
    """
    Transform NV-Ingest pipeline results into LanceDB ingestible rows.

    Extracts appropriate text based on document_type rather than storing
    raw content (which may include base64 images).

    Parameters
    ----------
    results : list
        Pipeline output results.
    """
    lancedb_rows = []

    for result in results:
        for element in result:
            metadata = element.get("metadata", {})
            doc_type = element.get("document_type")

            # Check if embedding exists
            embedding = metadata.get("embedding")
            if embedding is None:
                continue

            content_meta = metadata.get("content_metadata", {})

            # Extract appropriate text based on document type
            text = _get_text_for_element(element)

            if not text:
                source_name = metadata.get("source_metadata", {}).get("source_name", "unknown")
                pg_num = content_meta.get("page_number")
                logger.debug(f"No text found for entity: {source_name} page: {pg_num} type: {doc_type}")
                continue

            lancedb_rows.append(
                {
                    "vector": embedding,
                    "text": text,
                    "metadata": str(content_meta.get("page_number", "")),
                    "source": metadata.get("source_metadata", {}).get("source_id", ""),
                }
            )

    return lancedb_rows


class LanceDB(VDB):
    """LanceDB operator implementing the VDB interface."""

    def __init__(
        self,
        uri=None,
        overwrite=True,
        table_name="nv-ingest",
        index_type="IVF_HNSW_SQ",
        metric="l2",
        num_partitions=16,
        num_sub_vectors=256,
        **kwargs,
    ):
        self.uri = uri or "lancedb"
        self.overwrite = overwrite
        self.table_name = table_name
        self.index_type = index_type
        self.metric = metric
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        super().__init__(**kwargs)

    def create_index(self, records=None, table_name="nv-ingest", **kwargs):
        """Create a LanceDB table and populate it with transformed records."""
        db = lancedb.connect(uri=self.uri)
        results = create_lancedb_results(records)
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), 2048)),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("source", pa.string()),
            ]
        )
        table = db.create_table(
            table_name, data=results, schema=schema, mode="overwrite" if self.overwrite else "append"
        )
        return table

    def write_to_index(
        self,
        records,
        table=None,
        index_type="IVF_HNSW_SQ",
        metric="l2",
        num_partitions=16,
        num_sub_vectors=256,
        **kwargs,
    ):
        """Create an index on the LanceDB table and wait for it to become ready."""
        table.create_index(
            index_type=index_type,
            metric=metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="vector",
        )
        for index_stub in table.list_indices():
            table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))

    def run(self, records):
        """Orchestrate index creation and data ingestion."""
        table = self.create_index(records=records, table_name=self.table_name)
        self.write_to_index(
            records,
            table=table,
            index_type=self.index_type,
            metric=self.metric,
            num_partitions=self.num_partitions,
            num_sub_vectors=self.num_sub_vectors,
        )
        return records

    def retrieval(self, queries, **kwargs):
        """
        Run LanceDB retrieval by delegating to lancedb_retrieval.

        Uses instance defaults for table_path (self.uri) and table_name (self.table_name)
        unless overridden via kwargs.
        """
        return lancedb_retrieval(
            queries,
            table_path=kwargs.pop("table_path", self.uri),
            table_name=kwargs.pop("table_name", self.table_name),
            **kwargs,
        )


def lancedb_retrieval(
    queries,
    table_path: str = None,
    table_name: str = "nv-ingest",
    table=None,
    embedding_endpoint: str = "http://localhost:8012/v1",
    nvidia_api_key: str = None,
    model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
    top_k: int = 10,
    refine_factor: int = 50,
    n_probe: int = 64,
    nv_ranker: bool = False,
    nv_ranker_endpoint: str = None,
    nv_ranker_model_name: str = None,
    nv_ranker_top_k: int = 50,
    result_fields=None,
    **kwargs,
):
    """
    Standalone LanceDB retrieval function.

    This is the primary interface for LanceDB vector search. It embeds queries using the
    specified embedding model and performs vector similarity search against
    a LanceDB table.

    Parameters
    ----------
    queries : list[str]
        Text queries to search for.
    table_path : str, optional
        Path to the LanceDB database directory (the `uri` used during ingestion).
        Required if `table` is not provided.
    table_name : str, optional
        Name of the table within the LanceDB database (default: "nv-ingest").
    table : object, optional
        Pre-opened LanceDB table object. If provided, table_path and table_name
        are ignored. Useful for reusing connections or testing.
    embedding_endpoint : str, optional
        URL of the embedding microservice (default: "http://localhost:8012/v1").
    nvidia_api_key : str, optional
        NVIDIA API key for authentication. If None, no auth is used.
    model_name : str, optional
        Name of the embedding model (default: "nvidia/llama-3.2-nv-embedqa-1b-v2").
    top_k : int, optional
        Number of results to return per query (default: 10).
    refine_factor : int, optional
        LanceDB search refine factor for accuracy (default: 50).
    n_probe : int, optional
        Number of partitions to probe during search (default: 64).
    nv_ranker : bool, optional
        Whether to apply NV reranker after retrieval (default: False).
    nv_ranker_endpoint : str, optional
        URL of the reranker microservice.
    nv_ranker_model_name : str, optional
        Name of the reranker model.
    nv_ranker_top_k : int, optional
        Number of candidates to fetch before reranking (default: 50).
    result_fields : list, optional
        List of field names to retrieve from each hit document (default:
        ["text", "metadata", "source"]).
    **kwargs
        Additional keyword arguments (ignored, for API compatibility).

    Returns
    -------
    list[list[dict]]
        For each query, a list of result dicts formatted to match Milvus output
        structure for recall scoring compatibility.
    """
    if table is None:
        if table_path is None:
            raise ValueError("Either table or table_path must be provided")
        db = lancedb.connect(uri=table_path)
        table = db.open_table(table_name)

    if result_fields is None:
        result_fields = ["text", "metadata", "source"]

    embed_model = partial(
        infer_microservice,
        model_name=model_name,
        embedding_endpoint=embedding_endpoint,
        nvidia_api_key=nvidia_api_key,
        input_type="query",
        output_names=["embeddings"],
        grpc=not ("http" in urlparse(embedding_endpoint).scheme),
    )

    search_top_k = nv_ranker_top_k if nv_ranker else top_k

    results = []
    query_embeddings = embed_model(queries)
    for query_embed in query_embeddings:
        search_results = (
            table.search([query_embed], vector_column_name="vector")
            .select(result_fields)
            .limit(search_top_k)
            .refine_factor(refine_factor)
            .nprobes(n_probe)
            .to_list()
        )
        formatted = []
        for r in search_results:
            formatted.append(
                {
                    "entity": {
                        "source": {"source_id": r.get("source")},
                        "content_metadata": {"page_number": r.get("metadata")},
                        "text": r.get("text"),
                    }
                }
            )
        results.append(formatted)

    if nv_ranker:
        rerank_results = []
        num_queries = len(queries)
        for idx, (query, candidates) in enumerate(zip(queries, results), 1):
            if num_queries > 1:
                logger.info(f"Reranking query {idx}/{num_queries}")
            rerank_results.append(
                nv_rerank(
                    query,
                    candidates,
                    reranker_endpoint=nv_ranker_endpoint,
                    model_name=nv_ranker_model_name,
                    topk=top_k,
                )
            )
        results = rerank_results

    return results
