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


def create_lancedb_results(results):
    """Transform NV-Ingest pipeline results into LanceDB ingestible rows."""
    old_results = [res["metadata"] for result in results for res in result]
    lancedb_rows = []
    for result in old_results:
        if result["embedding"] is None:
            continue
        lancedb_rows.append(
            {
                "vector": result["embedding"],
                "text": result["content"],
                "metadata": result["content_metadata"]["page_number"],
                "source": result["source_metadata"]["source_id"],
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

    def retrieval(
        self,
        queries,
        table=None,
        table_path=None,
        table_name=None,
        embedding_endpoint: str = "http://localhost:8012/v1",
        nvidia_api_key: str = None,
        model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        top_k: int = 10,
        refine_factor: int = 50,
        n_probe: int = 64,
        nv_ranker: bool = False,
        nv_ranker_endpoint: str = None,
        nv_ranker_model_name: str = None,
        nv_ranker_top_k: int = 100,
        result_fields=None,
        **kwargs,
    ):
        """Run LanceDB retrieval and return results in Milvus-style format."""
        table_path = table_path or self.uri
        table_name = table_name or self.table_name
        if result_fields is None:
            result_fields = ["text", "metadata", "source", "_distance"]

        if table is None:
            db = lancedb.connect(uri=table_path)
            table = db.open_table(table_name)

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
            for query, candidates in zip(queries, results):
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


def retrieval(
    queries,
    table_path: str,
    table_name: str = "nv-ingest",
    embedding_endpoint: str = "http://localhost:8012/v1",
    nvidia_api_key: str = None,
    model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
    top_k: int = 10,
    refine_factor: int = 50,
    n_probe: int = 64,
    nv_ranker: bool = False,
    nv_ranker_endpoint: str = None,
    nv_ranker_model_name: str = None,
    nv_ranker_top_k: int = 100,
    **kwargs,
):
    """
    LanceDB retrieval function.

    Parameters
    ----------
    queries : list[str]
        Text queries to search for.
    table_path : str
        Path to the LanceDB database directory.
    table_name : str, optional
        Name of the table within the LanceDB database.
    embedding_endpoint : str, optional
        URL of the embedding microservice.
    nvidia_api_key : str, optional
        NVIDIA API key for authentication.
    model_name : str, optional
        Name of the embedding model.
    top_k : int, optional
        Number of results to return per query.
    refine_factor : int, optional
        LanceDB search refine factor for accuracy.
    n_probe : int, optional
        Number of partitions to probe during search.
    nv_ranker : bool, optional
        Whether to apply NV reranker after retrieval.
    nv_ranker_endpoint : str, optional
        URL of the reranker microservice.
    nv_ranker_model_name : str, optional
        Name of the reranker model.
    nv_ranker_top_k : int, optional
        Number of candidates to fetch before reranking.

    Returns
    -------
    list[list[dict]]
        For each query, a list of result dicts formatted to match Milvus output structure.
    """
    db = lancedb.connect(uri=table_path)
    table = db.open_table(table_name)

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
            .select(["text", "metadata", "source", "_distance"])
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
                        "source": {"source_id": r["source"]},
                        "content_metadata": {"page_number": r["metadata"]},
                        "text": r["text"],
                    }
                }
            )
        results.append(formatted)

    if nv_ranker:
        rerank_results = []
        for query, candidates in zip(queries, results):
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
