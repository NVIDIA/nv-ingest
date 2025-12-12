import logging


from nv_ingest_client.util.vdb.adt_vdb import VDB
from datetime import timedelta
from functools import partial
from urllib.parse import urlparse
from nv_ingest_client.util.transport import infer_microservice
import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)


def create_lancedb_results(results):
    """Transform NV-Ingest pipeline results into LanceDB ingestible rows.

    The NV-Ingest pipeline provides nested lists of record dictionaries. This
    helper extracts the inner `metadata` dict for each record, filters out
    entries without an embedding, and returns a list of dictionaries with the
    exact fields expected by the LanceDB table schema used in
    `LanceDB.create_index`.

    Parameters
    ----------
    results : list
        Nested list-of-lists containing record dicts in the NV-Ingest format.

    Returns
    -------
    list
        List of dictionaries with keys: `vector` (embedding list), `text`
        (string content), `metadata` (page number) and `source` (source id).

    Notes
    -----
    - The function expects each inner record to have a `metadata` mapping
        containing `embedding`, `content`, `content_metadata.page_number`, and
        `source_metadata.source_id`.
    - Records with `embedding is None` are skipped.
    """
    old_results = [res["metadata"] for result in results for res in result]
    results = []
    for result in old_results:
        if result["embedding"] is None:
            continue
        results.append(
            {
                "vector": result["embedding"],
                "text": result["content"],
                "metadata": result["content_metadata"]["page_number"],
                "source": result["source_metadata"]["source_id"],
            }
        )
    return results


class LanceDB(VDB):
    """LanceDB operator implementing the VDB interface.

    This class adapts NV-Ingest records to LanceDB, providing index creation,
    ingestion, and retrieval hooks. The implementation is intentionally small
    and focuses on the example configuration used in NV-Ingest evaluation
    scripts.
    """

    def __init__(
        self,
        uri=None,
        overwrite=True,
        table_name="nv-ingest",
        index_type="IVF_HNSW_SQ",
        metric="l2",
        num_partitions=16,
        num_sub_vectors=256,
        **kwargs
    ):
        """Initialize the LanceDB VDB operator.

        Parameters
        ----------
        uri: str, optional
            LanceDB connection URI (default is "lancedb" for local file-based
            storage).
        overwrite : bool, optional
            If True, existing tables will be overwritten during index creation.
            If False, new data will be appended to existing tables.
        table_name : str, optional
            Name of the LanceDB table to create/use (default is "nv-ingest").
        index_type : str, optional
            Type of vector index to create (default is "IVF_HNSW_SQ").
        metric : str, optional
            Distance metric for the vector index (default is "l2").
        num_partitions : int, optional
            Number of partitions for the vector index (default is 16).
        num_sub_vectors : int, optional
            Number of sub-vectors for the vector index (default is 256).
        **kwargs : dict
            Forwarded configuration options. This implementation does not
            actively consume specific keys, but passing parameters such as
            `uri`, `index_name`, or security options is supported by the
            interface pattern and may be used by future enhancements.
        """
        self.uri = uri or "lancedb"
        self.overwrite = overwrite
        self.table_name = table_name
        self.index_type = index_type
        self.metric = metric
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        super().__init__(**kwargs)

    def create_index(self, records=None, table_name="nv-ingest", **kwargs):
        """Create a LanceDB table and populate it with transformed records.

        This method connects to LanceDB, transforms NV-Ingest records using
        `create_lancedb_results`, builds a PyArrow schema that matches the
        expected table layout, and creates/overwrites a table named `bo`.

        Parameters
        ----------
        records : list, optional
            NV-Ingest records in nested list format (the same structure passed
            to `run`). If ``None``, an empty table will be created.

        table_name : str, optional
            Name of the LanceDB table to create (default is "nv-ingest").

        Returns
        -------
        table
            The LanceDB table object returned by `db.create_table`.
        """
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
        **kwargs
    ):
        """Create an index on the LanceDB table and wait for it to become ready.

        This function calls `table.create_index` with an IVF+HNSW+SQ index
        configuration used in NV-Ingest benchmarks. After requesting index
        construction it lists available indices and waits for each one to
        reach a ready state using `table.wait_for_index`.

        Parameters
        ----------
        records : list
            The original records being indexed (not used directly in this
            implementation but kept in the signature for consistency).
        table : object
            LanceDB table object returned by `create_index`.
        """
        table.create_index(
            index_type=index_type,
            metric=metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            # accelerator="cuda",
            vector_column_name="vector",
        )
        for index_stub in table.list_indices():
            table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))

    def retrieval(
        self,
        queries,
        table=None,
        embedding_endpoint="http://localhost:8012/v1",
        nvidia_api_key=None,
        model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
        result_fields=["text", "metadata", "source"],
        top_k=10,
        **kwargs
    ):
        """Run similarity search for a list of text queries.

        This method converts textual queries to embeddings by calling the
        transport helper `infer_microservice` (configured to use an NVIDIA
        embedding model in the example) and performs a vector search against
        the LanceDB `table`.

        Parameters
        ----------
        queries : list[str]
            Text queries to be embedded and searched.
        table : object
            LanceDB table object with a built vector index.
        embedding_endpoint : str, optional
            URL of the embedding microservice (default is
            "http://localhost:8012/v1").
        nvidia_api_key : str, optional
            NVIDIA API key for authentication with the embedding service. If
            ``None``, no authentication is used.
        model_name : str, optional
            Name of the embedding model to use (default is
            "nvidia/llama-3.2-nv-embedqa-1b-v2").
        result_fields : list, optional
            List of field names to retrieve from each hit document (default is
            `["text", "metadata", "source"]`).
        top_k : int, optional
            Number of top results to return per query (default is 10).

        Returns
        -------
        list[list[dict]]
            For each input query, a list of hit documents (each document is a
            dict with fields such as `text`, `metadata`, and `source`). The
            example limits each query to 20 results.
        """
        embed_model = partial(
            infer_microservice,
            model_name=model_name,
            embedding_endpoint=embedding_endpoint,
            nvidia_api_key=nvidia_api_key,
            input_type="query",
            output_names=["embeddings"],
            grpc=not ("http" in urlparse(embedding_endpoint).scheme),
        )
        results = []
        query_embeddings = embed_model(queries)
        for query_embed in query_embeddings:
            results.append(
                table.search([query_embed], vector_column_name="vector").select(result_fields).limit(top_k).to_list()
            )
        return results

    def run(self, records):
        """Orchestrate index creation and data ingestion.

        The `run` method is the public entry point used by NV-Ingest pipeline
        tasks. A minimal implementation first ensures the table exists by
        calling `create_index` and then kicks off index construction with
        `write_to_index`.

        Parameters
        ----------
        records : list
            NV-Ingest records to index.

        Returns
        -------
        list
            The original `records` list is returned unchanged to make the
            operator composable in pipelines.
        """
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
