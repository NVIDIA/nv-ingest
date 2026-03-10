from abc import ABC, abstractmethod


"""Abstract Vector Database (VDB) operator API.

This module defines the `VDB` abstract base class which specifies the
interface that custom vector-database operators must implement to integrate
with NV-Ingest.

The implementation details and an example OpenSearch operator are described
in the `examples/building_vdb_operator.ipynb` notebook in this repository, and a
production-ready OpenSearch implementation is available at
`client/src/nv_ingest_client/util/vdb/opensearch.py`.

Design goals:
- Provide a small, well-documented interface that supports common vector
    database operations: index creation, batch ingestion, nearest-neighbor
    retrieval, and a simple `run` orchestration entry-point used by the
    NV-Ingest pipeline.
- Keep the API flexible by accepting `**kwargs` on methods so implementers can
    pass database-specific options without changing the interface.

Typical implementation notes (inferred from the example OpenSearch operator):
- Constructor accepts connection and index configuration parameters such as
    `host`, `port`, `index_name`, `dense_dim` and feature toggles for content
    types (e.g. `enable_text`, `enable_images`).
- `create_index` should be able to create (and optionally recreate) an
    index with appropriate vector settings (k-NN, HNSW/FAISS parameters, etc.).
- `write_to_index` should accept batches of NV-Ingest records, perform
    validation/transformation, and write documents into the database efficiently
    (bulk APIs are recommended).
- `retrieval` should accept a list of textual queries, convert them to
    embeddings (by calling an external embedding service or model), perform a
    vector search (top-k), and return cleaned results (e.g., removing stored
    dense vectors from returned payloads).

"""


class VDB(ABC):
    """Abstract base class for Vector Database operators.

    Subclasses must implement the abstract methods below. The interface is
    intentionally small and uses `**kwargs` to allow operator-specific
    configuration without changing the common API.

    Example (high level):

            class OpenSearch(VDB):
                    def __init__(self, **kwargs):
                            # parse kwargs, initialize client, call super().__init__(**kwargs)
                            ...

                    def create_index(self, **kwargs):
                            # create index, mappings, settings
                            ...

                    def write_to_index(self, records: list, **kwargs):
                            # transform NV-Ingest records and write to database
                            ...

                    def retrieval(self, queries: list, **kwargs):
                            # convert queries to embeddings, k-NN search, format results
                            ...

                    def run(self, records):
                            # orchestrate create_index + write_to_index
                            ...

    Notes on recommended constructor parameters (not enforced by this ABC):
    - host (str): database hostname (default: 'localhost')
    - port (int): database port (default: 9200 for OpenSearch/Elasticsearch)
    - index_name (str): base index name used by the operator
    - dense_dim (int): dimensionality of stored dense embeddings
    - enable_text/enable_images/... (bool): content-type toggles used when
        extracting text from NV-Ingest records before indexing

    The concrete operator may accept additional parameters (username,
    password, ssl options, client-specific flags). Passing these via
    `**kwargs` is the intended pattern.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the VDB operator.

        Implementations should extract configuration values from `kwargs`
        (or use defaults) and initialize any client connections required to
        talk to the target vector database. Implementations are encouraged to
        call `super().__init__(**kwargs)` only if they want the base-class
        behavior of storing kwargs on the instance (the base class itself does
        not require that behavior).

        Parameters (suggested/common):
        - host (str): database host
        - port (int): database port
        - index_name (str): base name for created indices
        - dense_dim (int): embedding vector dimension
        - enable_text (bool): whether text content should be extracted/indexed
        - enable_images (bool), enable_audio (bool), etc.: other toggles

        The constructor should not perform heavy operations (like creating
        indices) unless explicitly desired; prefer leaving that work to
        `create_index` to make the operator easier to test.
        """
        self.__dict__.update(kwargs)

    @abstractmethod
    def create_index(self, **kwargs):
        """Create and configure the index(es) required by this operator.

        Implementations must ensure an appropriate index (or indices) exist
        before data ingestion. For vector indexes this typically means
        creating settings and mappings that enable k-NN/vector search (for
        example, enabling an HNSW/FAISS engine, setting `dimension`, and any
        engine-specific parameters).

        Common keyword arguments (operator-specific):
        - recreate (bool): if True, delete and recreate the index even if it
            already exists (default: False)
        - index_name (str): override the operator's configured index name for
            this call

        Returns:
                implementation-specific result (e.g., a boolean, the created
                index name, or the raw response from the database client).  There
                is no strict requirement here because different DB clients return
                different values; document behavior in concrete implementations.
        """
        pass

    @abstractmethod
    def write_to_index(self, records: list, **kwargs):
        """Write a batch of NV-Ingest records to the vector database.

        This method receives `records` formatted as NV-Ingest provides them
        (commonly a list of record-sets). Implementations are responsible for
        transforming each record into the target database document format,
        validating the presence of embeddings and content, and using the most
        efficient ingestion API available (for example a bulk endpoint).

        Expected behavior:
        - Iterate over the provided `records` (which can be nested lists of
            record dictionaries) and transform each record to the DB document
            structure (fields such as `dense` for the vector, `text` for the
            content, and `metadata` for auxiliary fields are common in the
            repository examples).
        - Skip records missing required fields (for example, missing
            embeddings) and log or report failures as appropriate.
        - Use batching / bulk APIs to reduce overhead when writing large
            volumes of documents.

        Parameters:
        - records (list): NV-Ingest records (see repository examples for
            structure)
        - batch_size (int, optional): how many documents to send per bulk
            request; database-specific implementations can use this hint

        Returns:
                implementation-specific result (e.g., number of documents
                indexed, client response for bulk API). Concrete implementations
                should document exact return values and failure semantics.
        """
        pass

    @abstractmethod
    def retrieval(self, queries: list, **kwargs):
        """Perform similarity search for a list of text queries.

        The typical retrieval flow implemented by operators in this ecosystem
        is:
        1. Convert each textual `query` into a dense embedding using an
             external embedding model or service (the example uses an NVIDIA
             embedding model via `llama_index.embeddings.nvidia.NVIDIAEmbedding`).
        2. Issue a vector (k-NN) search to the database using the generated
             embedding, requesting the top-k (configurable) neighbors.
        3. Post-process results (for example, remove stored dense vectors
             from returned documents to reduce payload size) and return a
             list-of-lists of result documents aligned with the input `queries`.

        Keyword arguments (common):
        - index_name (str): index to search (default: operator's configured
            index_name)
        - top_k (int): number of nearest neighbors to return (default: 10)
        - embedding_endpoint / model_name / nvidia_api_key: parameters needed
            when the operator integrates with an external embedding service.

        Parameters:
        - queries (list[str]): list of text queries to be vectorized and
            searched

        Returns:
        - results (list[list[dict]]): for each query, a list of hit documents
            (concrete implementations should specify the document shape they
            return). Operators should remove large binary/vector fields from
            responses where possible.
        """
        pass

    @abstractmethod
    def run(self, records):
        """Main entry point used by the NV-Ingest pipeline.

        The `run` method is intended to be a simple orchestration layer that
        ensures the index exists and then ingests provided records. A minimal
        recommended implementation is::

                def run(self, records):
                        self.create_index()
                        self.write_to_index(records)

        Implementers can add pre/post hooks, metrics, retries, or error
        handling as needed for production readiness. Keep `run` simple so the
        pipeline orchestration remains predictable.

        Parameters:
        - records: NV-Ingest records to index (format follows repository
            conventions)

        Returns:
        - implementation-specific result (for example, a summary dict or
            boolean success flag).
        """
        pass

    def reindex(self, records: list, **kwargs):
        """Optional helper to rebuild or re-populate indexes with new data.

        This non-abstract method is provided as an optional hook that concrete
        classes may override. A typical reindex implementation will:
        - optionally delete the existing index and recreate it (via
            `create_index(recreate=True)`)
        - call `write_to_index(records)` to populate the new index

        Parameters:
        - records (list): records used to populate the index
        - recreate (bool, optional): whether to delete and recreate the
            index before writing

        Returns:
        - implementation-specific result
        """
        pass
