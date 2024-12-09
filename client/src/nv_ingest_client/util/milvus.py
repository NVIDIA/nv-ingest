from pymilvus import MilvusClient, DataType, Collection, CollectionSchema, connections, utility, BulkInsertState, AnnSearchRequest, RRFRanker
from pymilvus.milvus_client.index import IndexParams
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from scipy.sparse import csr_array
import time

def create_nvingest_schema(
    dense_dim: int = 1024, 
    sparse: bool = False
    ) -> CollectionSchema: 
    """
    Creates a schema for the nv-ingest produced data. This is currently setup to follow 
    the default expected schema fields in nv-ingest. You can see more about the declared fields
    in the `nv_ingest.schemas.vdb_task_sink_schema.build_default_milvus_config` function. This 
    schema should have the fields declared in that function, at a minimum. To ensure proper
    data propagation to milvus.  

    Parameters
    ----------
    dense_dim : int, optional
        The size of the embedding dimension.
    sparse : bool, optional
        When set to true, this adds a Sparse field to the schema, usually activated for 
        hybrid search.

    Returns
    -------
    CollectionSchema
        Returns a milvus collection schema, with the minimum required nv-ingest fields
        and extra fields (sparse), if specified by the user.
    """
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
    schema.add_field(field_name="source", datatype=DataType.JSON)
    schema.add_field(field_name="content_metadata", datatype=DataType.JSON)
    if sparse:
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

    return schema


def create_nvingest_index_params(sparse:bool = False) -> IndexParams:
    """
    Creates index params necessary to create an index for a collection. At a minimum,
    this function will create a dense embedding index but can also create a sparse 
    embedding index (BM25) for hybrid search.  

    Parameters
    ----------
    sparse : bool, optional
        When set to true, this adds a Sparse index to the IndexParams, usually activated for 
        hybrid search.

    Returns
    -------
    IndexParams
        Returns index params setup for a dense embedding index and if specified, a sparse
        embedding index.
    """
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name="dense_index",
        index_type= "GPU_CAGRA",
        metric_type="L2",
        params={
            'intermediate_graph_degree':128,
            'graph_degree': 64,
            "build_algo": "NN_DESCENT",
        },
    )
    if sparse:
        index_params.add_index(
            field_name="sparse",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
            metric_type="IP",  # Currently, only IP (Inner Product) is supported for sparse vectors
            params={"drop_ratio_build": 0.2},  # The ratio of small vector values to be dropped during indexing
        )
    return index_params


def create_collection(
    client: MilvusClient, 
    collection_name: str, 
    schema: CollectionSchema, 
    index_params: IndexParams=None, 
    recreate=True
    ):
    """
    Creates a milvus collection with the supplied name and schema. Within that collection,
    this function ensures that the desired indexes are created based on the IndexParams 
    supplied.  

    Parameters
    ----------
    client : MilvusClient
        Client connected to mivlus instance.
    collection_name : str
        Name of the collection to be created.
    schema : CollectionSchema,
        Schema that identifies the fields of data that will be available in the collection.
    index_params : IndexParams, optional
        The parameters used to create the index(es) for the associated collection fields.
    recreate : bool, optional
        If true, and the collection is detected, it will be dropped before being created 
        again with the provided information (schema, index_params).
    """
    if recreate and client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

def _record_dict(text, element, sparse_vector: csr_array = None):
    record = {
        "text": text,
        "vector": element['metadata']['embedding'],
        "source": element['metadata']['source_metadata'],
        "content_metadata":  element['metadata']['content_metadata'],
    }
    if sparse_vector is not None:
        record["sparse"] = {int(k[1]): float(v) for k, v in sparse_vector.todok()._dict.items()}
    return record

def _pull_text(element, enable_text: bool, enable_charts: bool, enable_tables: bool):
    text = None
    if element['document_type'] == 'text' and enable_text:
        text =  element['metadata']['content']
    elif element['document_type'] == 'structured':
        text = element['metadata']['table_metadata']['table_content']
        if element["metadata"]["content_metadata"]["subtype"] == "chart" and not enable_charts:
            text = None
        elif element["metadata"]["content_metadata"]["subtype"] == "table" and not enable_tables:
            text = None
    return text

def write_records_minio(
    records, 
    writer: RemoteBulkWriter, 
    sparse_model = None,
    enable_text:bool = True, 
    enable_charts: bool = True, 
    enable_tables:bool = True, 
    record_func=_record_dict
    ) -> RemoteBulkWriter: 
    """
    Writes the supplied records to milvus using the supplied writer. 
    If a sparse model is supplied, it will be used to generate sparse 
    embeddings to allow for hybrid search. Will filter records based on 
    type, depending on what types are enabled via the boolean parameters.  

    Parameters
    ----------
    records : List
        List of chunks with attached metadata
    writer : RemoteBulkWriter
        The Milvus Remote BulkWriter instance that was created with necessary 
        params to access the minio instance corresponding to milvus.
    sparse_model : model,
        Sparse model used to generate sparse embedding in the form of 
        scipy.sparse.csr_array 
    enable_text : bool, optional
        When true, ensure all text type records are used.
    enable_charts : bool, optional
        When true, ensure all chart type records are used.
    enable_tables : bool, optional
        When true, ensure all table type records are used.
    record_func : function, optional
        This function will be used to parse the records for necessary information.

    Returns
    -------
    RemoteBulkWriter
        Returns the writer supplied, with information related to minio records upload.
    """
    for result in records:
        for element in result:
            text = _pull_text(element, enable_text, enable_charts, enable_tables)
            if text and sparse_model is not None:
                writer.append_row(record_func(text, element, sparse_model.encode_documents([text])))
            else:
                writer.append_row(record_func(text, element))

    writer.commit()
    print(f"Wrote data to: {writer.batch_files}")
    return writer


def bulk_insert_milvus(
    collection_name: str, 
    writer:RemoteBulkWriter, 
    milvus_uri: str="http://localhost:19530"
    ):
    """
    This function initialize the bulk ingest of all minio uploaded records, and checks for
    milvus task completion. Once the function is complete all records have been uploaded 
    to the milvus collection.  

    Parameters
    ----------
    collection_name : str
        Name of the milvus collection.
    writer : RemoteBulkWriter
        The Milvus Remote BulkWriter instance that was created with necessary 
        params to access the minio instance corresponding to milvus.
    milvus_uri : str,
        The location of the milvus instance where the selected collection exists.
    """

    connections.connect(uri=milvus_uri)
    t_bulk_start = time.time()
    task_id = utility.do_bulk_insert(
        collection_name=collection_name,
        files=writer.batch_files[0]
    )
    state = "Pending"
    while state != "Completed":
        task = utility.get_bulk_insert_state(task_id=task_id)
        state = task.state_name
        if state == "Completed":
          t_bulk_end = time.time()
          print("Start time:", task.create_time_str)
          print("Imported row count:", task.row_count)
          print(f"Bulk {collection_name} upload took {t_bulk_end - t_bulk_start} s")
        if task.state == BulkInsertState.ImportFailed:
            print("Failed reason:", task.failed_reason)
        time.sleep(1)


def create_bm25_model(
    records, 
    enable_text: bool = True, 
    enable_charts:bool = True, 
    enable_tables: bool = True
    ) -> BM25EmbeddingFunction:
    """
    This function takes the input records and creates a corpus, 
    factoring in filters (i.e. texts, charts, tables) and fits
    a BM25 model with that information.  

    Parameters
    ----------
    records : List
        List of chunks with attached metadata
    enable_text : bool, optional
        When true, ensure all text type records are used.
    enable_charts : bool, optional
        When true, ensure all chart type records are used.
    enable_tables : bool, optional
        When true, ensure all table type records are used.

    Returns
    -------
    BM25EmbeddingFunction
        Returns the model fitted to the selected corpus.
    """
    all_text = []
    for result in records:
        for element in result:
            text = _pull_text(element, enable_text, enable_charts, enable_tables)
            if text:
                all_text.append(text)

    analyzer = build_default_analyzer(language="en")
    bm25_ef = BM25EmbeddingFunction(analyzer)

    bm25_ef.fit(all_text)
    return bm25_ef


def dense_retrieval(
    queries, 
    collection: Collection, 
    dense_model, 
    top_k: int, 
    dense_field:str = "vector"
    ):
    """
    This function takes the input queries and conducts a dense
    embedding search against the dense vector and return the top_k
    nearest records in the collection.

    Parameters
    ----------
    queries : List
        List of queries
    collection : Collection
        Milvus Collection to search against
    dense_model : NVIDIAEmbedding
        Dense model to generate dense embeddings for queries.
    top_k : int
        Number of search results to return per query.
    dense_field : str
        The name of the anns_field that holds the dense embedding 
        vector the collection.

    Returns
    -------
    List
        Nested list of top_k results per query.
    """
    dense_embeddings = []
    for query in queries:
        dense_embeddings.append(dense_model.get_query_embedding(query))

    results =  collection.search(
        data = dense_embeddings,
        anns_field=dense_field,
        param={"metric_type": "L2"},
        limit=top_k,
        output_fields=["text"]
    )
    return results


def hybrid_retrieval(
    queries, 
    collection: Collection, 
    dense_model, 
    sparse_model, 
    top_k: int, 
    dense_field: str = "vector",
    sparse_field: str = "sparse" 
    ):
    """
    This function takes the input queries and conducts a hybrid
    embedding search against the dense and sparse vectors, returning 
    the top_k nearest records in the collection.

    Parameters
    ----------
    queries : List
        List of queries
    collection : Collection
        Milvus Collection to search against
    dense_model : NVIDIAEmbedding
        Dense model to generate dense embeddings for queries.
    sparse_model : model,
        Sparse model used to generate sparse embedding in the form of
        scipy.sparse.csr_array 
    top_k : int
        Number of search results to return per query.
    dense_field : str
        The name of the anns_field that holds the dense embedding 
        vector the collection.
    sparse_field : str
        The name of the anns_field that holds the sparse embedding 
        vector the collection.

    Returns
    -------
    List
        Nested list of top_k results per query.
    """
    dense_embeddings = []
    sparse_embeddings = []
    for query in queries:
        dense_embeddings.append(dense_model.get_query_embedding(query))
        sparse_embeddings.append({int(k[1]): float(v) for k, v in sparse_model.encode_queries([query]).todok()._dict.items()})

    # Create search requests for both vector types
    search_param_1 = {
        "data": dense_embeddings,
        "anns_field": dense_field,
        "param": {
            "metric_type": "L2",
        },
        "limit": top_k
    }
    dense_req = AnnSearchRequest(**search_param_1)
    
    search_param_2 = {
        "data": sparse_embeddings,
        "anns_field": sparse_field,
        "param": {
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2}
        },
        "limit": top_k
    }
    sparse_req = AnnSearchRequest(**search_param_2)

    results =  collection.hybrid_search(
        [sparse_req, dense_req],
        rerank=RRFRanker(),
        limit=top_k,
        output_fields=["text"]
    )
    return results
