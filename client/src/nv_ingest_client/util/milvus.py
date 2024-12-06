from pymilvus import MilvusClient, DataType, Collection, CollectionSchema, utility, BulkInsertState, AnnSearchRequest, RRFRanker
from pymilvus.milvus_client.index import IndexParams

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


