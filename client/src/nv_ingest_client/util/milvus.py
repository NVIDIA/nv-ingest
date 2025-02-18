from pymilvus import (
    MilvusClient,
    Collection,
    DataType,
    CollectionSchema,
    connections,
    Function,
    FunctionType,
    utility,
    BulkInsertState,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.milvus_client.index import IndexParams
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from scipy.sparse import csr_array
from typing import List
import time
from urllib.parse import urlparse
from typing import Union, Dict
from nv_ingest_client.util.util import ClientConfigSchema
import logging


logger = logging.getLogger(__name__)


def _dict_to_params(collections_dict: dict, write_params: dict):
    params_tuple_list = []
    for coll_name, data_type in collections_dict.items():
        cp_write_params = write_params.copy()
        enabled_dtypes = {
            "enable_text": False,
            "enable_charts": False,
            "enable_tables": False,
            "enable_images": False,
        }
        if not isinstance(data_type, list):
            data_type = [data_type]
        for d_type in data_type:
            enabled_dtypes[f"enable_{d_type}"] = True
        cp_write_params.update(enabled_dtypes)
        params_tuple_list.append((coll_name, cp_write_params))
    return params_tuple_list


class MilvusOperator:
    def __init__(
        self,
        collection_name: Union[str, Dict] = "nv_ingest_collection",
        milvus_uri: str = "http://localhost:19530",
        sparse: bool = True,
        recreate: bool = True,
        gpu_index: bool = True,
        gpu_search: bool = True,
        dense_dim: int = 2048,
        minio_endpoint: str = "localhost:9000",
        enable_text: bool = True,
        enable_charts: bool = True,
        enable_tables: bool = True,
        enable_images: bool = True,
        bm25_save_path: str = "bm25_model.json",
        compute_bm25_stats: bool = True,
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket_name: str = "a-bucket",
        **kwargs,
    ):
        self.milvus_kwargs = locals()
        self.milvus_kwargs.pop("self")
        self.collection_name = self.milvus_kwargs.pop("collection_name")
        self.milvus_kwargs.pop("kwargs", None)

    def get_connection_params(self):
        conn_dict = {
            "milvus_uri": self.milvus_kwargs["milvus_uri"],
            "sparse": self.milvus_kwargs["sparse"],
            "recreate": self.milvus_kwargs["recreate"],
            "gpu_index": self.milvus_kwargs["gpu_index"],
            "gpu_search": self.milvus_kwargs["gpu_search"],
            "dense_dim": self.milvus_kwargs["dense_dim"],
        }
        return (self.collection_name, conn_dict)

    def get_write_params(self):
        write_params = self.milvus_kwargs.copy()
        del write_params["recreate"]
        del write_params["gpu_index"]
        del write_params["gpu_search"]
        del write_params["dense_dim"]

        return (self.collection_name, write_params)

    def run(self, records):
        collection_name, create_params = self.get_connection_params()
        _, write_params = self.get_write_params()
        if isinstance(collection_name, str):
            create_nvingest_collection(collection_name, **create_params)
            write_to_nvingest_collection(records, collection_name, **write_params)
        elif isinstance(collection_name, dict):
            split_params_list = _dict_to_params(collection_name, write_params)
            for sub_params in split_params_list:
                coll_name, sub_write_params = sub_params
                create_nvingest_collection(coll_name, **create_params)
                write_to_nvingest_collection(records, coll_name, **sub_write_params)
        else:
            raise ValueError(f"Unsupported type for collection_name detected: {type(collection_name)}")


def create_nvingest_schema(dense_dim: int = 1024, sparse: bool = False, local_index: bool = False) -> CollectionSchema:
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
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
    schema.add_field(field_name="source", datatype=DataType.JSON)
    schema.add_field(field_name="content_metadata", datatype=DataType.JSON)
    if sparse and local_index:
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    elif sparse:
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "english"},
            enable_match=True,
        )
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names="sparse",
            )
        )

    else:
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    return schema


def create_nvingest_index_params(
    sparse: bool = False, gpu_index: bool = True, gpu_search: bool = True, local_index: bool = True
) -> IndexParams:
    """
    Creates index params necessary to create an index for a collection. At a minimum,
    this function will create a dense embedding index but can also create a sparse
    embedding index (BM25) for hybrid search.

    Parameters
    ----------
    sparse : bool, optional
        When set to true, this adds a Sparse index to the IndexParams, usually activated for
        hybrid search.
    gpu_index : bool, optional
        When set to true, creates an index on the GPU. The index is GPU_CAGRA.
    gpu_search : bool, optional
        When set to true, if using a gpu index, the search will be conducted using the GPU.
        Otherwise the search will be conducted on the CPU (index will be turned into HNSW).

    Returns
    -------
    IndexParams
        Returns index params setup for a dense embedding index and if specified, a sparse
        embedding index.
    """
    index_params = MilvusClient.prepare_index_params()
    if local_index:
        index_params.add_index(
            field_name="vector",
            index_name="dense_index",
            index_type="FLAT",
            metric_type="L2",
        )
    else:
        if gpu_index:
            index_params.add_index(
                field_name="vector",
                index_name="dense_index",
                index_type="GPU_CAGRA",
                metric_type="L2",
                params={
                    "intermediate_graph_degree": 128,
                    "graph_degree": 64,
                    "build_algo": "NN_DESCENT",
                    "adapt_for_cpu": "false" if gpu_search else "true",
                },
            )
        else:
            index_params.add_index(
                field_name="vector",
                index_name="dense_index",
                index_type="HNSW",
                metric_type="L2",
                params={"M": 64, "efConstruction": 512},
            )
    if sparse and local_index:
        index_params.add_index(
            field_name="sparse",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
            metric_type="IP",  # Currently, only IP (Inner Product) is supported for sparse vectors
            params={"drop_ratio_build": 0.2},  # The ratio of small vector values to be dropped during indexing
        )
    elif sparse:
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
    return index_params


def create_collection(
    client: MilvusClient,
    collection_name: str,
    schema: CollectionSchema,
    index_params: IndexParams = None,
    recreate=True,
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
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def create_nvingest_collection(
    collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    sparse: bool = False,
    recreate: bool = True,
    gpu_index: bool = True,
    gpu_search: bool = True,
    dense_dim: int = 2048,
) -> CollectionSchema:
    """
    Creates a milvus collection with an nv-ingest compatible schema under
    the target name.

    Parameters
    ----------
    collection_name : str
        Name of the collection to be created.
    milvus_uri : str,
        Milvus address with http(s) preffix and port. Can also be a file path, to activate
        milvus-lite.
    sparse : bool, optional
        When set to true, this adds a Sparse index to the IndexParams, usually activated for
        hybrid search.
    recreate : bool, optional
        If true, and the collection is detected, it will be dropped before being created
        again with the provided information (schema, index_params).
    gpu_cagra : bool, optional
        If true, creates a GPU_CAGRA index for dense embeddings.
    dense_dim : int, optional
        Sets the dimension size for the dense embedding in the milvus schema.

    Returns
    -------
    CollectionSchema
        Returns a milvus collection schema, that represents the fields in the created
        collection.
    """
    local_index = False
    if urlparse(milvus_uri).scheme:
        connections.connect(uri=milvus_uri)
        server_version = utility.get_server_version()
        if "lite" in server_version:
            gpu_index = False
    else:
        gpu_index = False
        if milvus_uri.endswith(".db"):
            local_index = True

    client = MilvusClient(milvus_uri)
    schema = create_nvingest_schema(dense_dim=dense_dim, sparse=sparse, local_index=local_index)
    index_params = create_nvingest_index_params(
        sparse=sparse, gpu_index=gpu_index, gpu_search=gpu_search, local_index=local_index
    )
    create_collection(client, collection_name, schema, index_params, recreate=recreate)


def _format_sparse_embedding(sparse_vector: csr_array):
    sparse_embedding = {int(k[1]): float(v) for k, v in sparse_vector.todok()._dict.items()}
    return sparse_embedding if len(sparse_embedding) > 0 else {int(0): float(0)}


def _record_dict(text, element, sparse_vector: csr_array = None):
    record = {
        "text": text,
        "vector": element["metadata"]["embedding"],
        "source": element["metadata"]["source_metadata"],
        "content_metadata": element["metadata"]["content_metadata"],
    }
    if sparse_vector is not None:
        record["sparse"] = _format_sparse_embedding(sparse_vector)
    return record


def verify_embedding(element):
    if element["metadata"]["embedding"] is not None:
        return True
    return False


def _pull_text(element, enable_text: bool, enable_charts: bool, enable_tables: bool, enable_images: bool):
    text = None
    if element["document_type"] == "text" and enable_text:
        text = element["metadata"]["content"]
    elif element["document_type"] == "structured":
        text = element["metadata"]["table_metadata"]["table_content"]
        if element["metadata"]["content_metadata"]["subtype"] == "chart" and not enable_charts:
            text = None
        elif element["metadata"]["content_metadata"]["subtype"] == "table" and not enable_tables:
            text = None
    elif element["document_type"] == "image" and enable_images:
        text = element["metadata"]["image_metadata"]["caption"]
    verify_emb = verify_embedding(element)
    if not text or not verify_emb:
        source_name = element["metadata"]["source_metadata"]["source_name"]
        pg_num = element["metadata"]["content_metadata"]["page_number"]
        doc_type = element["document_type"]
        if not verify_emb:
            logger.error(f"failed to find embedding for entity: {source_name} page: {pg_num} type: {doc_type}")
        if not text:
            logger.error(f"failed to find text for entity: {source_name} page: {pg_num} type: {doc_type}")
        # if we do find text but no embedding remove anyway
        text = None
    return text


def _insert_location_into_content_metadata(element, enable_charts: bool, enable_tables: bool, enable_images: bool):
    location = max_dimensions = None
    if element["document_type"] == "structured":
        location = element["metadata"]["table_metadata"]["table_location"]
        max_dimensions = element["metadata"]["table_metadata"]["table_location_max_dimensions"]
        if element["metadata"]["content_metadata"]["subtype"] == "chart" and not enable_charts:
            location = max_dimensions = None
        elif element["metadata"]["content_metadata"]["subtype"] == "table" and not enable_tables:
            location = max_dimensions = None
    elif element["document_type"] == "image" and enable_images:
        location = element["metadata"]["image_metadata"]["image_location"]
        max_dimensions = element["metadata"]["image_metadata"]["image_location_max_dimensions"]
    if (not location) and (element["document_type"] != "text"):
        source_name = element["metadata"]["source_metadata"]["source_name"]
        pg_num = element["metadata"]["content_metadata"]["page_number"]
        doc_type = element["document_type"]
        logger.error(f"failed to find location for entity: {source_name} page: {pg_num} type: {doc_type}")
        location = max_dimensions = None
    element["metadata"]["content_metadata"]["location"] = location
    element["metadata"]["content_metadata"]["max_dimensions"] = max_dimensions


def write_records_minio(
    records,
    writer: RemoteBulkWriter,
    sparse_model=None,
    enable_text: bool = True,
    enable_charts: bool = True,
    enable_tables: bool = True,
    enable_images: bool = True,
    record_func=_record_dict,
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
    enable_images : bool, optional
        When true, ensure all image type records are used.
    record_func : function, optional
        This function will be used to parse the records for necessary information.

    Returns
    -------
    RemoteBulkWriter
        Returns the writer supplied, with information related to minio records upload.
    """
    for result in records:
        for element in result:
            text = _pull_text(element, enable_text, enable_charts, enable_tables, enable_images)
            _insert_location_into_content_metadata(element, enable_charts, enable_tables, enable_images)
            if text:
                if sparse_model is not None:
                    writer.append_row(record_func(text, element, sparse_model.encode_documents([text])))
                else:
                    writer.append_row(record_func(text, element))

    writer.commit()
    print(f"Wrote data to: {writer.batch_files}")
    return writer


def bulk_insert_milvus(collection_name: str, writer: RemoteBulkWriter, milvus_uri: str = "http://localhost:19530"):
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
        Milvus address with http(s) preffix and port. Can also be a file path, to activate
        milvus-lite.
    """

    connections.connect(uri=milvus_uri)
    t_bulk_start = time.time()
    task_id = utility.do_bulk_insert(collection_name=collection_name, files=writer.batch_files[0])
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
    enable_charts: bool = True,
    enable_tables: bool = True,
    enable_images: bool = True,
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
    enable_images : bool, optional
        When true, ensure all image type records are used.

    Returns
    -------
    BM25EmbeddingFunction
        Returns the model fitted to the selected corpus.
    """
    all_text = []
    for result in records:
        for element in result:
            text = _pull_text(element, enable_text, enable_charts, enable_tables, enable_images)
            if text:
                all_text.append(text)

    analyzer = build_default_analyzer(language="en")
    bm25_ef = BM25EmbeddingFunction(analyzer)

    bm25_ef.fit(all_text)
    return bm25_ef


def stream_insert_milvus(
    records,
    client: MilvusClient,
    collection_name: str,
    sparse_model=None,
    enable_text: bool = True,
    enable_charts: bool = True,
    enable_tables: bool = True,
    enable_images: bool = True,
    record_func=_record_dict,
):
    """
    This function takes the input records and creates a corpus,
    factoring in filters (i.e. texts, charts, tables) and fits
    a BM25 model with that information.

    Parameters
    ----------
    records : List
        List of chunks with attached metadata
    collection_name : str
        Milvus Collection to search against
    sparse_model : model,
        Sparse model used to generate sparse embedding in the form of
        scipy.sparse.csr_array
    enable_text : bool, optional
        When true, ensure all text type records are used.
    enable_charts : bool, optional
        When true, ensure all chart type records are used.
    enable_tables : bool, optional
        When true, ensure all table type records are used.
    enable_images : bool, optional
        When true, ensure all image type records are used.
    record_func : function, optional
        This function will be used to parse the records for necessary information.

    """
    data = []
    for result in records:
        for element in result:
            text = _pull_text(element, enable_text, enable_charts, enable_tables, enable_images)
            _insert_location_into_content_metadata(element, enable_charts, enable_tables, enable_images)
            if text:
                if sparse_model is not None:
                    data.append(record_func(text, element, sparse_model.encode_documents([text])))
                else:
                    data.append(record_func(text, element))
    client.insert(collection_name=collection_name, data=data)


def write_to_nvingest_collection(
    records,
    collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    minio_endpoint: str = "localhost:9000",
    sparse: bool = True,
    enable_text: bool = True,
    enable_charts: bool = True,
    enable_tables: bool = True,
    enable_images: bool = True,
    bm25_save_path: str = "bm25_model.json",
    compute_bm25_stats: bool = True,
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    bucket_name: str = "a-bucket",
    threshold: int = 10,
):
    """
    This function takes the input records and creates a corpus,
    factoring in filters (i.e. texts, charts, tables) and fits
    a BM25 model with that information.

    Parameters
    ----------
    records : List
        List of chunks with attached metadata
    collection_name : str
        Milvus Collection to search against
    milvus_uri : str,
        Milvus address with http(s) preffix and port. Can also be a file path, to activate
        milvus-lite.
    minio_endpoint : str,
        Endpoint for the minio instance attached to your milvus.
    enable_text : bool, optional
        When true, ensure all text type records are used.
    enable_charts : bool, optional
        When true, ensure all chart type records are used.
    enable_tables : bool, optional
        When true, ensure all table type records are used.
    enable_images : bool, optional
        When true, ensure all image type records are used.
    sparse : bool, optional
        When true, incorporates sparse embedding representations for records.
    bm25_save_path : str, optional
        The desired filepath for the sparse model if sparse is True.
    access_key : str, optional
        Minio access key.
    secret_key : str, optional
        Minio secret key.
    bucket_name : str, optional
        Minio bucket name.
    """
    stream = False
    local_index = False
    connections.connect(uri=milvus_uri)
    if urlparse(milvus_uri).scheme:
        server_version = utility.get_server_version()
        if "lite" in server_version:
            stream = True
    else:
        stream = True
    if milvus_uri.endswith(".db"):
        local_index = True
    bm25_ef = None
    if local_index and sparse and compute_bm25_stats:
        bm25_ef = create_bm25_model(
            records,
            enable_text=enable_text,
            enable_charts=enable_charts,
            enable_tables=enable_tables,
            enable_images=enable_images,
        )
        bm25_ef.save(bm25_save_path)
    elif local_index and sparse:
        bm25_ef = BM25EmbeddingFunction(build_default_analyzer(language="en"))
        bm25_ef.load(bm25_save_path)
    client = MilvusClient(milvus_uri)
    schema = Collection(collection_name).schema
    if len(records) < threshold:
        stream = True
    if stream:
        stream_insert_milvus(
            records,
            client,
            collection_name,
            bm25_ef,
            enable_text=enable_text,
            enable_charts=enable_charts,
            enable_tables=enable_tables,
            enable_images=enable_images,
        )
    else:
        # Connections parameters to access the remote bucket
        conn = RemoteBulkWriter.S3ConnectParam(
            endpoint=minio_endpoint,  # the default MinIO service started along with Milvus
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket_name,
            secure=False,
        )
        text_writer = RemoteBulkWriter(
            schema=schema, remote_path="/", connect_param=conn, file_type=BulkFileType.PARQUET
        )
        writer = write_records_minio(
            records,
            text_writer,
            bm25_ef,
            enable_text=enable_text,
            enable_charts=enable_charts,
            enable_tables=enable_tables,
            enable_images=enable_images,
        )
        bulk_insert_milvus(collection_name, writer, milvus_uri)
        # this sleep is required, to ensure atleast this amount of time
        # passes before running a search against the collection.\
    time.sleep(20)


def dense_retrieval(
    queries,
    collection_name: str,
    client: MilvusClient,
    dense_model,
    top_k: int,
    dense_field: str = "vector",
    output_fields: List[str] = ["text"],
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
    client : MilvusClient
        Client connected to mivlus instance.
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

    results = client.search(
        collection_name=collection_name,
        data=dense_embeddings,
        anns_field=dense_field,
        limit=top_k,
        output_fields=output_fields,
    )
    return results


def hybrid_retrieval(
    queries,
    collection_name: str,
    client: MilvusClient,
    dense_model,
    sparse_model,
    top_k: int,
    dense_field: str = "vector",
    sparse_field: str = "sparse",
    output_fields: List[str] = ["text"],
    gpu_search: bool = True,
    local_index: bool = False,
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
    client : MilvusClient
        Client connected to mivlus instance.
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
        if sparse_model:
            sparse_embeddings.append(_format_sparse_embedding(sparse_model.encode_queries([query])))
        else:
            sparse_embeddings.append(query)

    s_param_1 = {
        "metric_type": "L2",
    }
    if not gpu_search and not local_index:
        s_param_1["params"] = {"ef": top_k * 2}

    # Create search requests for both vector types
    search_param_1 = {
        "data": dense_embeddings,
        "anns_field": dense_field,
        "param": s_param_1,
        "limit": top_k * 2,
    }

    dense_req = AnnSearchRequest(**search_param_1)
    s_param_2 = {"metric_type": "BM25"}
    if local_index:
        s_param_2 = {"metric_type": "IP", "params": {"drop_ratio_build": 0.0}}

    search_param_2 = {
        "data": sparse_embeddings,
        "anns_field": sparse_field,
        "param": s_param_2,
        "limit": top_k * 2,
    }
    sparse_req = AnnSearchRequest(**search_param_2)

    results = client.hybrid_search(
        collection_name, [sparse_req, dense_req], RRFRanker(), limit=top_k, output_fields=output_fields
    )
    return results


def nvingest_retrieval(
    queries,
    collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    top_k: int = 5,
    hybrid: bool = False,
    dense_field: str = "vector",
    sparse_field: str = "sparse",
    embedding_endpoint=None,
    sparse_model_filepath: str = "bm25_model.json",
    model_name: str = None,
    output_fields: List[str] = ["text", "source", "content_metadata"],
    gpu_search: bool = True,
):
    """
    This function takes the input queries and conducts a hybrid/dense
    embedding search against the vectors, returning the top_k nearest
    records in the collection.

    Parameters
    ----------
    queries : List
        List of queries
    collection : Collection
        Milvus Collection to search against
    milvus_uri : str,
        Milvus address with http(s) preffix and port. Can also be a file path, to activate
        milvus-lite.
    top_k : int
        Number of search results to return per query.
    hybrid: bool, optional
        If True, will calculate distances for both dense and sparse embeddings.
    dense_field : str, optional
        The name of the anns_field that holds the dense embedding
        vector the collection.
    sparse_field : str, optional
        The name of the anns_field that holds the sparse embedding
        vector the collection.
    embedding_endpoint : str, optional
        Number of search results to return per query.
    sparse_model_filepath : str, optional
        The path where the sparse model has been loaded.
    model_name : str, optional
        The name of the dense embedding model available in the NIM embedding endpoint.

    Returns
    -------
    List
        Nested list of top_k results per query.
    """
    client_config = ClientConfigSchema()
    nvidia_api_key = client_config.nvidia_build_api_key
    # required for NVIDIAEmbedding call if the endpoint is Nvidia build api.
    embedding_endpoint = embedding_endpoint if embedding_endpoint else client_config.embedding_nim_endpoint
    model_name = model_name if model_name else client_config.embedding_nim_model_name
    local_index = False
    embed_model = NVIDIAEmbedding(base_url=embedding_endpoint, model=model_name, nvidia_api_key=nvidia_api_key)
    client = MilvusClient(milvus_uri)
    if milvus_uri.endswith(".db"):
        local_index = True
    if hybrid:
        bm25_ef = None
        if local_index:
            bm25_ef = BM25EmbeddingFunction(build_default_analyzer(language="en"))
            bm25_ef.load(sparse_model_filepath)
        results = hybrid_retrieval(
            queries,
            collection_name,
            client,
            embed_model,
            bm25_ef,
            top_k,
            output_fields=output_fields,
            gpu_search=gpu_search,
            local_index=local_index,
        )
    else:
        results = dense_retrieval(queries, collection_name, client, embed_model, top_k, output_fields=output_fields)
    return results


def remove_records(source_name: str, collection_name: str, milvus_uri: str = "http://localhost:19530"):
    """
    This function allows a user to remove chunks associated with an ingested file.
    Supply the full path of the file you would like to remove and this function will
    remove all the chunks associated with that file in the target collection.

    Parameters
    ----------
    source_name : str
        The full file path of the file you would like to remove from the collection.
    collection_name : str
        Milvus Collection to query against
    milvus_uri : str,
        Milvus address with http(s) preffix and port. Can also be a file path, to activate
        milvus-lite.

    Returns
    -------
    Dict
        Dictionary with one key, `delete_cnt`. The value represents the number of entities
        removed.
    """
    client = MilvusClient(milvus_uri)
    result_ids = client.delete(
        collection_name=collection_name,
        filter=f'(source["source_name"] == "{source_name}")',
    )
    return result_ids
