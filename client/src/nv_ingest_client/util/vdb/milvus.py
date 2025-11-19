import ast
import copy
import datetime
import json
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from minio import Minio
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob
from nv_ingest_client.util.transport import infer_microservice
from nv_ingest_client.util.util import ClientConfigSchema
from nv_ingest_client.util.vdb.adt_vdb import VDB
from pymilvus import AnnSearchRequest
from pymilvus import BulkInsertState
from pymilvus import Collection
from pymilvus import CollectionSchema
from pymilvus import DataType
from pymilvus import Function
from pymilvus import FunctionType
from pymilvus import MilvusClient
from pymilvus import RRFRanker
from pymilvus import connections
from pymilvus import utility
from pymilvus.bulk_writer import BulkFileType
from pymilvus.bulk_writer import RemoteBulkWriter
from pymilvus.milvus_client.index import IndexParams
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.orm.types import CONSISTENCY_BOUNDED
from scipy.sparse import csr_array


logger = logging.getLogger(__name__)

CONSISTENCY = CONSISTENCY_BOUNDED
DENSE_INDEX_NAME = "dense_index"

pandas_reader_map = {
    ".json": pd.read_json,
    ".csv": partial(pd.read_csv, index_col=0),
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
}


def pandas_file_reader(input_file: str):
    path_file = Path(input_file)
    if not path_file.exists:
        raise ValueError(f"File does not exist: {input_file}")
    file_type = path_file.suffix
    return pandas_reader_map[file_type](input_file)


def create_nvingest_meta_schema():
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    # collection name, timestamp, index_types - dimensions, embedding_model, fields
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(
        field_name="collection_name",
        datatype=DataType.VARCHAR,
        max_length=65535,
        # enable_analyzer=True,
        # enable_match=True
    )
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2)
    schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="indexes", datatype=DataType.JSON)
    schema.add_field(field_name="models", datatype=DataType.JSON)
    schema.add_field(field_name="user_fields", datatype=DataType.JSON)
    return schema


def create_meta_collection(
    schema: CollectionSchema,
    collection_name: str = "meta",
    recreate=False,
    client: MilvusClient = None,
):
    if client.has_collection(collection_name) and not recreate:
        # already exists, dont erase and recreate
        return
    schema = create_nvingest_meta_schema()
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name=DENSE_INDEX_NAME,
        index_type="FLAT",
        metric_type="L2",
    )
    create_collection(client, collection_name, schema, index_params=index_params, recreate=recreate)


def write_meta_collection(
    collection_name: str,
    fields: List[str],
    creation_timestamp: str = None,
    dense_index: str = None,
    dense_dim: int = None,
    sparse_index: str = None,
    embedding_model: str = None,
    sparse_model: str = None,
    meta_collection_name: str = "meta",
    client: MilvusClient = None,
):
    client_config = ClientConfigSchema()
    data = {
        "collection_name": collection_name,
        "vector": [0.0] * 2,
        "timestamp": str(creation_timestamp or datetime.datetime.now()),
        "indexes": {
            "dense_index": dense_index,
            "dense_dimension": dense_dim,
            "sparse_index": sparse_index,
        },
        "models": {
            "embedding_model": embedding_model or client_config.embedding_nim_model_name,
            "embedding_dim": dense_dim,
            "sparse_model": sparse_model,
        },
        "user_fields": [field.name for field in fields],
    }
    client.insert(collection_name=meta_collection_name, data=data)


def log_new_meta_collection(
    collection_name: str,
    fields: List[str],
    creation_timestamp: str = None,
    dense_index: str = None,
    dense_dim: int = None,
    sparse_index: str = None,
    embedding_model: str = None,
    sparse_model: str = None,
    meta_collection_name: str = "meta",
    recreate: bool = False,
    client: MilvusClient = None,
):
    schema = create_nvingest_meta_schema()
    create_meta_collection(schema, client=client, recreate=recreate)
    write_meta_collection(
        collection_name,
        fields=fields,
        creation_timestamp=creation_timestamp,
        dense_index=dense_index,
        dense_dim=dense_dim,
        sparse_index=sparse_index,
        embedding_model=embedding_model,
        sparse_model=sparse_model,
        meta_collection_name=meta_collection_name,
        client=client,
    )


def grab_meta_collection_info(
    collection_name: str,
    meta_collection_name: str = "meta",
    timestamp: str = None,
    embedding_model: str = None,
    embedding_dim: int = None,
    client: MilvusClient = None,
    milvus_uri: str = None,
    username: str = None,
    password: str = None,
):
    timestamp = timestamp or ""
    embedding_model = embedding_model or ""
    embedding_dim = embedding_dim or ""
    if milvus_uri:
        client = MilvusClient(milvus_uri, token=f"{username}:{password}")
    results = client.query_iterator(
        collection_name=meta_collection_name,
        output_fields=[
            "collection_name",
            "timestamp",
            "indexes",
            "models",
            "user_fields",
        ],
    )
    query_res = []
    res = results.next()
    while res:
        query_res += res
        res = results.next()
    result = []
    for res in query_res:
        if (
            collection_name in res["collection_name"]
            and timestamp in res["timestamp"]
            and embedding_model in res["models"]["embedding_model"]
            and str(embedding_dim) in str(res["models"]["embedding_dim"])
        ):
            result.append(res)
    return result


def _dict_to_params(collections_dict: dict, write_params: dict):
    params_tuple_list = []
    for coll_name, data_type in collections_dict.items():
        cp_write_params = write_params.copy()
        enabled_dtypes = {
            "enable_text": False,
            "enable_charts": False,
            "enable_tables": False,
            "enable_images": False,
            "enable_infographics": False,
        }
        if not isinstance(data_type, list):
            data_type = [data_type]
        for d_type in data_type:
            enabled_dtypes[f"enable_{d_type}"] = True
        cp_write_params.update(enabled_dtypes)
        params_tuple_list.append((coll_name, cp_write_params))
    return params_tuple_list


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
    schema.add_field(
        field_name="content_metadata",
        datatype=DataType.JSON,
        nullable=True if not local_index else False,
    )
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
    sparse: bool = False,
    gpu_index: bool = True,
    gpu_search: bool = False,
    local_index: bool = True,
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
            index_name=DENSE_INDEX_NAME,
            index_type="FLAT",
            metric_type="L2",
        )
    else:
        if gpu_index:
            index_params.add_index(
                field_name="vector",
                index_name=DENSE_INDEX_NAME,
                index_type="GPU_CAGRA",
                metric_type="L2",
                params={
                    "intermediate_graph_degree": 128,
                    "graph_degree": 100,
                    "build_algo": "NN_DESCENT",
                    "cache_dataset_on_device": "true",
                    "adapt_for_cpu": "false" if gpu_search else "true",
                },
            )
        else:
            index_params.add_index(
                field_name="vector",
                index_name=DENSE_INDEX_NAME,
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
            index_name="sparse_index",
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
    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=CONSISTENCY,
        )


def create_nvingest_collection(
    collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    sparse: bool = False,
    recreate: bool = True,
    gpu_index: bool = True,
    gpu_search: bool = False,
    dense_dim: int = 2048,
    recreate_meta: bool = False,
    username: str = None,
    password: str = None,
) -> CollectionSchema:
    """
    Creates a milvus collection with an nv-ingest compatible schema under
    the target name.

    Parameters
    ----------
    collection_name : str
        Name of the collection to be created.

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
    username : str, optional
        Milvus username.
    password : str, optional
        Milvus password.


    Returns
    -------
    CollectionSchema
        Returns a milvus collection schema, that represents the fields in the created
        collection.
    """
    local_index = False
    if urlparse(milvus_uri).scheme:
        connections.connect(uri=milvus_uri, token=f"{username}:{password}")
        server_version = utility.get_server_version()
        if "lite" in server_version:
            gpu_index = False
    else:
        gpu_index = False
        if milvus_uri.endswith(".db"):
            local_index = True

    client = MilvusClient(milvus_uri, token=f"{username}:{password}")
    schema = create_nvingest_schema(dense_dim=dense_dim, sparse=sparse, local_index=local_index)
    index_params = create_nvingest_index_params(
        sparse=sparse,
        gpu_index=gpu_index,
        gpu_search=gpu_search,
        local_index=local_index,
    )
    create_collection(client, collection_name, schema, index_params, recreate=recreate)
    d_idx, s_idx = _get_index_types(index_params, sparse=sparse)
    log_new_meta_collection(
        collection_name,
        fields=schema.fields,
        dense_index=str(d_idx),
        dense_dim=dense_dim,
        sparse_index=str(s_idx),
        recreate=recreate_meta,
        client=client,
    )
    return schema


def _get_index_types(index_params: IndexParams, sparse: bool = False) -> Tuple[str, str]:
    """
    Returns the dense and optional sparse index types from Milvus index_params,
    handling both old (dict) and new (list) formats.

    Parameters:
        index_params: The index parameters object with a _indexes attribute.
        sparse (bool): Whether to look for sparse_index as well.

    Returns:
        tuple: (dense_index_type, sparse_index_type or None)
    """
    d_idx = None
    s_idx = None

    indexes = getattr(index_params, "_indexes", None)
    if indexes is None:
        indexes = {(idx, index_param.index_name): index_param for idx, index_param in enumerate(index_params)}

    if isinstance(indexes, dict):
        # Old Milvus behavior (< 2.5.6)
        for k, v in indexes.items():
            if k[1] == DENSE_INDEX_NAME and hasattr(v, "_index_type"):
                d_idx = v._index_type
            if sparse and k[1] == "sparse_index" and hasattr(v, "_index_type"):
                s_idx = v._index_type

    elif isinstance(indexes, list):
        # New Milvus behavior (>= 2.5.6)
        for idx in indexes:
            index_name = getattr(idx, "index_name", None)
            index_type = getattr(idx, "index_type", None)

            if index_name == DENSE_INDEX_NAME:
                d_idx = index_type
            if sparse and index_name == "sparse_index":
                s_idx = index_type

    else:
        raise TypeError(f"Unexpected type for index_params._indexes: {type(indexes)}")

    return str(d_idx), str(s_idx)


def _format_sparse_embedding(sparse_vector: csr_array):
    sparse_embedding = {int(k[1]): float(v) for k, v in sparse_vector.todok()._dict.items()}
    return sparse_embedding if len(sparse_embedding) > 0 else {int(0): float(0)}


def _record_dict(text, element, sparse_vector: csr_array = None):
    cp_element = copy.deepcopy(element)
    cp_element["metadata"].pop("content")
    record = {
        "text": text,
        "vector": cp_element["metadata"].pop("embedding"),
        "source": cp_element["metadata"].pop("source_metadata"),
        "content_metadata": cp_element["metadata"].pop("content_metadata"),
    }
    # need to grab the user defined fields and add them to the content_metadata
    record["content_metadata"].update(cp_element["metadata"])
    if sparse_vector is not None:
        record["sparse"] = _format_sparse_embedding(sparse_vector)
    return record


def verify_embedding(element):
    if element["metadata"]["embedding"] is not None:
        return True
    return False


def cleanup_records(
    records,
    enable_text: bool = True,
    enable_charts: bool = True,
    enable_tables: bool = True,
    enable_images: bool = True,
    enable_infographics: bool = True,
    enable_audio: bool = True,
    meta_dataframe: pd.DataFrame = None,
    meta_source_field: str = None,
    meta_fields: list[str] = None,
    record_func=_record_dict,
    sparse_model=None,
):
    cleaned_records = []
    for result in records:
        if result is not None:
            if isinstance(result, dict):
                result = [result]
            for element in result:
                text = _pull_text(
                    element,
                    enable_text,
                    enable_charts,
                    enable_tables,
                    enable_images,
                    enable_infographics,
                    enable_audio,
                )
                _insert_location_into_content_metadata(
                    element,
                    enable_charts,
                    enable_tables,
                    enable_images,
                    enable_infographics,
                )
                if meta_dataframe is not None and meta_source_field and meta_fields:
                    add_metadata(element, meta_dataframe, meta_source_field, meta_fields)
                if text:
                    if sparse_model is not None:
                        element = record_func(text, element, sparse_model.encode_documents([text]))
                    else:
                        element = record_func(text, element)
                    cleaned_records.append(element)
    return cleaned_records


def _pull_text(
    element,
    enable_text: bool,
    enable_charts: bool,
    enable_tables: bool,
    enable_images: bool,
    enable_infographics: bool,
    enable_audio: bool,
):
    text = None
    if element["document_type"] == "text" and enable_text:
        text = element["metadata"]["content"]
    elif element["document_type"] == "structured":
        text = element["metadata"]["table_metadata"]["table_content"]
        if element["metadata"]["content_metadata"]["subtype"] == "chart" and not enable_charts:
            text = None
        elif element["metadata"]["content_metadata"]["subtype"] == "table" and not enable_tables:
            text = None
        elif element["metadata"]["content_metadata"]["subtype"] == "infographic" and not enable_infographics:
            text = None
    elif element["document_type"] == "image" and enable_images:
        if element["metadata"]["content_metadata"]["subtype"] == "page_image":
            text = element["metadata"]["image_metadata"]["text"]
        else:
            text = element["metadata"]["image_metadata"]["caption"]
    elif element["document_type"] == "audio" and enable_audio:
        text = element["metadata"]["audio_metadata"]["audio_transcript"]
    verify_emb = verify_embedding(element)
    if not text or not verify_emb:
        source_name = element["metadata"]["source_metadata"]["source_name"]
        pg_num = element["metadata"]["content_metadata"].get("page_number", None)
        doc_type = element["document_type"]
        if not verify_emb:
            logger.debug(f"failed to find embedding for entity: {source_name} page: {pg_num} type: {doc_type}")
        if not text:
            logger.debug(f"failed to find text for entity: {source_name} page: {pg_num} type: {doc_type}")
        # if we do find text but no embedding remove anyway
        text = None
    if text and len(text) > 65535:
        logger.warning(
            f"Text is too long, skipping. It is advised to use SplitTask, to make smaller chunk sizes."
            f"text_length: {len(text)}, file_name: {element['metadata']['source_metadata'].get('source_name', None)} "
            f"page_number: {element['metadata']['content_metadata'].get('page_number', None)}"
        )
        text = None
    return text


def _insert_location_into_content_metadata(
    element,
    enable_charts: bool,
    enable_tables: bool,
    enable_images: bool,
    enable_infographic: bool,
):
    location = max_dimensions = None
    if element["document_type"] == "structured":
        location = element["metadata"]["table_metadata"]["table_location"]
        max_dimensions = element["metadata"]["table_metadata"]["table_location_max_dimensions"]
        if element["metadata"]["content_metadata"]["subtype"] == "chart" and not enable_charts:
            location = max_dimensions = None
        elif element["metadata"]["content_metadata"]["subtype"] == "table" and not enable_tables:
            location = max_dimensions = None
        elif element["metadata"]["content_metadata"]["subtype"] == "infographic" and not enable_infographic:
            location = max_dimensions = None
    elif element["document_type"] == "image" and enable_images:
        location = element["metadata"]["image_metadata"]["image_location"]
        max_dimensions = element["metadata"]["image_metadata"]["image_location_max_dimensions"]
    if (not location) and (element["document_type"] != "text"):
        source_name = element["metadata"]["source_metadata"]["source_name"]
        pg_num = element["metadata"]["content_metadata"].get("page_number")
        doc_type = element["document_type"]
        logger.info(f"failed to find location for entity: {source_name} page: {pg_num} type: {doc_type}")
        location = max_dimensions = None
    element["metadata"]["content_metadata"]["location"] = location
    element["metadata"]["content_metadata"]["max_dimensions"] = max_dimensions


def add_metadata(element, meta_dataframe, meta_source_field, meta_data_fields):
    element_name = element["metadata"]["source_metadata"]["source_name"]
    df = meta_dataframe[meta_dataframe[meta_source_field] == element_name]
    if df is None:
        logger.info(f"NO METADATA ENTRY found for {element_name}")
    if df.shape[0] > 1:
        logger.info(f"FOUND MORE THAN ONE metadata entry for {element_name}, will use first entry")
    meta_fields = df[meta_data_fields]
    for col in meta_data_fields:
        field = meta_fields.iloc[0][col]
        # catch any nan values
        if pd.isna(field):
            field = None
        elif isinstance(field, str):
            if field == "":
                field = None
            # this is specifically for lists
            elif field[0] == "[":
                field = ast.literal_eval(field)
        elif isinstance(field, (np.int32, np.int64)):
            field = int(field)
        elif isinstance(field, (np.float32, np.float64)):
            field = float(field)
        elif isinstance(field, (np.bool_)):
            field = bool(field)
        element["metadata"][col] = field


def write_records_minio(records, writer: RemoteBulkWriter) -> RemoteBulkWriter:
    """
    Writes the supplied records to milvus using the supplied writer.
    If a sparse model is supplied, it will be used to generate sparse
    embeddings to allow for hybrid search. Will filter records based on
    type, depending on what types are enabled via the boolean parameters.
    If the user sets the log level to info, any time a record fails
    ingestion, it will be reported to the user.

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
    enable_infographics : bool, optional
        When true, ensure all infographic type records are used.
    enable_audio : bool, optional
        When true, ensure all audio transcript type records are used.
    record_func : function, optional
        This function will be used to parse the records for necessary information.

    Returns
    -------
    RemoteBulkWriter
        Returns the writer supplied, with information related to minio records upload.
    """
    for element in records:
        writer.append_row(element)
    writer.commit()
    logger.debug(f"Wrote data to: {writer.batch_files}")
    return writer


def bulk_insert_milvus(
    collection_name: str,
    writer: RemoteBulkWriter,
    milvus_uri: str = "http://localhost:19530",
    minio_endpoint: str = "localhost:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    bucket_name: str = None,
    username: str = None,
    password: str = None,
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
        Milvus address with http(s) preffix and port. Can also be a file path, to activate
        milvus-lite.
    username : str, optional
        Milvus username.
    password : str, optional
        Milvus password.
    """
    connections.connect(uri=milvus_uri, token=f"{username}:{password}")
    t_bulk_start = time.time()
    task_ids = []

    for files in writer.batch_files:
        task_id = utility.do_bulk_insert(
            collection_name=collection_name,
            files=files,
            consistency_level=CONSISTENCY,
        )
        task_ids.append(task_id)

    while len(task_ids) > 0:
        time.sleep(1)
        tasks = copy.copy(task_ids)
        for task_id in tasks:
            task = utility.get_bulk_insert_state(task_id=task_id)
            state = task.state_name
            logger.info(f"Checking task: {task_id} - imported rows: {task.row_count}")
            if state == "Completed":
                logger.info(f"Task: {task_id}")
                logger.info(f"Start time: {task.create_time_str}")
                logger.info(f"Imported row count: {task.row_count}")
                task_ids.remove(task_id)
            if task.state == BulkInsertState.ImportFailed:
                logger.error(f"Task: {task_id}")
                logger.error(f"Failed reason: {task.failed_reason}")
                task_ids.remove(task_id)

    t_bulk_end = time.time()
    logger.info(f"Bulk {collection_name} upload took {t_bulk_end - t_bulk_start} s")


def create_bm25_model(
    records,
    enable_text: bool = True,
    enable_charts: bool = True,
    enable_tables: bool = True,
    enable_images: bool = True,
    enable_infographics: bool = True,
    enable_audio: bool = True,
) -> BM25EmbeddingFunction:
    """
    This function takes the input records and creates a corpus,
    factoring in filters (i.e. texts, charts, tables) and fits
    a BM25 model with that information. If the user sets the log
    level to info, any time a record fails ingestion, it will be
    reported to the user.

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
    enable_infographics : bool, optional
        When true, ensure all infographic type records are used.
    enable_audio : bool, optional
        When true, ensure all audio transcript type records are used.

    Returns
    -------
    BM25EmbeddingFunction
        Returns the model fitted to the selected corpus.
    """
    all_text = []
    for result in records:
        if isinstance(result, dict):
            result = [result]
        for element in result:
            text = _pull_text(
                element,
                enable_text,
                enable_charts,
                enable_tables,
                enable_images,
                enable_infographics,
                enable_audio,
            )
            if text:
                all_text.append(text)

    analyzer = build_default_analyzer(language="en")
    bm25_ef = BM25EmbeddingFunction(analyzer)

    bm25_ef.fit(all_text)
    return bm25_ef


def stream_insert_milvus(records, client: MilvusClient, collection_name: str, batch_size: int = 5000):
    """
    This function takes the input records and creates a corpus,
    factoring in filters (i.e. texts, charts, tables) and fits
    a BM25 model with that information. If the user sets the log
    level to info, any time a record fails ingestion, it will be
    reported to the user.

    Parameters
    ----------
    records : List
        List of chunks with attached metadata
    client : MilvusClient
        Milvus client instance
    collection_name : str
        Milvus Collection to search against
    """
    count = 0
    for idx in range(0, len(records), batch_size):
        client.insert(collection_name=collection_name, data=records[idx : idx + batch_size])
        count += len(records[idx : idx + batch_size])
    logger.info(f"streamed {count} records")


def wait_for_index(collection_name: str, num_elements: int, client: MilvusClient):
    """
    This function waits for the index to be built. It checks
    the indexed_rows of the index and waits for it to be equal
    to the number of records. This only works for streaming inserts,
    bulk inserts are not supported by this function
    (refer to MilvusClient.refresh_load for bulk inserts).
    """
    client.flush(collection_name)
    # index_names = utility.list_indexes(collection_name)
    indexed_rows = 0
    # observe dense_index, all indexes get populated simultaneously
    for index_name in [DENSE_INDEX_NAME]:
        indexed_rows = 0
        expected_rows = client.describe_index(collection_name, index_name)["indexed_rows"] + num_elements
        while indexed_rows < expected_rows:
            pos_movement = 10  # number of iteration allowed without noticing an increase in indexed_rows
            for i in range(20):
                current_indexed_rows = client.describe_index(collection_name, index_name)["indexed_rows"]
                time.sleep(1)
                logger.info(
                    f"Indexed rows, {collection_name}, {index_name} -  {current_indexed_rows} / {expected_rows}"
                )
                if current_indexed_rows == expected_rows:
                    indexed_rows = current_indexed_rows
                    break
                # check if indexed_rows is staying the same, too many times means something is wrong
                if current_indexed_rows == indexed_rows:
                    pos_movement -= 1
                else:
                    pos_movement = 10
                # if pos_movement is 0, raise an error, means the rows are not getting indexed as expected
                if pos_movement == 0:
                    raise ValueError(f"Rows are not getting indexed as expected for: {index_name} - {collection_name}")
                indexed_rows = current_indexed_rows
    return indexed_rows


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
    enable_infographics: bool = True,
    bm25_save_path: str = "bm25_model.json",
    compute_bm25_stats: bool = True,
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    bucket_name: str = None,
    threshold: int = 1000,
    meta_dataframe=None,
    meta_source_field=None,
    meta_fields=None,
    stream: bool = False,
    username: str = None,
    password: str = None,
    **kwargs,
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
    enable_infographics : bool, optional
        When true, ensure all infographic type records are used.
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
    stream : bool, optional
        When true, the records will be inserted into milvus using the stream insert method.
    username : str, optional
        Milvus username.
    password : str, optional
        Milvus password.
    """
    local_index = False
    connections.connect(uri=milvus_uri, token=f"{username}:{password}")
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
            enable_infographics=enable_infographics,
        )
        bm25_ef.save(bm25_save_path)
    elif local_index and sparse:
        bm25_ef = BM25EmbeddingFunction(build_default_analyzer(language="en"))
        bm25_ef.load(bm25_save_path)
    client = MilvusClient(milvus_uri, token=f"{username}:{password}")
    schema = Collection(collection_name).schema
    if isinstance(meta_dataframe, str):
        meta_dataframe = pandas_file_reader(meta_dataframe)
    cleaned_records = cleanup_records(
        records,
        enable_text=enable_text,
        enable_charts=enable_charts,
        enable_tables=enable_tables,
        enable_images=enable_images,
        enable_infographics=enable_infographics,
        meta_dataframe=meta_dataframe,
        meta_source_field=meta_source_field,
        meta_fields=meta_fields,
        sparse_model=bm25_ef,
    )
    num_elements = len(cleaned_records)
    if num_elements == 0:
        raise ValueError("No records with Embeddings to insert detected.")
    logger.info(f"{num_elements} elements to insert to milvus")
    logger.info(f"threshold for streaming is {threshold}")
    if num_elements < threshold:
        stream = True
    if stream:
        stream_insert_milvus(
            cleaned_records,
            client,
            collection_name,
        )
        if not local_index:
            # Make sure all rows are indexed, decided not to wrap in a timeout because we dont
            # know how long this should take, it is num_elements dependent.
            wait_for_index(collection_name, num_elements, client)
    else:
        minio_client = Minio(minio_endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        bucket_name = bucket_name if bucket_name else ClientConfigSchema().minio_bucket_name
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Connections parameters to access the remote bucket
        conn = RemoteBulkWriter.S3ConnectParam(
            endpoint=minio_endpoint,  # the default MinIO service started along with Milvus
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket_name,
            secure=False,
        )
        text_writer = RemoteBulkWriter(
            schema=schema,
            remote_path="/",
            connect_param=conn,
            file_type=BulkFileType.PARQUET,
        )
        writer = write_records_minio(
            cleaned_records,
            text_writer,
        )
        bulk_insert_milvus(
            collection_name,
            writer,
            milvus_uri,
            minio_endpoint,
            access_key,
            secret_key,
            bucket_name,
            username=username,
            password=password,
        )
        # fixes bulk insert lag time https://github.com/milvus-io/milvus/issues/21746
        client.refresh_load(collection_name)
        logger.info(f"Refresh load response: {client.get_load_state(collection_name)}")


def dense_retrieval(
    queries,
    collection_name: str,
    client: MilvusClient,
    dense_model,
    top_k: int,
    dense_field: str = "vector",
    output_fields: List[str] = ["text"],
    _filter: str = "",
    gpu_search: bool = False,
    local_index: bool = False,
    ef_param: int = 100,
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
    dense_model : Partial Function
        Partial function to generate dense embeddings with queries.
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
        # dense_embeddings.append(dense_model.get_query_embedding(query))
        dense_embeddings += dense_model([query])

    search_params = {}
    if not gpu_search and not local_index:
        search_params["params"] = {"ef": ef_param}

    results = client.search(
        collection_name=collection_name,
        data=dense_embeddings,
        anns_field=dense_field,
        limit=top_k,
        output_fields=output_fields,
        filter=_filter,
        consistency_level=CONSISTENCY,
        search_params=search_params,
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
    gpu_search: bool = False,
    local_index: bool = False,
    _filter: str = "",
    ef_param: int = 100,
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
        dense_embeddings += dense_model([query])
        if sparse_model:
            sparse_embeddings.append(_format_sparse_embedding(sparse_model.encode_queries([query])))
        else:
            sparse_embeddings.append(query)

    s_param_1 = {
        "metric_type": "L2",
    }
    if not gpu_search and not local_index:
        s_param_1["params"] = {"ef": ef_param}

    # Create search requests for both vector types
    search_param_1 = {
        "data": dense_embeddings,
        "anns_field": dense_field,
        "param": s_param_1,
        "limit": top_k,
        "expr": _filter,
    }

    dense_req = AnnSearchRequest(**search_param_1)
    s_param_2 = {"metric_type": "BM25"}
    if local_index:
        s_param_2 = {"metric_type": "IP", "params": {"drop_ratio_build": 0.0}}

    search_param_2 = {
        "data": sparse_embeddings,
        "anns_field": sparse_field,
        "param": s_param_2,
        "limit": top_k,
        "expr": _filter,
    }
    sparse_req = AnnSearchRequest(**search_param_2)

    results = client.hybrid_search(
        collection_name,
        [sparse_req, dense_req],
        RRFRanker(),
        limit=top_k,
        output_fields=output_fields,
        consistency_level=CONSISTENCY,
    )
    return results


def nvingest_retrieval(
    queries,
    collection_name: str = None,
    vdb_op: VDB = None,
    milvus_uri: str = "http://localhost:19530",
    top_k: int = 5,
    hybrid: bool = False,
    dense_field: str = "vector",
    sparse_field: str = "sparse",
    embedding_endpoint=None,
    sparse_model_filepath: str = "bm25_model.json",
    model_name: str = None,
    output_fields: List[str] = ["text", "source", "content_metadata"],
    gpu_search: bool = False,
    nv_ranker: bool = False,
    nv_ranker_endpoint: str = None,
    nv_ranker_model_name: str = None,
    nv_ranker_nvidia_api_key: str = None,
    nv_ranker_truncate: str = "END",
    nv_ranker_top_k: int = 50,
    nv_ranker_max_batch_size: int = 64,
    _filter: str = "",
    ef_param: int = 200,
    client: MilvusClient = None,
    username: str = None,
    password: str = None,
    **kwargs,
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
    nv_ranker : bool
        Set to True to use the nvidia reranker.
    nv_ranker_endpoint : str
        The endpoint to the nvidia reranker
    nv_ranker_model_name: str
        The name of the model host in the nvidia reranker
    nv_ranker_nvidia_api_key : str,
        The nvidia reranker api key, necessary when using non-local asset
    truncate : str [`END`, `NONE`]
        Truncate the incoming texts if length is longer than the model allows.
    nv_ranker_max_batch_size : int
        Max size for the number of candidates to rerank.
    nv_ranker_top_k : int,
        The number of candidates to return after reranking.
    client : MilvusClient, optional
        Milvus client instance.
    username : str, optional
        Milvus username.
    password : str, optional
        Milvus password.
    Returns
    -------
    List
        Nested list of top_k results per query.
    """
    if vdb_op is not None and not isinstance(vdb_op, VDB):
        raise ValueError("vdb_op must be a VDB object")
    if isinstance(vdb_op, VDB):
        kwargs = locals().copy()
        kwargs.pop("vdb_op", None)
        queries = kwargs.pop("queries", [])
        return vdb_op.retrieval(queries, **kwargs)

    client_config = ClientConfigSchema()
    nvidia_api_key = client_config.nvidia_api_key
    embedding_endpoint = embedding_endpoint if embedding_endpoint else client_config.embedding_nim_endpoint
    model_name = model_name if model_name else client_config.embedding_nim_model_name
    local_index = False
    embed_model = partial(
        infer_microservice,
        model_name=model_name,
        embedding_endpoint=embedding_endpoint,
        nvidia_api_key=nvidia_api_key,
        input_type="query",
        output_names=["embeddings"],
        grpc=not ("http" in urlparse(embedding_endpoint).scheme),
    )
    client = client or MilvusClient(milvus_uri, token=f"{username}:{password}")
    final_top_k = top_k
    if nv_ranker:
        top_k = nv_ranker_top_k
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
            _filter=_filter,
            ef_param=ef_param,
        )
    else:
        results = dense_retrieval(
            queries,
            collection_name,
            client,
            embed_model,
            top_k,
            output_fields=output_fields,
            _filter=_filter,
            gpu_search=gpu_search,
            local_index=local_index,
            ef_param=ef_param,
        )
    if nv_ranker:
        rerank_results = []
        for query, candidates in zip(queries, results):
            rerank_results.append(
                nv_rerank(
                    query,
                    candidates,
                    reranker_endpoint=nv_ranker_endpoint,
                    model_name=nv_ranker_model_name,
                    nvidia_api_key=nv_ranker_nvidia_api_key,
                    truncate=nv_ranker_truncate,
                    topk=final_top_k,
                    max_batch_size=nv_ranker_max_batch_size,
                )
            )
        results = rerank_results
    return results


def remove_records(
    source_name: str,
    collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    username: str = None,
    password: str = None,
    client: MilvusClient = None,
):
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
    client : MilvusClient, optional
        Milvus client instance.
    username : str, optional
        Milvus username.
    password : str, optional
        Milvus password.

    Returns
    -------
    Dict
        Dictionary with one key, `delete_cnt`. The value represents the number of entities
        removed.
    """
    client = client or MilvusClient(milvus_uri, token=f"{username}:{password}")
    result_ids = client.delete(
        collection_name=collection_name,
        filter=f'(source["source_name"] == "{source_name}")',
    )
    return result_ids


def nv_rerank(
    query,
    candidates,
    reranker_endpoint: str = None,
    model_name: str = None,
    nvidia_api_key: str = None,
    truncate: str = "END",
    max_batch_size: int = 64,
    topk: int = 5,
):
    """
    This function allows a user to rerank a set of candidates using the nvidia reranker nim.

    Parameters
    ----------
    query : str
        Query the candidates are supposed to answer.
    candidates : list
        List of the candidates to rerank.
    reranker_endpoint : str
        The endpoint to the nvidia reranker
    model_name: str
        The name of the model host in the nvidia reranker
    nvidia_api_key : str,
        The nvidia reranker api key, necessary when using non-local asset
    truncate : str [`END`, `NONE`]
        Truncate the incoming texts if length is longer than the model allows.
    max_batch_size : int
        Max size for the number of candidates to rerank.
    topk : int,
        The number of candidates to return after reranking.

    Returns
    -------
    Dict
        Dictionary with top_k reranked candidates.
    """
    client_config = ClientConfigSchema()
    # reranker = NVIDIARerank(base_url=reranker_endpoint, nvidia_api_key=nvidia_api_key, top_n=top_k)
    reranker_endpoint = reranker_endpoint if reranker_endpoint else client_config.nv_ranker_nim_endpoint
    model_name = model_name if model_name else client_config.nv_ranker_nim_model_name
    nvidia_api_key = nvidia_api_key if nvidia_api_key else client_config.nvidia_api_key
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    if nvidia_api_key:
        headers["Authorization"] = f"Bearer {nvidia_api_key}"
    texts = []
    map_candidates = {}
    for idx, candidate in enumerate(candidates):
        map_candidates[idx] = candidate
        texts.append({"text": candidate["entity"]["text"]})
    payload = {
        "model": model_name,
        "query": {"text": query},
        "passages": texts,
        "truncate": truncate,
    }
    start = time.time()
    response = requests.post(f"{reranker_endpoint}", headers=headers, json=payload)
    logger.debug(f"RERANKER time: {time.time() - start}")
    if response.status_code != 200:
        raise ValueError(f"Failed retrieving ranking results: {response.status_code} - {response.text}")
    rank_results = []
    for rank_vals in response.json()["rankings"]:
        idx = rank_vals["index"]
        rank_results.append(map_candidates[idx])
    return rank_results


def recreate_elements(data):
    """
    This function takes the input data and creates a list of elements
    with the necessary metadata for ingestion.

    Parameters
    ----------
    data : List
        List of chunks with attached metadata

    Returns
    -------
    List
        List of elements with metadata.
    """
    elements = []
    for element in data:
        element["metadata"] = {}
        element["metadata"]["content_metadata"] = element.pop("content_metadata")
        element["metadata"]["source_metadata"] = element.pop("source")
        element["metadata"]["content"] = element.pop("text")
        elements.append(element)
    return elements


def pull_all_milvus(
    collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    write_dir: str = None,
    batch_size: int = 1000,
    include_embeddings: bool = False,
    username: str = None,
    password: str = None,
    client: MilvusClient = None,
):
    """
    This function takes the input collection name and pulls all the records
    from the collection. It will either return the records as a list of
    dictionaries or write them to a specified directory in JSON format.
    Parameters
    ----------
    collection_name : str
        Milvus Collection to query against
    milvus_uri : str,
        Milvus address with http(s) preffix and port. Can also be a file path, to activate
        milvus-lite.
    write_dir : str, optional
        Directory to write the records to. If None, the records will be returned as a list.
    batch_size : int, optional
        The number of records to pull in each batch. Defaults to 1000.
    include_embeddings : bool, optional
        Whether to include the embeddings in the output. Defaults to False.
    username : str, optional
        Milvus username.
    password : str, optional
        Milvus password.
    client : MilvusClient, optional
        Milvus client instance.
    Returns
    -------
    List
        List of records/files with records from the collection.
    """
    client = client or MilvusClient(milvus_uri, token=f"{username}:{password}")
    output_fields = ["source", "content_metadata", "text"]
    if include_embeddings:
        output_fields.append("vector")
    iterator = client.query_iterator(
        collection_name=collection_name,
        filter="pk >= 0",
        output_fields=output_fields,
        batch_size=batch_size,
        consistency_level=CONSISTENCY,
    )
    full_results = []
    write_dir = Path(write_dir) if write_dir else None
    batch_num = 0
    while True:
        results = recreate_elements(iterator.next())
        if not results:
            iterator.close()
            break
        if write_dir:
            # write to disk
            file_name = write_dir / f"milvus_data_{batch_num}.json"
            full_results.append(file_name)
            with open(file_name, "w") as outfile:
                outfile.write(json.dumps(results))
        else:
            full_results += results
        batch_num += 1
    return full_results


def get_embeddings(full_records, embedder, batch_size=256):
    """
    This function takes the input records and creates a list of embeddings.
    The default batch size is 256, but this can be adjusted based on the
    available resources, to a maximum of 259. This is set by the NVIDIA embedding
    microservice.
    """
    embedded = []
    embed_payload = [res["metadata"]["content"] for res in full_records]
    for i in range(0, len(embed_payload), batch_size):
        payload = embed_payload[i : i + batch_size]
        embedded += embedder._get_text_embeddings(payload)
    return embedded


def embed_index_collection(
    data,
    collection_name,
    batch_size: int = 256,
    embedding_endpoint: str = None,
    model_name: str = None,
    nvidia_api_key: str = None,
    milvus_uri: str = "http://localhost:19530",
    sparse: bool = False,
    recreate: bool = True,
    gpu_index: bool = True,
    gpu_search: bool = False,
    dense_dim: int = 2048,
    minio_endpoint: str = "localhost:9000",
    enable_text: bool = True,
    enable_charts: bool = True,
    enable_tables: bool = True,
    enable_images: bool = True,
    enable_infographics: bool = True,
    bm25_save_path: str = "bm25_model.json",
    compute_bm25_stats: bool = True,
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    bucket_name: str = None,
    meta_dataframe: Union[str, pd.DataFrame] = None,
    meta_source_field: str = None,
    meta_fields: list[str] = None,
    input_type: str = "passage",
    truncate: str = "END",
    client: MilvusClient = None,
    username: str = None,
    password: str = None,
    **kwargs,
):
    """
    This function takes the input data and creates a collection in Milvus,
    it will embed the records using the NVIDIA embedding model and store them in the collection.
    After embedding the records, it will run the same ingestion process as the vdb_upload stage in
    the Ingestor pipeline.

    Args:
        data (Union[str, List]): The data to be ingested. Can be a list of records or a file path.
        collection_name (Union[str, Dict], optional): The name of the Milvus collection or a dictionary
            containing collection configuration. Defaults to "nv_ingest_collection".
        embedding_endpoint (str, optional): The endpoint for the NVIDIA embedding service. Defaults to None.
        model_name (str, optional): The name of the embedding model. Defaults to None.
        nvidia_api_key (str, optional): The API key for NVIDIA services. Defaults to None.
        milvus_uri (str, optional): The URI of the Milvus server. Defaults to "http://localhost:19530".
        sparse (bool, optional): Whether to use sparse indexing. Defaults to False.
        recreate (bool, optional): Whether to recreate the collection if it already exists. Defaults to True.
        gpu_index (bool, optional): Whether to use GPU for indexing. Defaults to True.
        gpu_search (bool, optional): Whether to use GPU for search operations. Defaults to True.
        dense_dim (int, optional): The dimensionality of dense vectors. Defaults to 2048.
        minio_endpoint (str, optional): The endpoint for the MinIO server. Defaults to "localhost:9000".
        enable_text (bool, optional): Whether to enable text data ingestion. Defaults to True.
        enable_charts (bool, optional): Whether to enable chart data ingestion. Defaults to True.
        enable_tables (bool, optional): Whether to enable table data ingestion. Defaults to True.
        enable_images (bool, optional): Whether to enable image data ingestion. Defaults to True.
        enable_infographics (bool, optional): Whether to enable infographic data ingestion. Defaults to True.
        bm25_save_path (str, optional): The file path to save the BM25 model. Defaults to "bm25_model.json".
        compute_bm25_stats (bool, optional): Whether to compute BM25 statistics. Defaults to True.
        access_key (str, optional): The access key for MinIO authentication. Defaults to "minioadmin".
        secret_key (str, optional): The secret key for MinIO authentication. Defaults to "minioadmin".
        bucket_name (str, optional): The name of the MinIO bucket.
        meta_dataframe (Union[str, pd.DataFrame], optional): A metadata DataFrame or the path to a CSV file
            containing metadata. Defaults to None.
        meta_source_field (str, optional): The field in the metadata that serves as the source identifier.
            Defaults to None.
        meta_fields (list[str], optional): A list of metadata fields to include. Defaults to None.
        client : MilvusClient, optional
            Milvus client instance.
        username : str, optional
            Milvus username.
        password : str, optional
            Milvus password.
        **kwargs: Additional keyword arguments for customization.
    """
    client_config = ClientConfigSchema()
    nvidia_api_key = nvidia_api_key if nvidia_api_key else client_config.nvidia_api_key
    embedding_endpoint = embedding_endpoint if embedding_endpoint else client_config.embedding_nim_endpoint
    model_name = model_name if model_name else client_config.embedding_nim_model_name
    # if not scheme we assume we are using grpc
    grpc = "http" not in urlparse(embedding_endpoint).scheme
    kwargs.pop("input_type", None)
    kwargs.pop("truncate", None)
    mil_op = Milvus(
        collection_name=collection_name,
        milvus_uri=milvus_uri,
        sparse=sparse,
        recreate=recreate,
        gpu_index=gpu_index,
        gpu_search=gpu_search,
        dense_dim=dense_dim,
        minio_endpoint=minio_endpoint,
        enable_text=enable_text,
        enable_charts=enable_charts,
        enable_tables=enable_tables,
        enable_images=enable_images,
        enable_infographics=enable_infographics,
        bm25_save_path=bm25_save_path,
        compute_bm25_stats=compute_bm25_stats,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
        meta_dataframe=meta_dataframe,
        meta_source_field=meta_source_field,
        meta_fields=meta_fields,
        username=username,
        password=password,
        **kwargs,
    )
    # running in parts
    if data is not None and isinstance(data[0], (str, os.PathLike)):
        for results_file in data:
            results = None
            with open(results_file, "r") as infile:
                results = json.loads(infile.read())
                embeddings = infer_microservice(
                    results,
                    model_name,
                    embedding_endpoint,
                    nvidia_api_key,
                    input_type,
                    truncate,
                    batch_size,
                    grpc,
                )
            for record, emb in zip(results, embeddings):
                record["metadata"]["embedding"] = emb
                record["document_type"] = "text"
            if results is not None and len(results) > 0:
                mil_op.run(results)
                mil_op.milvus_kwargs["recreate"] = False
    # running all at once
    else:
        embeddings = infer_microservice(
            data,
            model_name,
            embedding_endpoint,
            nvidia_api_key,
            input_type,
            truncate,
            batch_size,
            grpc,
        )
        for record, emb in zip(data, embeddings):
            record["metadata"]["embedding"] = emb
            record["document_type"] = "text"
        # this check ensures that we do not purge the current collection
        # without having some actual data to insert.
        if data is not None and len(data) > 0:
            mil_op.run(data)


def reindex_collection(
    vdb_op: VDB = None,
    collection_name: str = None,
    new_collection_name: str = None,
    write_dir: str = None,
    embedding_endpoint: str = None,
    model_name: str = None,
    nvidia_api_key: str = None,
    milvus_uri: str = "http://localhost:19530",
    sparse: bool = False,
    recreate: bool = True,
    gpu_index: bool = True,
    gpu_search: bool = False,
    dense_dim: int = 2048,
    minio_endpoint: str = "localhost:9000",
    enable_text: bool = True,
    enable_charts: bool = True,
    enable_tables: bool = True,
    enable_images: bool = True,
    enable_infographics: bool = True,
    bm25_save_path: str = "bm25_model.json",
    compute_bm25_stats: bool = True,
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    bucket_name: str = None,
    meta_dataframe: Union[str, pd.DataFrame] = None,
    meta_source_field: str = None,
    meta_fields: list[str] = None,
    embed_batch_size: int = 256,
    query_batch_size: int = 1000,
    input_type: str = "passage",
    truncate: str = "END",
    **kwargs,
):
    """
    This function will reindex a collection in Milvus. It will pull all the records from the
    current collection, embed them using the NVIDIA embedding model, and store them in a new
    collection. After embedding the records, it will run the same ingestion process as the vdb_upload
    stage in the Ingestor pipeline. This function will get embedding_endpoint, model_name and nvidia_api_key
    defaults from the environment variables set in the environment if not explicitly set in the function call.

    Parameters
    ----------
        collection_name (str): The name of the current Milvus collection.
        new_collection_name (str, optional): The name of the new Milvus collection. Defaults to None.
        write_dir (str, optional): The directory to write the pulled records to. Defaults to None.
        embedding_endpoint (str, optional): The endpoint for the NVIDIA embedding service. Defaults to None.
        model_name (str, optional): The name of the embedding model. Defaults to None.
        nvidia_api_key (str, optional): The API key for NVIDIA services. Defaults to None.
        milvus_uri (str, optional): The URI of the Milvus server. Defaults to "http://localhost:19530".
        sparse (bool, optional): Whether to use sparse indexing. Defaults to False.
        recreate (bool, optional): Whether to recreate the collection if it already exists. Defaults to True.
        gpu_index (bool, optional): Whether to use GPU for indexing. Defaults to True.
        gpu_search (bool, optional): Whether to use GPU for search operations. Defaults to True.
        dense_dim (int, optional): The dimensionality of dense vectors. Defaults to 2048.
        minio_endpoint (str, optional): The endpoint for the MinIO server. Defaults to "localhost:9000".
        enable_text (bool, optional): Whether to enable text data ingestion. Defaults to True.
        enable_charts (bool, optional): Whether to enable chart data ingestion. Defaults to True.
        enable_tables (bool, optional): Whether to enable table data ingestion. Defaults to True.
        enable_images (bool, optional): Whether to enable image data ingestion. Defaults to True.
        enable_infographics (bool, optional): Whether to enable infographic data ingestion. Defaults to True.
        bm25_save_path (str, optional): The file path to save the BM25 model. Defaults to "bm25_model.json".
        compute_bm25_stats (bool, optional): Whether to compute BM25 statistics. Defaults to True.
        access_key (str, optional): The access key for MinIO authentication. Defaults to "minioadmin".
        secret_key (str, optional): The secret key for MinIO authentication. Defaults to "minioadmin".
        bucket_name (str, optional): The name of the MinIO bucket.
        meta_dataframe (Union[str, pd.DataFrame], optional): A metadata DataFrame or the path to a CSV file
            containing metadata. Defaults to None.
        meta_source_field (str, optional): The field in the metadata that serves as the source identifier.
            Defaults to None.
        meta_fields (list[str], optional): A list of metadata fields to include. Defaults to None.
        embed_batch_size (int, optional): The batch size for embedding. Defaults to 256.
        query_batch_size (int, optional): The batch size for querying. Defaults to 1000.
        **kwargs: Additional keyword arguments for customization.
    """
    if vdb_op is not None and not isinstance(vdb_op, VDB):
        raise ValueError("vdb_op must be a VDB object")
    if isinstance(vdb_op, VDB):
        kwargs = locals().copy()
        kwargs.pop("vdb_op", None)
        return vdb_op.reindex(**kwargs)
    new_collection_name = new_collection_name if new_collection_name else collection_name
    pull_results = pull_all_milvus(collection_name, milvus_uri, write_dir, query_batch_size)
    embed_index_collection(
        pull_results,
        new_collection_name,
        batch_size=embed_batch_size,
        embedding_endpoint=embedding_endpoint,
        model_name=model_name,
        nvidia_api_key=nvidia_api_key,
        milvus_uri=milvus_uri,
        sparse=sparse,
        recreate=recreate,
        gpu_index=gpu_index,
        gpu_search=gpu_search,
        dense_dim=dense_dim,
        minio_endpoint=minio_endpoint,
        enable_text=enable_text,
        enable_charts=enable_charts,
        enable_tables=enable_tables,
        enable_images=enable_images,
        enable_infographics=enable_infographics,
        bm25_save_path=bm25_save_path,
        compute_bm25_stats=compute_bm25_stats,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
        meta_dataframe=meta_dataframe,
        meta_source_field=meta_source_field,
        meta_fields=meta_fields,
        input_type=input_type,
        truncate=truncate,
        **kwargs,
    )


def reconstruct_pages(anchor_record, records_list, page_signum: int = 0):
    """
    This function allows a user reconstruct the pages for a retrieved chunk.

    Parameters
    ----------
    anchor_record : dict
        Query the candidates are supposed to answer.
    records_list : list
        List of the candidates to rerank.
    page_signum : int
        The endpoint to the nvidia reranker

    Returns
    -------
    String
        Full page(s) corresponding to anchor record.
    """

    source_file = anchor_record["entity"]["source"]["source_name"]
    page_number = anchor_record["entity"]["content_metadata"]["page_number"]
    min_page = page_number - page_signum
    max_page = page_number + 1 + page_signum
    page_numbers = list(range(min_page, max_page))

    target_records = []
    for sub_records in records_list:
        for record in sub_records:
            rec_src_file = record["metadata"]["source_metadata"]["source_name"]
            rec_pg_num = record["metadata"]["content_metadata"]["page_number"]
            if source_file == rec_src_file and rec_pg_num in page_numbers:
                target_records.append(record)

    return ingest_json_results_to_blob(target_records)


class Milvus(VDB):

    def __init__(
        self,
        collection_name: Union[str, Dict] = "nv_ingest_collection",
        milvus_uri: str = "http://localhost:19530",
        sparse: bool = False,
        recreate: bool = True,
        gpu_index: bool = True,
        gpu_search: bool = False,
        dense_dim: int = 2048,
        minio_endpoint: str = "localhost:9000",
        enable_text: bool = True,
        enable_charts: bool = True,
        enable_tables: bool = True,
        enable_images: bool = True,
        enable_infographics: bool = True,
        bm25_save_path: str = "bm25_model.json",
        compute_bm25_stats: bool = True,
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket_name: str = None,
        meta_dataframe: Union[str, pd.DataFrame] = None,
        meta_source_field: str = None,
        meta_fields: list[str] = None,
        stream: bool = False,
        threshold: int = 1000,
        username: str = None,
        password: str = None,
        **kwargs,
    ):
        """
        Initializes the Milvus operator class with the specified configuration parameters.
        Args:
            collection_name (Union[str, Dict], optional): The name of the Milvus collection or a dictionary
                containing collection configuration. Defaults to "nv_ingest_collection".
            milvus_uri (str, optional): The URI of the Milvus server. Defaults to "http://localhost:19530".
            sparse (bool, optional): Whether to use sparse indexing. Defaults to False.
            recreate (bool, optional): Whether to recreate the collection if it already exists. Defaults to True.
            gpu_index (bool, optional): Whether to use GPU for indexing. Defaults to True.
            gpu_search (bool, optional): Whether to use GPU for search operations. Defaults to True.
            dense_dim (int, optional): The dimensionality of dense vectors. Defaults to 2048.
            minio_endpoint (str, optional): The endpoint for the MinIO server. Defaults to "localhost:9000".
            enable_text (bool, optional): Whether to enable text data ingestion. Defaults to True.
            enable_charts (bool, optional): Whether to enable chart data ingestion. Defaults to True.
            enable_tables (bool, optional): Whether to enable table data ingestion. Defaults to True.
            enable_images (bool, optional): Whether to enable image data ingestion. Defaults to True.
            enable_infographics (bool, optional): Whether to enable infographic data ingestion. Defaults to True.
            bm25_save_path (str, optional): The file path to save the BM25 model. Defaults to "bm25_model.json".
            compute_bm25_stats (bool, optional): Whether to compute BM25 statistics. Defaults to True.
            access_key (str, optional): The access key for MinIO authentication. Defaults to "minioadmin".
            secret_key (str, optional): The secret key for MinIO authentication. Defaults to "minioadmin".
            bucket_name (str, optional): The name of the MinIO bucket.
            meta_dataframe (Union[str, pd.DataFrame], optional): A metadata DataFrame or the path to a CSV file
                containing metadata. Defaults to None.
            meta_source_field (str, optional): The field in the metadata that serves as the source identifier.
                Defaults to None.
            meta_fields (list[str], optional): A list of metadata fields to include. Defaults to None.
            stream (bool, optional): When true, the records will be inserted into milvus using the stream
                insert method.
            username (str, optional): The username for Milvus authentication. Defaults to None.
            password (str, optional): The password for Milvus authentication. Defaults to None.
            **kwargs: Additional keyword arguments for customization.
        """
        kwargs = locals().copy()
        kwargs.pop("self", None)
        super().__init__(**kwargs)

    def create_index(self, **kwargs):
        collection_name = kwargs.pop("collection_name")
        return create_nvingest_collection(collection_name, **kwargs)

    def write_to_index(self, records, **kwargs):
        collection_name = kwargs.pop("collection_name")
        write_to_nvingest_collection(records, collection_name=collection_name, **kwargs)

    def retrieval(self, queries, **kwargs):
        collection_name = kwargs.pop("collection_name")
        return nvingest_retrieval(queries, collection_name=collection_name, **kwargs)

    def reindex(self, **kwargs):
        collection_name = kwargs.pop("current_collection_name")
        reindex_collection(current_collection_name=collection_name, **kwargs)

    def get_connection_params(self):
        conn_dict = {
            "milvus_uri": self.__dict__.get("milvus_uri", "http://localhost:19530"),
            "sparse": self.__dict__.get("sparse", True),
            "recreate": self.__dict__.get("recreate", True),
            "gpu_index": self.__dict__.get("gpu_index", True),
            "gpu_search": self.__dict__.get("gpu_search", True),
            "dense_dim": self.__dict__.get("dense_dim", 2048),
            "username": self.__dict__.get("username", None),
            "password": self.__dict__.get("password", None),
        }
        return (self.collection_name, conn_dict)

    def get_write_params(self):
        write_params = self.__dict__.copy()
        write_params.pop("recreate", True)
        write_params.pop("gpu_index", True)
        write_params.pop("gpu_search", True)
        write_params.pop("dense_dim", 2048)

        return (self.collection_name, write_params)

    def run(self, records):
        collection_name, create_params = self.get_connection_params()
        _, write_params = self.get_write_params()
        if isinstance(collection_name, str):
            self.create_index(collection_name=collection_name, **create_params)
            self.write_to_index(records, **write_params)
        elif isinstance(collection_name, dict):
            split_params_list = _dict_to_params(collection_name, write_params)
            for sub_params in split_params_list:
                coll_name, sub_write_params = sub_params
                sub_write_params.pop("collection_name", None)
                self.create_index(collection_name=coll_name, **create_params)
                self.write_to_index(records, collection_name=coll_name, **sub_write_params)
        else:
            raise ValueError(f"Unsupported type for collection_name detected: {type(collection_name)}")
        return records

    def run_async(self, records):
        collection_name, create_params = self.get_connection_params()
        _, write_params = self.get_write_params()
        if isinstance(collection_name, str):
            logger.info(f"creating index - {collection_name}")
            self.create_index(collection_name=collection_name, **create_params)
            records = records.result()
            logger.info(f"writing to index, for collection - {collection_name}")
            self.write_to_index(records, **write_params)
        elif isinstance(collection_name, dict):
            split_params_list = _dict_to_params(collection_name, write_params)
            for sub_params in split_params_list:
                coll_name, sub_write_params = sub_params
                sub_write_params.pop("collection_name", None)
                self.create_index(collection_name=coll_name, **create_params)
                self.write_to_index(records, collection_name=coll_name, **sub_write_params)
        else:
            raise ValueError(f"Unsupported type for collection_name detected: {type(collection_name)}")
        return records
