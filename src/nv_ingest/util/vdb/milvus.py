# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# pylint: skip-file

import json
import os

from fastapi import HTTPException
from pymilvus import Collection, MilvusClient, DataType, connections, utility, BulkInsertState
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType
from typing import List


# We are using UUIDs for the collection_name. However, Milvus collection names
# cannot have `-` characters so we replace them with `_`
def reformat_collection_name(collection_name: str = None):
    collection_name = collection_name.replace("-", "_")
    # The start of a collection name must be a "_" or character. Since UUID could start
    # with number we just prepend a "_" if not already there
    if not collection_name.startswith("_"):
        collection_name = f"_{collection_name}"
    return collection_name


# Connect to Milvus
connections.connect("default", host="milvus", port="19530")  # Update with your Milvus host/port


def search_milvus(question_embedding: List[float], top_k: int = 5, collection_name: str = "nv_ingest_jeremy"):

    # Reformat to remove "-" characters from UUID
    collection_name = reformat_collection_name(collection_name)

    """Query Milvus for nearest neighbors of the question embedding."""
    print(f"Searching for milvus collection: {collection_name}")
    if not utility.has_collection(collection_name):
        raise HTTPException(status_code=500, detail=f"Milvus collection '{collection_name}' not found.")

    collection = Collection(collection_name)
    index_params = {
        "metric_type": "L2",
        "index_type": "GPU_CAGRA",
        "params": {
            "intermediate_graph_degree": 128,
            "graph_degree": 64,
            "build_algo": "NN_DESCENT",
        },
    }
    collection.create_index("vector", index_params)
    collection.load()

    results = collection.search(
        data=[question_embedding],
        anns_field="vector",  # Vector field in your collection
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"],  # Update based on your schema
    )
    return [hit.entity.get("text") for hit in results[0]]


def bulk_upload_results_to_milvus(ingest_results, collection_name: str = "nv_ingest_jeremy"):
    collection_name = reformat_collection_name(collection_name)
    milvus_url = os.getenv("MILVUS_ENDPOINT", "http://milvus:19530")
    minio_url = os.getenv("MINIO_ENDPOINT", "minio:9000")

    print(f"Milvus URL: {milvus_url}")
    client = MilvusClient(milvus_url)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="source", datatype=DataType.JSON)
    schema.add_field(field_name="content_metadata", datatype=DataType.JSON)

    client.create_collection(collection_name=collection_name, schema=schema)

    ACCESS_KEY = "minioadmin"
    SECRET_KEY = "minioadmin"
    BUCKET_NAME = "a-bucket"

    # Connections parameters to access the remote bucket
    conn = RemoteBulkWriter.S3ConnectParam(
        endpoint=minio_url,  # the default MinIO service started along with Milvus
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        bucket_name=BUCKET_NAME,
        secure=False,
    )

    remote_writer = RemoteBulkWriter(schema=schema, remote_path="/", connect_param=conn, file_type=BulkFileType.JSON)

    for file_result in ingest_results:
        # Convert the JSON str to a python dict
        file_result = json.loads(file_result)

        # The 'data' key of the full file results contains the list of results for each page in the PDF
        page_results = file_result["data"]
        num_none_types = 0
        num_non_none_types = 0
        num_total = 0
        for doc_element in page_results:
            text = None
            num_total = num_total + 1

            # Itereate through each page and prepare the data for the VDB
            if doc_element["document_type"] == "text":
                text = doc_element["metadata"]["content"]
            elif doc_element["document_type"] == "structured":
                text = doc_element["metadata"]["table_metadata"]["table_content"]

            if text is None:
                num_none_types = num_none_types + 1
            else:
                num_non_none_types = num_non_none_types + 1
                if len(text) >= 65535:
                    print(f"TEXT IS WAY TOO LONG!!!! Length: {len(text)}")

                remote_writer.append_row(
                    {
                        "text": text,
                        "vector": doc_element["metadata"]["embedding"],
                        "source": doc_element["metadata"]["source_metadata"],
                        "content_metadata": doc_element["metadata"]["content_metadata"],
                    }
                )

        print(f"{num_none_types}/{num_total} text elements were None. {num_non_none_types} actually had content.")

    remote_writer.commit()
    print(f"Wrote data to: {remote_writer.batch_files}")

    print(f"Bulk uploading: {remote_writer.batch_files[0]} to Milvus")

    # Bulk upload the Minio data to Milvus
    task_id = utility.do_bulk_insert(
        collection_name=collection_name,
        files=remote_writer.batch_files[0],
    )
    print(f"Milvus Task_id: {task_id} of type: {type(task_id)}")
    return task_id


def check_bulk_upload_status(task_id: str) -> bool:
    if isinstance(task_id, bytes):
        task_id = int(task_id.decode("utf-8"))  # Decode to string, then convert to int
    elif isinstance(task_id, str):
        task_id = int(task_id)  # Directly convert to int
    else:
        raise ValueError("task_id must be bytes or str")

    task = utility.get_bulk_insert_state(task_id=task_id)
    state = task.state_name
    print("Task state:", task.state_name)
    print("Imported files:", task.files)
    print("Collection name:", task.collection_name)
    print("Partition name:", task.partition_name)
    print("Start time:", task.create_time_str)
    print("Imported row count:", task.row_count)
    print("Entities ID array generated by this task:", task.ids)
    if state == "Completed":
        print("Start time:", task.create_time_str)
        print("Imported row count:", task.row_count)
        return True
    if task.state == BulkInsertState.ImportFailed:
        print("Failed reason:", task.failed_reason)
        raise HTTPException(
            status_code=500,
            detail=f"Milvus bulk upload for collection '{task.collection_name}' failed. {task.failed_reason}",
        )

    return False
