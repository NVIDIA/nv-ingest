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
import time

from fastapi import HTTPException
from pymilvus import Collection, MilvusClient, DataType, connections, utility, BulkInsertState
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType
from typing import List

# Connect to Milvus
connections.connect("default", host="milvus", port="19530")  # Update with your Milvus host/port

def search_milvus(question_embedding: List[float], top_k: int = 5):
    
    COLLECTION_NAME = "nv_ingest_jeremy"  # Update to your Milvus collection name
    
    """Query Milvus for nearest neighbors of the question embedding."""
    print(f"Searching for milvus collection: {COLLECTION_NAME}")
    if not utility.has_collection(COLLECTION_NAME):
        raise HTTPException(status_code=500, detail=f"Milvus collection '{COLLECTION_NAME}' not found.")
    
    collection = Collection(COLLECTION_NAME)
    results = collection.search(
        data=[question_embedding],
        anns_field="vector",  # Vector field in your collection
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["content"]  # Update based on your schema
    )
    return [hit.entity.get("content") for hit in results[0]]

def bulk_upload_results_to_milvus(ingest_results, collection_name: str = "nv_ingest_jeremy"):
    milvus_url = os.getenv("MILVUS_ENDPOINT", "http://milvus:19530")
    minio_url = os.getenv("MINIO_ENDPOINT", "minio:9000")
    
    print(f"Milvus URL: {milvus_url}")
    client = MilvusClient(milvus_url)
    
    
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="source", datatype=DataType.JSON)
    schema.add_field(field_name="content_metadata", datatype=DataType.JSON)

    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )

    ACCESS_KEY="minioadmin"
    SECRET_KEY="minioadmin"
    BUCKET_NAME="a-bucket"

    # Connections parameters to access the remote bucket
    conn = RemoteBulkWriter.S3ConnectParam(
        endpoint=minio_url, # the default MinIO service started along with Milvus
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        bucket_name=BUCKET_NAME,
        secure=False
    )

    remote_writer = RemoteBulkWriter(
        schema=schema,
        remote_path="/",
        connect_param=conn,
        file_type=BulkFileType.JSON
    )
    
    for file_result in ingest_results:
        # Convert the JSON str to a python dict
        file_result = json.loads(file_result)

        # The 'data' key of the full file results contains the list of results for each page in the PDF
        page_results = file_result['data']
        num_none_types = 0
        num_non_none_types = 0
        num_total = 0
        for doc_element in page_results:
            text = None
            num_total = num_total + 1
            
            # Itereate through each page and prepare the data for the VDB
            if doc_element['document_type'] == 'text':
                text = doc_element['metadata']['content']
            elif doc_element['document_type'] == 'structured':
                text = doc_element['metadata']['table_metadata']['table_content']
                
            
            if text == None:
                num_none_types = num_none_types + 1
            else:
                num_non_none_types = num_non_none_types + 1
                if len(text) >= 65535:
                    print(f"TEXT IS WAY TOO LONG!!!! Length: {len(text)}")
                    
                remote_writer.append_row({
                    "text": text,
                    "vector": doc_element['metadata']['embedding'],
                    "source": doc_element['metadata']['source_metadata'],
                    "content_metadata":  doc_element['metadata']['content_metadata'],
                })
    
        print(f"{num_none_types}/{num_total} text elements were None. {num_non_none_types} actually had content.")
    
    remote_writer.commit()
    print(f"Wrote data to: {remote_writer.batch_files}")
    
    print(f"Bulk uploading: {remote_writer.batch_files[0]} to Milvus")
    t_bulk_start = time.time()
    
    # Bulk upload the Minio data to Milvus
    task_id = utility.do_bulk_insert(
        collection_name=collection_name,
        files=remote_writer.batch_files[0],
    )
    
    state = "Pending"
    while state != "Completed":
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
            t_bulk_end = time.time()
            print("Start time:", task.create_time_str)
            print("Imported row count:", task.row_count)
            print(f"Bulk upload took: {t_bulk_end - t_bulk_start} s")
        if task.state == BulkInsertState.ImportFailed:
            print("Failed reason:", task.failed_reason)
        time.sleep(1)
    
    print(f"Bulk ingest into Milvus complete!")
