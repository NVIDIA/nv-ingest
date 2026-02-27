# Data Upload for NeMo Retriever Extraction

Use this documentation to learn how [NeMo Retriever Library](overview.md) handles and uploads data.

!!! note

    This library is the NeMo Retriever Library.


## Overview

NeMo Retriever Library supports extracting text representations of various forms of content, 
and ingesting to the [Milvus vector database](https://milvus.io/). 
The data upload task (`vdb_upload`) pulls extraction results to the Python client, 
and then pushes them to Milvus by using its underlying MinIO object store service.

The vector database stores only the extracted text representations of ingested data. 
It does not store the embeddings for images.

!!! tip "Storing Extracted Images"

    To persist extracted images, tables, and chart renderings to disk or object storage, use the `store` task in addition to `vdb_upload`. The `store` task supports any fsspec-compatible backend (local filesystem, S3, GCS, etc.). For details, refer to [Store Extracted Images](python-api-reference.md#store-extracted-images).

NeMo Retriever Library supports uploading data by using the [Ingestor.vdb_upload API](python-api-reference.md). 
Currently, data upload is not supported through the [NeMo Retriever CLI](cli-reference.md).



## Upload to Milvus

The `vdb_upload` method uses GPU Cagra accelerated bulk indexing support to load chunks into Milvus. 
To enable hybrid retrieval, NeMo Retriever supports both dense (llama-embedder embeddings) and sparse (bm25) embeddings. 

Bulk indexing is high throughput, but has a built-in overhead of around one minute. 
If the number of ingested documents is 10 or fewer, NeMo Retriever uses faster streaming inserts instead. 
You can control this by setting `stream=True`. 

If you set `recreate=True`, NeMo Retriever drops and recreates the collection given as *collection_name*. 
The Milvus service persists data to disk by using a Docker volume defined in docker-compose.yaml. 
You can delete all collections by deleting that volume, and then restarting the NeMo Retriever service.

!!! warning

    When you use the `vdb_upload` task with Milvus, you must expose the ports for the Milvus and MinIO containers to the NeMo Retriever client. This ensures that the NeMo Retriever client can connect to both services and perform the `vdb_upload` action.

!!! tip

    When you use the `vdb_upload` method, the behavior of the upload depends on the `return_failures` parameter of the `ingest` method. For details, refer to [Capture Job Failures](python-api-reference.md#capture-job-failures).

To upload to Milvus, use code similar to the following to define your `Ingestor`.

```python
Ingestor(client=client)
    .files("data/multimodal_test.pdf")
    .extract()
    .embed()
    .caption()
    .vdb_upload(
        collection_name=collection_name,
        milvus_uri=milvus_uri,
        sparse=sparse,
        # for llama-3.2 embedder, use 1024 for e5-v5
        dense_dim=2048,
        stream=False,
        recreate=False
    )
```



## Upload to a Custom Data Store

You can ingest to other data stores by using the `Ingestor.vdb_upload` method; 
however, you must configure other data stores and connections yourself. 
NeMo Retriever Library does not provide connections to other data sources. 

!!! important

    NVIDIA makes no claim about accuracy, performance, or functionality of any vector database except Milvus. If you use a different vector database, it's your responsibility to test and maintain it.

For more information, refer to [Build a Custom Vector Database Operator](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/building_vdb_operator.ipynb).



## Related Topics

- [Python API Reference](python-api-reference.md)
- [Store Extracted Images](python-api-reference.md#store-extracted-images)
- [Environment Variables](environment-config.md)
- [Troubleshoot NeMo Retriever Library](troubleshoot.md)
