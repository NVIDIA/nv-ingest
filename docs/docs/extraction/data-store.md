## Data Upload for NeMo Retriever Extraction

Use this documentation to learn how [NeMo Retriever extraction](overview.md) handles and uploads data.

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.

## Overview

NeMo Retriever extraction supports extracting text representations of various forms of content, 
and ingesting to the [Milvus vector database](https://milvus.io/). 
NeMo Retriever extraction does not store data on disk directly, except through Milvus. 
The data upload task pulls extraction results to the Python client, 
and then pushes them to Milvus by using its underlying Minio object store service.

The vector database stores only the extracted text representations of ingested data. 
It does not store the embeddings for images.

NeMo Retriever extraction supports uploading data by using the [Ingestor.vdb_upload API](nv-ingest-python-api.md). 
Currently, data upload is not supported through the [NV Ingest CLI](nv-ingest_cli.md).
 


## Upload to Milvus

The `vdb_upload` method uses GPU Cagra accelerated bulk indexing support to load chunks into Milvus. 
To enable hybrid retrieval, nv-ingest supports both dense (llama-embedder embeddings) and sparse (bm25) embeddings. 

Bulk indexing is high throughput, but has a built-in overhead of around one minute. 
If the number of ingested documents is 10 or fewer, nv-ingest uses faster streaming inserts instead. 
You can control this by setting `stream=True`. 

If you set `recreate=True`, nv-ingest drops and recreates the collection given as *collection_name*. 
The Milvus service persists data to disk by using a Docker volume defined in docker-compose.yaml. 
You can delete all collections by deleting that volume, and then restarting the nv-ingest service.

To upload to Milvus, use code similar to the following.

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
NeMo Retriever extraction does not provide connections to other data sources. 
