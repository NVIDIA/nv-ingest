## Data Upload for NeMo Retriever Extraction

Use this documentation to learn how [NeMo Retriever extraction](overview.md) handles and uploads data.

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.


NeMo Retriever extraction supports extracting text representations of various forms of content, and ingesting to the [Milvus vector database](https://milvus.io/). 
NeMo Retriever extraction does not store data on disk except through Milvus and its underlying Minio object store.
You can ingest to other data stores by using the `Ingestor.vdb_upload` method. 
However, you must configure other data stores yourself. 


No, NeMo Retriever extraction does not have connections to data sources. 



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
        dense_dim=2048
    )
```




How do VDB Uploads work?
nv-ingest natively supports efficient uploads into Milvus.
…
.vdb_upload(
        collection_name=collection_name,
        milvus_uri=milvus_uri,
        sparse=sparse,
        # for llama-3.2 embedder, use 1024 for e5-v5
        dense_dim=2048,
        recreate=False
    )
…

In the snippet above, the vdb_upload() task will use GPU Cagra accelerated bulk indexing support to load chunks into Milvus.

It supports both dense (llama-embedder embeddings), and sparse (bm25) embeddings. This enables hybrid retrieval. 

Bulk indexing is high throughput, but has a built-in overhead of around a minute. IIf the number of ingested documents is <= 10, we use faster streaming inserts instead. This is also directly controllable with stream=True. 

If recreate is True, we will drop and recreate the collection given as “collection_name”.

The Milvus service persists data to disk via a Docker volume defined in docker-compose.yaml. It’s possible to delete all collections by deleting that volume and restarting the service.

Note that the VDB upload task is implemented by pulling extraction results to the Python client, then pushing them up to Milvus via its underlying Minio service.

The VDB upload task is not currently supported via the nv-ingest-cli.
 
Does the vector database contain the embeddings for images as well?
No. The vector database stores only the extracted text representations of ingested data.
