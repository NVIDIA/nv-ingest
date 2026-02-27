# Data Upload for NeMo Retriever Extraction

Use this documentation to learn how [NeMo Retriever Library](overview.md) handles and uploads data.

!!! note

    NeMo Retriever Library is also known as NVIDIA Ingest.


## Overview

NeMo Retriever Library supports extracting text representations of various forms of content,
and ingesting to a vector database. **[LanceDB](https://lancedb.com/) is the default vector database backend** for storing and retrieving extracted embeddings. [Milvus](https://milvus.io/) remains fully supported as an alternative.

The data upload task (`vdb_upload`) pulls extraction results to the Python client,
and then pushes them to the configured vector database (LanceDB or Milvus). When using Milvus, data is pushed by using its underlying MinIO object store service.

The vector database stores only the extracted text representations of ingested data.
It does not store the embeddings for images.

!!! tip "Storing Extracted Images"

    To persist extracted images, tables, and chart renderings to disk or object storage, use the `store` task in addition to `vdb_upload`. The `store` task supports any fsspec-compatible backend (local filesystem, S3, GCS, etc.). For details, refer to [Store Extracted Images](nv-ingest-python-api.md#store-extracted-images).

NeMo Retriever Library supports uploading data by using the [Ingestor.vdb_upload API](nv-ingest-python-api.md).
Currently, data upload is not supported through the [NV Ingest CLI](nv-ingest_cli.md).



## Why LanceDB?

LanceDB delivers measurably lower retrieval latency through three architectural advantages over the previous Milvus default:

- **Lance columnar format** — Data is stored in Lance files, an Arrow/Parquet-style analytics layout optimized for fast local scans and indexed retrieval. This eliminates the serialization overhead of client-server protocols.
- **IVF_HNSW_SQ index** — Vectors are scalar-quantized (SQ) within an IVF-HNSW index, compressing them for faster search with lower memory bandwidth cost.
- **Embedded runtime** — LanceDB runs in-process, removing the multi-service dependency chain required by Milvus (Milvus server + etcd + MinIO). No external containers to start, configure, or maintain.

This combination of file format, index strategy, and simpler runtime path produces the latency improvements observed in benchmarks.



## Upload to LanceDB (default)

LanceDB uses the `LanceDB` operator class from the client library. You can configure it via the Python API or via the test harness.

### Programmatic API (Python)

```python
from nv_ingest_client.util.vdb.lancedb import LanceDB

vdb = LanceDB(
    uri="lancedb",           # Path to LanceDB database directory
    table_name="nv-ingest",  # Table name
    index_type="IVF_HNSW_SQ",  # Index type (default)
    hybrid=False,            # Enable hybrid search (BM25 FTS + vector)
)

# Ingest
vdb.run(results)

# Retrieve
docs = vdb.retrieval(queries, top_k=10)
```

When using the `Ingestor` with `vdb_upload`, the backend defaults to LanceDB unless you configure Milvus (see [Upload to Milvus](#upload-to-milvus)).

### Test harness configuration

In `tools/harness/test_configs.yaml`:

```yaml
active:
  vdb_backend: lancedb   # Options: "lancedb" (default) or "milvus"
  hybrid: false          # LanceDB only: enable hybrid retrieval (FTS + vector)
  sparse: false          # Milvus only: enable BM42 sparse embeddings
```

Or via environment variables:

```bash
# Switch to Milvus
VDB_BACKEND=milvus uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# Enable LanceDB hybrid search
HYBRID=true uv run nv-ingest-harness-run --case=e2e --dataset=bo767
```



## Hybrid search (LanceDB)

LanceDB supports **hybrid retrieval**, combining dense vector similarity with BM25 full-text search. Results are fused using Reciprocal Rank Fusion (RRF) reranking.

Hybrid search improves recall by approximately +0.5% to +3.5% over vector-only retrieval with negligible latency impact:

| Dataset            | Vector-Only Recall@5 | Hybrid Recall@5 | Delta  |
|--------------------|----------------------|-----------------|--------|
| bo767 (76K rows)   | 84.5%                | 85.0%           | +0.5%  |
| bo767 (reranked)   | 90.7%                | 91.8%           | +1.1%  |
| earnings (19K rows)| 61.5%                | 65.0%           | +3.5%  |
| earnings (reranked)| 74.5%                | 76.4%           | +1.9%  |

Hybrid search latency is typically 28–57 ms/query (vs. 31–37 ms/query for vector-only). The one-time FTS index build adds approximately 6.5 seconds for a 76K-row dataset.

Enable hybrid search by setting `hybrid=True` when creating the LanceDB operator or via the harness/config (e.g. `HYBRID=true`).



## Infrastructure: LanceDB vs Milvus

| Aspect              | LanceDB (default)       | Milvus                    |
|---------------------|-------------------------|---------------------------|
| Runtime model       | Embedded (in-process)   | Client-server             |
| External services   | None                    | Milvus + etcd + MinIO     |
| Docker Compose profile | Not needed           | `--profile retrieval`     |
| Index type          | IVF_HNSW_SQ             | HNSW, GPU_CAGRA, etc.     |
| Hybrid search       | BM25 FTS + vector (RRF) | BM42 sparse embeddings    |
| Persistence         | Lance files on disk     | Milvus server + MinIO     |



## Upload to Milvus

You can continue using Milvus with no code changes — set `vdb_backend: milvus` in the harness config or use the existing Milvus API calls (`vdb_upload(milvus_uri=...)`, `nvingest_retrieval(...)`).

The `vdb_upload` method uses GPU Cagra accelerated bulk indexing support to load chunks into Milvus.
To enable hybrid retrieval with Milvus, nv-ingest supports both dense (llama-embedder embeddings) and sparse (BM42) embeddings.

Bulk indexing is high throughput, but has a built-in overhead of around one minute.
If the number of ingested documents is 10 or fewer, nv-ingest uses faster streaming inserts instead.
You can control this by setting `stream=True`.

If you set `recreate=True`, nv-ingest drops and recreates the collection given as *collection_name*.
The Milvus service persists data to disk by using a Docker volume defined in docker-compose.yaml.
You can delete all collections by deleting that volume, and then restarting the nv-ingest service.

!!! warning

    When you use the `vdb_upload` task with Milvus, you must expose the ports for the Milvus and MinIO containers to the nv-ingest client. This ensures that the nv-ingest client can connect to both services and perform the `vdb_upload` action.

!!! tip

    When you use the `vdb_upload` method, the behavior of the upload depends on the `return_failures` parameter of the `ingest` method. For details, refer to [Capture Job Failures](nv-ingest-python-api.md#capture-job-failures).

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

For more information, refer to [Build a Custom Vector Database Operator](https://github.com/NVIDIA/nv-ingest/blob/main/examples/building_vdb_operator.ipynb).



## Related Topics

- [Use the NeMo Retriever Extraction Python API](nv-ingest-python-api.md)
- [Store Extracted Images](nv-ingest-python-api.md#store-extracted-images)
- [Environment Variables](environment-config.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
