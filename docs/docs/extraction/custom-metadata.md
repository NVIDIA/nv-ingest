# Use Custom Metadata to Filter Search Results

You can upload custom metadata for documents during ingestion. 
By uploading custom metadata you can attach additional information to documents, 
and use it for filtering results during retrieval operations. 
For example, you can add author metadata to your documents, and filter by author when you retrieve results. 
To create filters, you use [Milvus Filtering Expressions](https://milvus.io/docs/boolean.md).

Use this documentation to use custom metadata to filter search results when you work with [NeMo Retriever extraction](overview.md).


## Limitations

The following are limitation when you use custom metadata:

- Metadata fields must be consistent across documents in the same collection.
- Complex filter expressions may impact retrieval performance.
- If you update your custom metadata, you must ingest your documents again to use the new metadata.



## Add Custom Metadata During Ingestion

You can add custom metadata during the document ingestion process. 
You can specify metadata for each file, 
and you can specify different metadata for different documents in the same ingestion batch.


### Metadata Structure

You specify custom metadata as a dataframe or a file (json, csv, or parquet). 

The following example contains metadata fields for category, department, and timestamp. 
You can create whatever metadata is helpful for your scenario.

```python
import pandas as pd

meta_df = pd.DataFrame(
    {
        "source": ["data/woods_frost.pdf", "data/multimodal_test.pdf"],
        "category": ["Alpha", "Bravo"],
        "department": ["Language", "Engineering"],
        "timestamp": ["2025-05-01T00:00:00", "2025-05-02T00:00:00"]
    }
)

# Convert the dataframe to a csv file, 
# to demonstrate how to ingest a metadata file in a later step.

file_path = "./meta_file.csv"
meta_df.to_csv(file_path)
```


### Example: Add Custom Metadata During Ingestion

The following example adds custom metadata during ingestion. 
For more information about the `Ingestor` class, see [Use the NV-Ingest Python API](nv-ingest-python-api.md).
For more information about the `vdb_upload` method, see [Upload Data](data-store.md).

```python
from nv_ingest_client.client import Ingestor

hostname="localhost"
collection_name = "nv_ingest_collection"
sparse = True

ingestor = ( 
    Ingestor(message_client_hostname=hostname)
        .files(["data/woods_frost.pdf", "data/multimodal_test.pdf"])
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            text_depth="page"
        )
        .embed()
        .vdb_upload(
            collection_name=collection_name, 
            milvus_uri=f"http://{hostname}:19530", 
            sparse=sparse, 
            minio_endpoint=f"{hostname}:9000", 
            dense_dim=2048,
            meta_dataframe=file_path, 
            meta_source_field="source", 
            meta_fields=["category", "department", "timestamp"]
        )
)
results = ingestor.ingest_async().result()
```



## Best Practices

The following are the best practices when you work with custom metadata:

- Plan metadata structure before ingestion.
- Test filter expressions with small datasets first.
- Consider performance implications of complex filters.
- Validate metadata during ingestion.
- Handle missing metadata fields gracefully.
- Log invalid filter expressions.



## Use Custom Metadata to Filter Results During Retrieval

You can use custom metadata to filter documents during retrieval operations. 
Use filter expressions that follow the Milvus boolean expression syntax. 
For more information, refer to [Filtering Explained](https://milvus.io/docs/boolean.md).


### Example Filter Expressions

The following example filters results by category.

```python
filter_expr = 'content_metadata["category"] == "technical"'
```

The following example filters results by time range.

```python
filter_expr = 'content_metadata["timestamp"] >= "2024-03-01T00:00:00" and content_metadata["timestamp"] <= "2025-12-31T00:00:00"'
```

The following example filters by category and uses multiple logical operators.

```python
filter_expr = '(content_metadata["department"] == "engineering" and content_metadata["priority"] == "high") or content_metadata["category"] == "critical"'
```


### Example: Use a Filter Expression in Search

After ingestion is complete, and documents are uploaded to the database with metadata, 
you can use the `content_metadata` field to filter search results.

The following example uses a filter expression to narrow results by department.

```python
from nv_ingest_client.util.milvus import nvingest_retrieval

hostname="localhost"
collection_name = "nv_ingest_collection"
sparse = True
top_k = 5
model_name="nvidia/llama-3.2-nv-embedqa-1b-v2"

filter_expr = 'content_metadata["department"] == "Engineering"'

queries = ["this is expensive"]
q_results = []
for que in queries:
    q_results.append(
        nvingest_retrieval(
            [que], 
            collection_name, 
            milvus_uri=f"http://{hostname}:19530", 
            embedding_endpoint=f"http://{hostname}:8012/v1",  
            hybrid=sparse, 
            top_k=top_k, 
            model_name=model_name, 
            gpu_search=False, 
            _filter=filter_expr
        )
    )

print(f"{q_results}")
```



## Related Content

- For a notebook that uses the CLI to add custom metadata and filter query results, see [metadata_and_filtered_search.ipynb
](https://github.com/NVIDIA/nv-ingest/blob/main/examples/metadata_and_filtered_search.ipynb).
