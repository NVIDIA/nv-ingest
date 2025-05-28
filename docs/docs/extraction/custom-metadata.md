# Use Custom Metadata to Filter Search Results

You can upload custom metadata for documents during ingestion. 
By uploading custom metadata you can attach additional information to documents, 
and use it for filtering results during retrieval operations. 
For example, you can add author metadata to your documents, and filter by author when you retrieve results.

Use this documentation to use custom metadata to filter search results when you work with [NeMo Retriever extraction](overview.md).


## Limitations

The following are limitation when you use custom metadata:

- Metadata fields must be consistent across documents in the same collection.
- Complex filter expressions may impact retrieval performance.
- Timestamp filtering requires strict ISO 8601 format compliance.
- If you update your custom metadata, you must ingest your documents again to use the new metadata.



## Add Custom Metadata During Ingestion

You can add custom metadata during the document ingestion process by using the `/v1/documents` endpoint. 
You can specify metadata for each file, 
and you can specify different metadata for different documents in the same ingestion batch.


### Metadata Structure

You specify custom metadata as a list of objects, where each object contains the following:

- `filename` — The name of the document.
- `metadata` — A dictionary that contains key-value pairs of metadata.

The following example contains metadata fields `timestamp`, `category`, and `department`. 
You can create whatever metadata is helpful for your scenario.

```json
[
    {
        "filename": "document1.pdf",
        "metadata": {
            "timestamp": "2024-03-15T10:23:00",
            "category": "technical",
            "department": "engineering"
        }
    },
    {
        "filename": "document2.pdf",
        "metadata": {
            "timestamp": "2024-03-16T14:30:00",
            "category": "marketing",
            "department": "sales"
        }
    }
]
```


### Example: Add Custom Metadata During Ingestion

The following example adds custom metadata during ingestion.

```python
CUSTOM_METADATA = [
    {
        "filename": "technical_doc.pdf",
        "metadata": {
            "timestamp": "2024-03-15T10:23:00",  # ISO 8601 format as string
            "category": "technical",              # string
            "department": "engineering",          # string
            "priority": "1",                      # numeric as string
            "is_active": "true"                   # boolean as string
        }
    },
    {
        "filename": "marketing_doc.pdf",
        "metadata": {
            "timestamp": "2024-03-16T14:30:00",
            "category": "marketing",
            "department": "sales",
            "priority": "2",
            "is_active": "false"
        }
    }
]

# Include in upload request
data = {
    "collection_name": "my_collection",
    "blocking": False,
    "split_options": {
        "chunk_size": 512,
        "chunk_overlap": 150
    },
    "custom_metadata": CUSTOM_METADATA
}
```



## Considerations for Custom Metadata

Consider the following before you create your custom metadata.

- **Metadata types** — You can specify strings, numeric values, boolean values, and timestamps, but you must specify all values as strings. For example, specify `"priority": "1"` and `"is_active": "true"`.
- **Timestamp format** — Specify timestamps in ISO 8601 format. For example, `"timestamp": "2024-03-15T10:23:00"`.


### Best Practices

The following are the best practices when you work with custom metadata:

- Metadata Design
  - Plan metadata structure before ingestion.
  - Use consistent naming conventions.
  - Include essential filtering fields.
  - Keep metadata values consistent across documents.
  - Document the expected string format for each metadata field.

- Timestamp Usage
  - Consider time zone implications.
  - Use consistent timestamp precision.

- Filter Expressions
  - Test filter expressions with small datasets first.
  - Use parentheses to clarify complex expressions.
  - Consider performance implications of complex filters.

- Error Handling
  - Validate metadata during ingestion.
  - Handle missing metadata fields gracefully.
  - Log invalid filter expressions.



## Use Custom Metadata to Filter Results During Retrieval

You can use custom metadata to filter documents during retrieval operations 
by using the `filter_expr` parameter in both the `/v1/search` and `/v1/generate` endpoints.


### Filter Expression Syntax

Use filter expressions that follow the Milvus boolean expression syntax. 
For more information, refer to [Filtering Explained](https://milvus.io/docs/boolean.md).

Use the following information to write filter expressions:

- Access metadata fields by using `content_metadata["field_name"]`.
- You can use the following operators:
  - Comparison: ==, !=, >, >=, <, <=
  - Logical: AND, OR, NOT
  - Range: LIKE, IN
- Since all metadata values are strings, comparisons are done with string values. For example, `content_metadata["priority"] == "1"`.


### Example Filter Expressions

The following example filters results by category.

```python
filter_expr = 'content_metadata["category"] == "technical"'
```

The following example filters results by time range.

```python
filter_expr = 'content_metadata["timestamp"] >= "2024-03-01T00:00:00" and content_metadata["timestamp"] <= "2024-03-31T23:59:59"'
```

The following example filters by category and uses multiple logical operators.

```python
filter_expr = '(content_metadata["department"] == "engineering" and content_metadata["priority"] == "high") or content_metadata["category"] == "critical"'
```


### Example: Use a Filter Expression in Search

The following example uses a filter expression to narrow results.

```python
payload = {
    "query": "What are the technical specifications?",
    "reranker_top_k": 10,
    "vdb_top_k": 100,
    "collection_names": ["my_collection"],
    "enable_query_rewriting": True,
    "enable_reranker": True,
    "filter_expr": 'content_metadata["category"] == "technical" and content_metadata["priority"] == "high"'
}
```


### Example: Using Filter Expressions in Generate

The following example uses a filter expression to narrow results.

```python
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What are the latest engineering updates?"
        }
    ],
    "use_knowledge_base": True,
    "collection_names": ["my_collection"],
    "filter_expr": 'content_metadata["department"] == "engineering" and content_metadata["timestamp"] >= "2024-03-01T00:00:00"'
}
```



## Troubleshooting

The following are some issues that might arise when you work with custom metadata:

- Filter Expression Errors
  - Verify that the metadata field names are correct.
  - Verify that all values are correctly enclosed in quotes.
  - Verify all metadata values are strings in filter expressions.
  - Verify the operator syntax. For valid expression syntax, refer to [Milvus Filtering Documentation](https://milvus.io/docs/boolean.md).

- Timestamp Filtering Issues
  - Verify that the metadata uses the ISO 8601 format.
  - Verify that the time zones are consistent.
  - Validate the date range logic.

- Missing Metadata
  - Verify that the metadata was added during ingestion.
  - Verify that you specified the correct document filename.
  - Validate the metadata structure.



## Related Content

- For a notebook that uses the CLI to add custom metadata and filter query results, see [metadata_and_filtered_search.ipynb
](https://github.com/NVIDIA/nv-ingest/blob/main/examples/metadata_and_filtered_search.ipynb).
