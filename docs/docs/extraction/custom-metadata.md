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

You specify custom metadata as a TODO 

The following example contains metadata fields TODO. 
You can create whatever metadata is helpful for your scenario.

TODO
```
```


### Example: Add Custom Metadata During Ingestion

The following example adds custom metadata during ingestion.

TODO
```
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


### Filter Expression Syntax

Use filter expressions that follow the Milvus boolean expression syntax. 
For more information, refer to [Filtering Explained](https://milvus.io/docs/boolean.md).


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

TODO

```python
```



## Related Content

- For a notebook that uses the CLI to add custom metadata and filter query results, see [metadata_and_filtered_search.ipynb
](https://github.com/NVIDIA/nv-ingest/blob/main/examples/metadata_and_filtered_search.ipynb).
