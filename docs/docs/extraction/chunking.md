# Split Documents

Splitting, also known as chunking, breaks large documents or text into smaller, manageable sections to improve retrieval efficiency. 
After chunking, only the most relevant pieces of information are retrieved for a given query. 
Chunking also prevents text from exceeding the context window of the embedding model.

There are two ways that NV Ingest chunks text:

- By using the `text_depth` parameter in the `extraction` task.
- Token-based splitting by using the `split` task.


## Extraction Text Depth

You can use the `text_depth` parameter to specify how extracted text is chunked together by the extractor. 
For example, the following code chunks the document text by page, for document types that have pages.

```python
ingestor = ingestor.extract(
    extract_text=True,
    text_depth="page"
)
```

The following table contains the `text_depth` parameter values.

| Value | Description |
| ------ | ----------- |
| `document` | Doesn't perform any splitting, and returns the full document as one chunk. |
| `page` | Returns a single chunk of text for each page. |

For most documents, we recommend that you set `text_depth` to `page`, because this tends to give the best performance for retrieval. 
However, in some cases, such as with `.txt` documents, there aren't any page breaks to split on. 

If you want chunks smaller than `page`, use token-based splitting as described in the following section.



## Token-Based Splitting

The `split` task uses a tokenizer to count the number of tokens in the document, 
and splits the document based on the desired maximum chunk size and chunk overlap. 
We recommend that you use the `meta-llama/Llama-3.2-1B` tokenizer, 
because it's the same tokenizer as the llama-3.2 embedding model that we use for embedding.
However, you can use any tokenizer from any HuggingFace model that includes a tokenizer file.

Use the `split` method to chunk large documents as shown in the following code.

!!! note

    The default tokenizer (`meta-llama/Llama-3.2-1B`) requires a [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens). You must set `hf_access_token": "hf_***` to authenticate.

```python
ingestor = ingestor.split(
    tokenizer="meta-llama/Llama-3.2-1B",
    chunk_size=1024,
    chunk_overlap=150,
    params={"split_source_types": ["text", "PDF"], "hf_access_token": "hf_***"}
)
```

To use a different tokenizer, such as `intfloat/e5-large-unsupervised`, you can modify the `split` call as shown following.

```python
ingestor = ingestor.split(
    tokenizer="intfloat/e5-large-unsupervised",
    chunk_size=1024,
    chunk_overlap=150
)
```

### Split Parameters

The following table contains the `split` parameters.

| Parameter | Description | Default |
| ------ | ----------- | -------- |
| `tokenizer` | HuggingFace Tokenizer identifier or path. | `meta-llama/Llama-3.2-1B`|
| `chunk_size` | Maximum number of tokens per chunk.  | `1024` |
| `chunk_size` | Number of tokens to overlap between chunks.  | `150` |
| `params` | A sub-dictionary that can contain `split_source_types` and `hf_access_token` | `{}` |
| `hf_access_token` | Your Hugging Face access token. | — |
| `split_source_types` | The source types to split on (only splits on text by default). | — |



### Pre-download the Tokenizer

When the NV Ingest container is built, it pre-downloads a default tokenizer,  
so that it doesn't have to download the tokenizer at runtime. 

By default, the NV Ingest container downloads the `intfloat/e5-large-unsupervised` tokenizer,
which is not gated, and does not require any special permissions.

You can use the `meta-llama/Llama-3.2-1B` tokenizer instead, 
but this is a gated model, and requires special permissions.
To pre-download the `meta-llama/Llama-3.2-1B` tokenizer, you must do the following:

- Review the [license agreement](https://huggingface.co/meta-llama/Llama-3.2-1B).
- [Request access](https://huggingface.co/meta-llama/Llama-3.2-1B).
- Set the `HF_ACCESS_TOKEN` environment variable to your HuggingFace access token.
- Set the `DOWNLOAD_LLAMA_TOKENIZER` environment variable to `true`.
