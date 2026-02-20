# Split Documents

Splitting, also known as chunking, breaks large documents or text into smaller, manageable sections to improve retrieval efficiency. 
After chunking, only the most relevant pieces of information are retrieved for a given query. 
Chunking also prevents text from exceeding the context window of the embedding model.

There are two ways that NV Ingest chunks text:

- By using the `text_depth` parameter in the `extraction` task.
- Token-based splitting by using the `split` task.


!!! warning

    NeMo Retriever extraction is designed to process language and language-length strings. If you submit a document that contains extremely long, or non-language text strings, such as a DNA sequence, errors or unexpected results occur.



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

The `split` task uses a tokenizer to count tokens and split by chunk size and overlap. For the default tokenizer, optional Llama tokenizer, and environment variables, refer to the canonical subsection below.

### Token-based splitting and tokenizers

This section is the **canonical reference** for tokenizer behavior in NV-Ingest. When tokenizer behavior, gating, or env vars change, update this section first; then update or link from any other doc that mentions tokenizers (see [Documentation maintenance](contributing.md#documentation-maintenance)).

**Default tokenizer behavior**

- When you do not specify a tokenizer, the service uses a pre-downloaded tokenizer if present: if the container was built with `DOWNLOAD_LLAMA_TOKENIZER=True`, it uses `meta-llama/Llama-3.2-1B` from the container; otherwise it uses `intfloat/e5-large-unsupervised`. If no tokenizer is pre-downloaded, the runtime default is `intfloat/e5-large-unsupervised`.
- **Recommended for embedding alignment:** Use `meta-llama/Llama-3.2-1B` with `chunk_size=1024` and `chunk_overlap=150` so split boundaries align with the default embedding model. You can use any Hugging Face model that provides a tokenizer.

**Optional Llama tokenizer (`meta-llama/Llama-3.2-1B`)**

- The model is **gated on Hugging Face**. You must accept the [license](https://huggingface.co/meta-llama/Llama-3.2-1B) and [request access](https://huggingface.co/meta-llama/Llama-3.2-1B).
- **At runtime:** If the Llama tokenizer is not pre-downloaded in the container, you must provide a Hugging Face access token via `params={"hf_access_token": "hf_***"}` in the split task or set the `HF_ACCESS_TOKEN` environment variable for the ingest service. See [Hugging Face access tokens](https://huggingface.co/docs/hub/en/security-tokens).
- **At build time:** To pre-download the Llama tokenizer into the image (no runtime token needed), set `DOWNLOAD_LLAMA_TOKENIZER=True` and provide `HF_ACCESS_TOKEN` (or the equivalent build secret) during the Docker build.

**Environment variables**

| Variable | When it applies | What it does |
|----------|------------------|--------------|
| `DOWNLOAD_LLAMA_TOKENIZER` | Build time only | When `True`, pre-downloads `meta-llama/Llama-3.2-1B` into the container. Requires `HF_ACCESS_TOKEN` (or build secret) during build. When `False` (default in docker-compose), the container pre-downloads `intfloat/e5-large-unsupervised` instead. |
| `HF_ACCESS_TOKEN` | Build and/or runtime | Hugging Face access token. **Required at build** when `DOWNLOAD_LLAMA_TOKENIZER=True`. **Required at runtime** when the split task uses `meta-llama/Llama-3.2-1B` and the tokenizer is not pre-downloaded in the container. |

All other tokenizer and split-default details (e.g. in environment variable reference, client examples) should link here. See [Support Matrix](support-matrix.md) for the default embedding model.

---

**Examples**

```python
# Recommended: Llama tokenizer aligned with default embedder (provide hf_access_token if not pre-downloaded)
ingestor = ingestor.split(
    tokenizer="meta-llama/Llama-3.2-1B",
    chunk_size=1024,
    chunk_overlap=150,
    params={"split_source_types": ["text", "PDF"], "hf_access_token": "hf_***"}
)

# Alternative tokenizer
ingestor = ingestor.split(
    tokenizer="intfloat/e5-large-unsupervised",
    chunk_size=1024,
    chunk_overlap=150,
    params={"split_source_types": ["text", "PDF"], "hf_access_token": "hf_***"}
)
```

### Split Parameters

| Parameter | Description | Default |
| --------- | ----------- | -------- |
| `tokenizer` | HuggingFace tokenizer identifier or path. | See [Token-based splitting and tokenizers](#token-based-splitting-and-tokenizers) above. |
| `chunk_size` | Maximum number of tokens per chunk. | `1024` |
| `chunk_overlap` | Number of tokens to overlap between chunks. | `150` |
| `params` | Can include `split_source_types` and `hf_access_token`. | `{}` |
| `hf_access_token` | Hugging Face access token (required for gated Llama tokenizer when not pre-downloaded). | — |
| `split_source_types` | Source types to split on (text only by default). | — |

## Related Topics

- [Use the Python API](nv-ingest-python-api.md)
- [NeMo Retriever Extraction V2 API Guide](v2-api-guide.md)
- [Environment Variables](environment-config.md)
