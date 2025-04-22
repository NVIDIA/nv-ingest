# Chunking Text


Chunking or splitting a document involves breaking down the extracted text into smaller chunks so that only the most relevant pieces of information are retrieved for a given query. It also prevents the text from exceeding the context window of the embedding model.

There are two ways that NV-Ingest chunks text, the `text_depth` parameter in the `extraction` task and token based splitting performed by the `split` task.

## Text Depth

When performing text extraction on a document you can use the `text_depth` parameter to specify how extracted text is chunked together by the extractor. For example, the following code will split the document text by page for document types that have pages.
```
ingestor = ingestor.extract(
    extract_text=True,
    text_depth="page"
)
```


The following table provides the possible values for `text_depth`
| Depth | Description |
| ------ | ----------- |
| `"Document"` | Doesn't perform any splitting and returns full document text as one chunk |
| `"Page"` | Returns a single chunk of text for each page |

For most documents, we recommend setting `text_depth` to `"page"` as this tends to give the best performance for retrieval. However, in some cases, such as with `.txt`, there aren't any page breaks to split on. In other cases you may want samller chunks. For these cases we offer the `split` task, which splits text based on token count.


 ## Token Based Splitting

 !!! note

    The default tokenizer (`"meta-llama/Llama-3.2-1B"`) requires a [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens). You must set `"hf_access_token": "hf_***"` to authenticate.

 The `split` task uses a tokenizer to count the number of tokens in the document and segment based on the desired maximum chunk  size and chunk overlap. We reccommend using the `"meta-llama/Llama-3.2-1B"` tokenizer as it's the same tokenizer as the llama-3.2 embedding model that we use for embedding. However, a tokenizer can be configured from any HuggingFace model that includes a tokenizer file.

 ```python
ingestor = ingestor.split(
    tokenizer="meta-llama/Llama-3.2-1B",
    chunk_size=1024,
    chunk_overlap=150,
    params={"split_source_types": ["text", "PDF"], "hf_access_token": "hf_***"}
)
```


 ### Parameters

- "tokenizer": HuggingFace Tokenizer identifier.
- "chunk_size": Maximum number of tokens per chunk.
- "chunk_overlap": Number of tokens to overlap between chunks.
- "params": A sub-dictionary that may contain:
    - "hf_access_token": Hugging Face access token.
    - "split_source_types": List of source types to filter for splitting.

| Parameter | Description | Default |
| ------ | ----------- | -------- |
| `"tokenizer"` | Tokenizer identifier or path | `"meta-llama/Llama-3.2-1B"`|
| `"chunk_size"` | Chunks based on page breaks  | `1024` |
| `"chunk_size"` | Chunks based on page breaks  | `150` |
| `"params"` | A sub-dictionary that may contain `"split_source_types"` which determines the source types to split on (only splits on text by default) and `"hf_access_token"` which can be set to your HuggingFace access token  | `{}` |

### Predownloading the Tokenizer 

When the NV-Ingest container is built, it automatically downloads a tokenizer that is used when `"tokenizer"` is set to `None`. This means that no HuggingFace downloads will take place at runtime. If the `DOWNLOAD_LLAMA_TOKENIZER` env varibale is set to `"True"`, the `"meta-llama/Llama-3.2-1B"` tokenizer will be downloaded. Please review the [license agreement](https://huggingface.co/meta-llama/Llama-3.2-1B) for Llama 3.2 materials before using this. This is a gated model so you'll need to [request access](https://huggingface.co/meta-llama/Llama-3.2-1B) and set `HF_ACCESS_TOKEN` to your HuggingFace access token in order to predownload it.

Otherwise, the `"intfloat/e5-large-unsupervised"` tokenizer will be predowloaded, which is ungated and does not require any special permissions.
