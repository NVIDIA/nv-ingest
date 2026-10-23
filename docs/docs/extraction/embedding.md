# Use Embedding with NeMo Retriever Library

## Local checkpoint defaults

This development branch defaults to the Nemotron 3.5 Embed 1B family for local inference. These changes are not part of the documented 26.08.1 release.

This integration remains experimental while the checkpoints require private repository access. Validate retrieval quality on your data before deployment; successful model loading alone is not production retrieval validation.

When you omit the local embedding model, the library selects a checkpoint as follows:

- For in-process inference with the default `vllm` backend, it selects `nvidia/Nemotron-3.5-Embed-1B-NVFP4` when every visible CUDA device supports native NVFP4.
- On other supported local GPUs, it selects `nvidia/Nemotron-3.5-Embed-1B-BF16`.
- With the `hf` backend, it selects the BF16 checkpoint. The compressed-tensors NVFP4 checkpoint requires vLLM.
- For Ray batch GPU execution, automatic selection uses BF16 because cluster GPU counts do not establish worker GPU architectures. Explicit Nemotron 3.5 precision checkpoint IDs remain unchanged; select Nemotron 3.5 NVFP4 only when every eligible worker supports it.

The `retriever ingest batch` CLI treats explicit Nemotron 3 precision IDs as the logical Nemotron 3 model and selects BF16.

The default modality remains `text`. Nemotron 3.5 also supports `image` and `text_image` inputs when you explicitly select those modalities.

The prerelease checkpoints require Hugging Face repository access. Set `HF_TOKEN` to a token from an account with access before loading them. Do not include tokens in source files or shared configuration.

For Nemotron 3.5, local inference uses the checkpoint's processor and retrieval chat template for query and document formatting. Pass unformatted content; do not add `query: ` or `passage: ` yourself. By default, the local backends produce normalized, masked mean-pooled embeddings.

The local adapters preserve the checkpoint's bidirectional text attention. The vLLM path registers an NRL-specific embedding model; it does not patch vendor model classes or change attention for unrelated models.

The 1B checkpoints produce 2,048-dimensional vectors. This adapter supports only the checkpoint's full output dimension; reduced-dimension overrides are rejected.

Before changing an existing index's model or precision, validate retrieval quality on representative data. Follow [Keep the embedding model aligned](vdbs.md#lancedb-embedding-model-compatibility) to retain its recorded model or rebuild and re-ingest with the new default.

## Remote and Helm defaults

Hosted embedding and Helm retain the text-only Nemotron 3 Embed 1B default, with logical model ID `nvidia/nemotron-3-embed-1b`. An explicitly selected Nemotron 3 model also retains its existing local resolution behavior.

When graph execution selects the CPU embedding actor without an explicit endpoint, it calls `https://integrate.api.nvidia.com/v1/embeddings`. This hosted fallback retains the Nemotron 3 default and records that model on the index. Set `NVIDIA_API_KEY` or `NGC_API_KEY` before ingest.

Helm deploys `nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.2`. The NIM selects a supported NVFP4 or BF16 engine automatically. For NIM product details, refer to the [NeMo Retriever Text Embedding NIM documentation](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html).

## Development Compose defaults

The development service Compose `nims-core` profile uses
`nvcr.io/nim/nvidia/nemotron-embed:latest`. The native runtime automatically
selects NVFP4 on supported GPUs and the BF16 checkpoint with FP16 execution
on other supported GPUs. The matching `nims-core.env` preset uses
`nvidia/nemotron-3.5-embed-1b` for ingestion and query embedding.
This served alias does not select the checkpoint or precision.
For client-side input validation and splitting, the library resolves the alias
to the revision-pinned `nvidia/Nemotron-3.5-Embed-1B-BF16` tokenizer and input
policy. HTTP embedding requests retain the served alias, including when the
NIM selects NVFP4.
The default modality remains `text`.
For startup commands and cache configuration, refer to the
[development Compose README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/dev/compose/README.md#self-hosted-nim-profiles).

Before migrating an existing Nemotron 3 index, back up its data and rebuild
and re-ingest the affected tables or collections. Matching vector dimensions
do not make embeddings from different models compatible. Follow
[Keep the embedding model aligned](vdbs.md#lancedb-embedding-model-compatibility)
and evaluate retrieval quality on representative data.
Changing GPU hardware can change the selected checkpoint or precision.
Validate retrieval quality and rebuild and re-ingest affected collections
when their embedding model or precision changes, even if the API alias stays
the same.

## Example with Default Text-Based Embedding { #example-with-default-text-based-embedding }

By default, all extracted content, including tables and charts, is treated as plain text.
The following example provides a strong baseline for retrieval.

- The `embed` method is called with no arguments.

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed()  # Default behavior embeds all content as text
)
results = ingestor.ingest()
```

## Text inputs that exceed the model limit { #text-input-overflow }

For text inputs, including `text_image` inputs without an image, NeMo Retriever
Library checks the complete formatted input against the embedding model's token
limit. Local Nemotron 3.5 paired text-and-image inputs also use admission planning, as described below. Legacy and remote image-bearing inputs retain their existing behavior.
The check includes the model's document or query prefix and special tokens.
The default configured limits are 8,192 tokens for passages and 128 tokens for
queries. If the checkpoint declares a smaller supported limit, the checkpoint
limit takes precedence.

To set the passage limit in the Python API, use
`EmbedParams(runtime=ModelRuntimeParams(max_length=N))`. The limit includes the
model prefix and special tokens and cannot exceed the checkpoint's supported
limit. The same text-input policy applies to the `hf` and `vllm` backends for
both text-only and vision-language embedding models. Oversized text is split
losslessly, not truncated.

`TextChunkParams.max_tokens` controls earlier text chunking, not the final
formatted embedding-input limit. `EmbedParams.query_max_length` controls the
separate query limit.

With its default query graph, `Retriever` searches every split query chunk
using the chunk's embedding and matching text for hybrid retrieval. It combines
the chunk rankings for each original query with equal-weight reciprocal rank
fusion: `score(hit) = sum(1 / (60 + rank))`, with ranks starting at 1.
Each hit contributes at most once per chunk. Equal scores retain first-encounter
order. Queries that do not split retain their existing ranking and scores.

`Retriever.queries()` returns one result group per input query, in input order,
including repeated identical inputs. `top_k` limits the final hits for each
original query. When enabled, reranking runs after fusion and uses the original
full query text. Caller-supplied query graphs and direct vector retrieval retain
their own result contracts.

For local Nemotron 3.5 `text_image` ingestion, `max_length` covers the combined retrieval template, expanded image tokens, and text. The default passage limit remains 8,192 tokens; it is not a separate text budget plus an image allowance.

The library counts tokens with the checkpoint's processor before inference. If paired text does not fit, it splits the text losslessly and repeats the unchanged full image for every child row. Children preserve source and page metadata and receive the same `metadata["embedding_split"]` provenance described below. This policy does not crop images or truncate image tokens. Repeating an image can increase inference work and the number of stored vectors.

Processor truncation is disabled for image-bearing admission and inference. The vLLM adapter rejects a request if its expanded token count differs from the checkpoint processor's admission count.

If the image and template leave no text budget, planning raises an actionable error. Increase `max_length` within the checkpoint's supported maximum. Direct `embed_text_image()` calls cannot change the number of returned vectors and reject oversized pairs instead of splitting them.

For a registered revision-pinned model, an explicitly revision-pinned model,
or a local checkpoint, the library loads the tokenizer and prompt configuration
for that exact model version. If the text does not fit, the library splits it
into deterministic contiguous token ranges that fit. This split does not
truncate text and occurs before either local or remote embedding.
The tokenizer used for text counting and splitting disables checkpoint-defined
truncation and padding so it measures the complete input. This does not change
the inference tokenizer's settings.
If the exact tokenizer, selected prompt prefix, or checkpoint-supported token
limit is unavailable, embedding stage setup fails before inference rather than
guessing an admission policy.

Each split row preserves the source, page, element, bounding box, and existing
document chunk metadata from its parent. The library adds one
`metadata["embedding_split"]` mapping so you can identify and order the
embedding-specific children:

- `parent_id` identifies the parent content and provenance.
- `chunk_id` identifies one deterministic child.
- `chunk_index` and `chunk_count` describe the child order.
- `start_token` and `end_token` describe the source token range.
- `content` preserves the child's exact text, including whitespace.

The returned `DataFrame` can therefore contain more rows than the embedding
stage received. Existing fields such as `chunk_index` and the physical page
number keep their original meaning.

Dense LanceDB and collection writes preserve valid split children, including
whitespace-only children, and store the complete `embedding_split` mapping in
the JSON `metadata` field. After decoding that field, use
`metadata["embedding_split"]["chunk_id"]` for the stable embedding child ID.
A collection row's top-level `chunk_id` remains a storage key derived from the
document, version, and row index; it is not the embedding child ID.

Local and remote text embedding use the same prepared rows. When this client-side
policy is active for a remote endpoint, text requests use `truncate="NONE"` so
the endpoint cannot silently replace the client decision. Image-bearing inputs
retain `truncate="END"`. For mixed `text_image` batches, the library sends
text-only and image-bearing inputs in separate requests and preserves result
order.

If a backend still rejects a prepared batch, the library reports a batch failure
rather than guessing from an HTTP status or exception that one document is
invalid. The VDB boundary refuses a mixed partial write when searchable rows
are missing embeddings.

For an unpinned custom remote model, the library does not guess its tokenizer
or input limit. Embedding stage setup fails with an actionable error. Use a
registered model, a local checkpoint, or an immutable model revision so the
library can enforce deterministic client-side admission.

The embedding stage records per-row counts in
`embedding_v1_counts_by_label`. When a batch contains an overlength or failed
row, it also logs a summary with `input_rows`, `output_rows`, `overlength`,
`split`, `split_children`, `truncated`, `failed`, `embedded`, and `unembedded`.
The deterministic split policy reports `truncated=0`.

## Example with Embedding Structured Elements as Text + Images { #example-with-embedding-structured-elements-as-text-images }

It is common to process PDFs by embedding standard text as text and visual elements such as tables and charts as images.
The following example enables the multimodal model to capture the spatial and structural information of the visual content.

- For local inference, the Nemotron 3.5 default supports combined text and images without a model override.
- The `embed` method is configured with `embed_modality="text_image"` to embed the extracted tables and charts as images.
- Evaluate retrieval quality and throughput on your data before selecting this modality.

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed(
        embed_modality="text_image",
    )
)
results = ingestor.ingest()
```

For a remote endpoint, explicitly select a multimodal model that the endpoint supports. The hosted and Helm Nemotron 3 defaults are text-only.

For CPU preparation concurrency and remote image request batch sizes, refer to [Tune embedding preparation and remote requests](performance_guide.md#tune-embedding-preparation-and-remote-requests).

## Example with Embedding Entire PDF Pages as Images { #example-with-embedding-entire-pdf-pages-as-images }

For documents where the entire page layout is important (such as infographics, complex diagrams, or forms), 
you can configure NeMo Retriever Library to treat every page as a single image.
The following example extracts and embeds each page as an image.

- Set `embed_modality="image"` to use the rendered page image as the embedding input.
- Set `embed_granularity="page"` to create one result row for each PDF page.
- For local inference, use the multimodal Nemotron 3.5 default.

These arguments work together. When you set both arguments, the pipeline
enables page-image rendering during extraction, creates one row for each page,
and embeds the full rendered page image. Either argument alone does not enable
the complete page-as-image workflow.

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md) (`create_ingestor` and `.embed()`).

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="batch")
    .files("./data/*.pdf")
    .extract()
    .embed(
        embed_modality="image",
        embed_granularity="page",
    )
)
results = ingestor.ingest()
```

## Related Topics { #related-topics }

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
- [Use the Python API](nemo-retriever-api-reference.md)
