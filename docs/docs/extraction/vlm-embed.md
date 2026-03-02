# Use Multimodal Embedding with NeMo Retriever Library

This documentation describes how to use [NeMo Retriever Library](overview.md)
with the multimodal embedding model [Llama Nemotron Embed VL 1B v2](https://build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2).

The `Llama Nemotron Embed VL 1B v2` model is optimized for multimodal question-answering retrieval.
The model can embed documents in the form of text, an image, or a paired combination of text and image.
Documents can then be retrieved given a user query in text form.
The model supports three embedding modalities: `text`, `image`, and `text_image`.

!!! note

    This library is the NeMo Retriever Library.


## Configure and Run the Multimodal NIM

Use the following procedure to configure and run the multimodal embedding NIM locally.

1. Set the embedding model in your .env file. This tells NeMo Retriever Library to use the Llama Nemotron Embed VL model instead of the default text-only embedding model.

    ```
    EMBEDDING_IMAGE=nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2
    EMBEDDING_TAG=1.12.0
    EMBEDDING_NIM_MODEL_NAME=nvidia/llama-nemotron-embed-vl-1b-v2
    ```

2. Start the NeMo Retriever Library services. The multimodal embedding service is included by default.

    ```
    docker compose --profile retrieval up
    ```


After the services are running, you can interact with the extraction pipeline by using Python.
The key to leveraging the multimodal model is
to configure the `extract` and `embed` methods to process different content types with the appropriate modality.


## Supported Modalities

The multimodal embedding model supports three modalities:

- **`text`** — Embed content as plain text. This is the default modality. It is fast and provides a good baseline for retrieval.
- **`image`** — Embed content as an image. This captures the visual and spatial layout of the content, which is useful for tables, charts, and infographics.
- **`text_image`** — Embed paired text and image together. This combines the semantic richness of text with the visual layout of an image for the best retrieval quality.


## Per-Element Modality Control

Different modalities can be applied to different content types by passing per-element modality parameters to the `embed` method:

- **`text_elements_modality`** — Modality for text elements (default: `"text"`).
- **`structured_elements_modality`** — Modality for tables and charts (default: `"text"`).
- **`image_elements_modality`** — Modality for images, including page images (default: `"text"`).

This allows you to, for example, embed plain text as text while embedding tables as images or as paired text+image.


## Example 1: Default Text-Based Embedding

When you use the multimodal model, by default, all extracted content (text, tables, charts) is treated as plain text.
The following example provides a strong baseline for retrieval.

- The `extract` method is configured to pull out text, tables, and charts.
- The `embed` method is called with no arguments.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=False,
    )
    .embed() # Default behavior embeds all content as text
)
results = ingestor.ingest()
```


## Example 2: Structured Elements as Images

It is common to process PDFs by embedding standard text as text, and embed visual elements like tables and charts as images.
The following example enables the multimodal model to capture the spatial and structural information of the visual content.

- The `extract` method is configured to pull out text, tables, and charts.
- The `embed` method is configured with `structured_elements_modality="image"` to embed the extracted tables and charts as images.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=False,
    )
    .embed(
        structured_elements_modality="image",
    )
)
results = ingestor.ingest()
```


## Example 3: Structured Elements as Text+Image Pairs

For the highest quality retrieval on tables and charts, you can embed them as paired text+image.
This combines the extracted table text content with the rendered table image, giving the model both semantic and visual context.

- The `extract` method is configured to pull out text, tables, and charts.
- The `embed` method is configured with `structured_elements_modality="text_image"` to embed tables and charts as paired text+image.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=False,
    )
    .embed(
        structured_elements_modality="text_image",
    )
)
results = ingestor.ingest()
```


## Example 4: Full Page as Image

For documents where the entire page layout is important (such as infographics, complex diagrams, or forms),
you can configure NeMo Retriever Library to treat every page as a single image.
The following example extracts and embeds each page as an image.

- The `extract` method uses the `extract_page_as_image=True` parameter. All other extraction types are set to `False`.
- The `embed` method processes the page images with `image_elements_modality="image"`.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract(
        extract_text=False,
        extract_tables=False,
        extract_charts=False,
        extract_images=False,
        extract_page_as_image=True,
    )
    .embed(
        image_elements_modality="image",
    )
)
results = ingestor.ingest()
```


## Example 5: Full Page as Text+Image

For the best retrieval quality on full-page content, you can embed each page as a paired text+image.
When `image_elements_modality="text_image"` is set, the pipeline automatically aggregates text and structured content
(tables, charts) from each page and pairs the aggregated text with the page image.

This auto-enables the `image_elements_aggregate_page_content` behavior, which collects TEXT and STRUCTURED content
from each page and stores it alongside the page image for joint embedding.

- The `extract` method extracts both page images and text content. The text and structured content are used for aggregation.
- The `embed` method processes the page images with `image_elements_modality="text_image"`.
- When page content is aggregated, TEXT and STRUCTURED element embeddings are automatically skipped to avoid duplication.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=False,
        extract_page_as_image=True,
    )
    .embed(
        image_elements_modality="text_image",
    )
)
results = ingestor.ingest()
```

## Related Topics

- [Support Matrix](support-matrix.md)
- [Troubleshoot NeMo Retriever Library](troubleshoot.md)
- [Use the NeMo Retriever Python API](python-api-reference.md)
- [Extract Captions from Images](python-api-reference.md#extract-captions-from-images)
