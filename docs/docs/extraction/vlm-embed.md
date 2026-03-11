# Use Multimodal Embedding with NeMo Retriever Library

This guide explains how to use the [NeMo Retriever Library](overview.md) with the multimodal embedding model [Llama Nemotron Embed VL 1B v2](https://build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2).

The `Llama Nemotron Embed VL 1B v2` model is optimized for multimodal question-answering and retrieval tasks.
It can embed documents as text, images, or paired text-image combinations.
These embeddings enable retrieving relevant documents based on a text query.
The model supports three embedding modalities: `text`, `image`, and `text_image`.

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed to the NeMo Retriever Library.


## Configure and Run the Multimodal NIM

Use the following procedure to configure and run the multimodal embedding NIM locally.

1. Configure the embedding model in your `.env` file. This instructs the NeMo Retriever Library to use the Llama Nemotron Embed VL model instead of the default text-only model.

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
The key to using the multimodal model effectively is configuring the `extract` and `embed` methods to handle different content types with the correct modality.


## Supported Modalities

The multimodal embedding model supports three modalities:

- **`text`** – Embeds content as plain text. This is the default modality and provides a strong baseline for retrieval.
- **`image`** – Embeds content as an image, capturing visual and spatial layout details that are helpful for tables, charts, and infographics.
- **`text_image`** – Embeds paired text and image together, combining the semantic depth of text with the visual context of an image for higher retrieval quality.


## Per-Element Modality Control

You can apply different modalities to various content types by passing per-element modality parameters to the embed method:

- **`text_elements_modality`** – Specifies the modality for text elements (default: "text").
- **`structured_elements_modality`** – Specifies the modality for tables and charts (default: "text").
- **`image_elements_modality`** – Specifies the modality for images, including page images (default: "text").

This configuration lets you, for example, embed plain text as text while embedding tables as images or as combined text and image.


## Example 1: Default Text-Based Embedding

By default, when you use the multimodal model, all extracted content—such as text, tables, and charts—is processed as plain text.
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

It is common to process PDFs by embedding regular text as text and embedding visual elements, such as tables and charts, as images.
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

For the highest-quality retrieval of tables and charts, embed them as paired text and image.
This approach combines the extracted table text with the rendered table image, giving the model both semantic and visual context.

- The `extract` method is configured to capture text, tables, and charts.
- The embed method is configured with `structured_elements_modality="text_image"` so that tables and charts are embedded as paired text and image.

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

For documents where the full page layout matters (such as infographics, complex diagrams, or forms), you can configure NeMo Retriever Library to treat each page as a single image.
In the following example, every page is extracted and embedded as an image.

- The `extract` method uses `extract_page_as_image=True`, with all other extraction options set to `False`.
- The `embed` method then processes these page images with `image_elements_modality="image"`.

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

For the best retrieval quality on full-page content, you can embed each page as a paired text and image.
When `image_elements_modality="text_image"` is set, the pipeline automatically aggregates the text content from each page and pairs it with the page image for joint embedding.

- The `extract` method extracts both page images and text content, aggregating the text and pairing it with the corresponding page image.
- The `embed` method processes the page images with `image_elements_modality="text_image"`.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
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
- [Use the NeMo Retriever Library Python API](python-api-reference.md)
- [Extract Captions from Images](python-api-reference.md#extract-captions-from-images)
