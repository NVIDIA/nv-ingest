## Use Multimodal Embedding

This documentation describes how to use [NeMo Retriever extraction](overview.md)
with the [Llama 3.2 NeMo Retriever Multimodal Embedding 1B](https://build.nvidia.com/nvidia/llama-3_2-nemoretriever-1b-vlm-embed-v1) model.

The Llama 3.2 NeMo Retriever Multimodal Embedding 1B model is optimized for multimodal question-answering retrieval.
The model can embed 'documents' in the form of an image, text, or a combination of image and text.
Documents can then be retrieved given a user query in text form. The model supports images containing text, tables, charts, and infographics.

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.


## Configure and Run the Multimodal NIM

Use the following procedure to configure and run the multimodal embedding NIM locally.

1. Set the embedding model in your `.env` file. This tells nv-ingest to use the Llama 3.2 Multimodal model instead of the default text-only embedder.

    ```
    EMBEDDING_IMAGE=nvcr.io/nvidia/nemo-microservices/llama-3.2-nemoretriever-1b-vlm-embed-v1
    EMBEDDING_TAG=1.7.0
    EMBEDDING_NIM_MODEL_NAME=nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1
    ```

2. Start the nv-ingest services. The multimodal embedding service is included by default.

    ```
    docker compose --profile retrieval --profile table-structure up
    ```


## Usage Examples

After the services are running, you can interact with nv-ingest by using Python.
The key to leveraging the multimodal model is to configure the `.extract()` and `.embed()` methods to process different content types as either text or images.

### Default Text-Based Embedding

**This is the default pipeline.**
Even when using the multimodal model, all extracted content (text, tables, charts) is treated as plain text by default.
This provides a strong baseline for retrieval.

- The `extract` method is configured to pull out text and structured content.
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

### Embedding Structured Elements (Tables, Charts) as Images

This is a common use case where standard text from a PDF is embedded as text, but visual elements like tables and charts are embedded as images.
This allows the VLM to capture the spatial and structural information of the visual content.

- The `extract` method is configured to pull out text, tables, and charts.
- The `embed` method is configured with `structured_elements_modality="image"` to instruct the VLM to embed the extracted tables and charts as images.

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

### Embedding Entire PDF Pages as Images

For documents where the entire page layout is important (such as infographics, complex diagrams, or forms),
you can configure nv-ingest to treat every page as a single image.

!!! note "Experimental Feature"

    The `extract_page_as_image` feature is experimental. Its behavior may change in future releases.

- The `extract method` uses the `extract_page_as_image=True` flag. All other extraction types should be set to False.
- The `embed method` then processes these page images.

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
