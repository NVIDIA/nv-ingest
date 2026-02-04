# Use Multimodal Embedding with NeMo Retriever Extraction

This documentation describes how to use [NeMo Retriever extraction](overview.md) 
with the multimodal embedding model [Llama 3.2 NeMo Retriever Multimodal Embedding 1B](https://build.nvidia.com/nvidia/llama-3_2-nemoretriever-1b-vlm-embed-v1).

The `Llama 3.2 NeMo Retriever Multimodal Embedding 1B` model is optimized for multimodal question-answering retrieval. 
The model can embed documents in the form of an image, text, or a combination of image and text. 
Documents can then be retrieved given a user query in text form. 
The model supports images that contain text, tables, charts, and infographics.

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.


## Configure and Run the Multimodal NIM

Use the following procedure to configure and run the multimodal embedding NIM locally.

1. Set the embedding model in your .env file. This tells NeMo Retriever extraction to use the Llama 3.2 Multimodal model instead of the default text-only embedding model.

    ```
    EMBEDDING_IMAGE=nvcr.io/nvidia/nemo-microservices/llama-3.2-nemoretriever-1b-vlm-embed-v1
    EMBEDDING_TAG=1.7.0
    EMBEDDING_NIM_MODEL_NAME=nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1
    ```

2. Start the NeMo Retriever extraction services. The multimodal embedding service is included by default.

    ```
    docker compose --profile retrieval up
    ```


After the services are running, you can interact with the extraction pipeline by using Python.
The key to leveraging the multimodal model is 
to configure the `extract` and `embed` methods to process different content types as either text or images.


## Example with Default Text-Based Embedding

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


## Example with Embedding Structured Elements as Images

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


## Example with Embedding Entire PDF Pages as Images

For documents where the entire page layout is important (such as infographics, complex diagrams, or forms), 
you can configure NeMo Retriever extraction to treat every page as a single image.
The following example extracts and embeds each page as an image.

!!! note

    The `extract_page_as_image` feature is experimental. Its behavior may change in future releases.

- The `extract method` uses the `extract_page_as_image=True` parameter. All other extraction types are set to `False`.
- The `embed method` processes the page images.

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


## VLM Captioning for Infographics Example

For documents that contain infographics (visual representations that combine text, images, charts, and diagrams), 
you can use a vision-language model (VLM) to generate descriptive captions that capture the meaning and context of the infographic.
This is particularly useful for infographics because they often contain complex visual information that benefits from natural language descriptions.

!!! note

To use VLM captioning feature, enable the `vlm` profile when you start the NeMo Retriever extraction services. For more information, refer to [Profile Information](quickstart-guide.md#profile-information).

The following example demonstrates two different approaches for processing infographics:

### Approach 1: Extract and Caption Infographics

This approach extracts infographics from the document and generates text captions for them using a VLM.
The captions describe the content and meaning of each infographic.

Use this approach when you need searchable text descriptions of complex visual content. 

- The `extract` method is configured with `extract_infographics=True` to identify and extract infographics.
- The `caption` method calls a VLM to generate descriptive text for each infographic.

```python
ingestor = (
    Ingestor()
    .files("./data/*.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,  # Extract infographics
        extract_images=False,
    )
    .caption(
        prompt="Describe the content and key information in this infographic:",
        reasoning=True,  # Enable reasoning for better caption quality
    )
)
results = ingestor.ingest()
```

### Approach 2: Extract and Embed Infographics as Images

This approach treats infographics as visual elements and embeds them using the multimodal embedding model, 
preserving their spatial and visual characteristics without generating text captions.

Use this approach when you want to preserve the visual characteristics for similarity search.

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
    )
    .embed(
        structured_elements_modality="image",  # Embed infographics as images
    )
)
results = ingestor.ingest()
```

### Combining Both Approaches

You can also combine captioning and embedding to get both text descriptions and visual embeddings.

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
    )
    .caption(
        prompt="Describe the content and key information in this infographic:",
        reasoning=True,
    )
    .embed(
        structured_elements_modality="image",
    )
)
results = ingestor.ingest()
```


## Related Topics

- [Support Matrix](support-matrix.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
- [Use the NV-Ingest Python API](nv-ingest-python-api.md)
- [Extract Captions from Images](nv-ingest-python-api.md#extract-captions-from-images)