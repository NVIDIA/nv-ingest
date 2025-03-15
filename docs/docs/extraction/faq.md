# Frequently Asked Questions for NeMo Retriever Extraction

This documentation contains the Frequently Asked Questions (FAQ) for [NeMo Retriever extraction](overview.md).

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.



## What if I already have a retrieval pipeline? Can I just use NeMo Retriever extraction? 

You can use the nv-ingest-cli or Python APIs to perform extraction only, and then consume the results.
Using the Python API, `results` is a list object with one entry.
For code examples, see the Jupyter notebooks [Multimodal RAG with LlamaIndex](https://github.com/NVIDIA/nv-ingest/blob/main/examples/llama_index_multimodal_rag.ipynb) 
and [Multimodal RAG with LangChain](https://github.com/NVIDIA/nv-ingest/blob/main/examples/langchain_multimodal_rag.ipynb).



## Where does NeMo Retriever extraction (nv-ingest) ingest to?

NeMo Retriever extraction supports extracting text representations of various forms of content, 
and ingesting to the [Milvus vector database](https://milvus.io/). 
NeMo Retriever extraction does not store data on disk except through Milvus and its underlying Minio object store. 
You can ingest to other data stores; however, you must configure other data stores yourself. 
For more information, refer to [Data Upload](data-store.md).



## How would I process unstructured images?

For images that `nemoretriever-page-elements-v2` does not classify as tables, charts, or infographics, 
you can use our VLM caption task to create a dense caption of the detected image. 
That caption is then be embedded along with the rest of your content. 
For more information, refer to [Extract Captions from Images](nv-ingest-python-api.md#extract-captions-from-images).



## When should I consider using nemoretriever-parse?

For scanned documents, or documents with complex layouts, 
we recommend that you use [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse). 
Nemo Retriever parse provides higher-accuracy text extraction. 
For more information, refer to [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md).



## Why are the environment variables different between library mode and self-hosted mode?

### Self-Hosted Deployments

For [self-hosted deployments](quickstart-guide.md), you should set the environment variables `NGC_API_KEY` and `NIM_NGC_API_KEY`.
For more information, refer to [Generate Your NGC Keys](ngc-api-key.md).

For advanced scenarios, you might want to set `docker-compose` environment variables for NIM container paths, tags, and batch sizes. 
You can set those directly in `docker-compose.yaml`, or in an [environment variable file](environment-config.md) that docker compose uses.

### Library Mode

For [library mode](quickstart-library-mode.md), you should set the environment variables `NVIDIA_BUILD_API_KEY` and `NVIDIA_API_KEY`. 
For more information, refer to [Generate Your NGC Keys](ngc-api-key.md).

For advanced scenarios, you might want to use library mode with self-hosted NIM instances. 
You can set custom endpoints for each NIM. 
For examples of `*_ENDPOINT` variables, refer to [nv-ingest/docker-compose.yaml](https://github.com/NVIDIA/nv-ingest/blob/main/docker-compose.yaml).







## What parameters or settings can I adjust to optimize extraction from my documents or data? 

See the [Profile Information](quickstart-guide.md#profile-information) section 
for information about the optional NIM components of the pipeline.

You can configure the `extract`, `caption`, and other tasks by using the [Ingestor API](nv-ingest-python-api.md).

To choose what types of content to extract, use code similar to the following. 
For more information, refer to [Extract Specific Elements from PDFs](nv-ingest-python-api.md#extract-specific-elements-from-pdfs).

```python
Ingestor(client=client)
    .files("data/multimodal_test.pdf")
    .extract(              
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=True,
        paddle_output_format="markdown",
        extract_infographics=True,
        text_depth="page"
    )
```

To generate captions for images, use code similar to the following.
For more information, refer to [Extract Captions from Images](nv-ingest-python-api.md#extract-captions-from-images).

```python
Ingestor(client=client)
    .files("data/multimodal_test.pdf")
    .extract()
    .embed()
    .caption()
)
```
