# What is NVIDIA Ingest?

NVIDIA Ingest is a scalable, performance-oriented document content and metadata extraction microservice. 
NVIDIA Ingest uses specialized NVIDIA NIM microservices 
to find, contextualize, and extract text, tables, charts and images that you can use in downstream generative applications.

!!! note

    NVIDIA Ingest is also known as NV-Ingest and NeMo Retriever Extraction.

NVIDIA Ingest also enables parallelization of the process of splitting documents into pages where contents are classified (such as tables, charts, images, text), 
extracted into discrete content, and further contextualized through optical character recognition (OCR) into a well defined JSON schema. 
From there, NVIDIA Ingest can optionally manage computation of embeddings for the extracted content, 
and optionally manage storing into a vector database [Milvus](https://milvus.io/).

!!! note

    Cached and Deplot are deprecated. Instead, docker-compose now uses a beta version of the yolox-graphic-elements container. With this change, you should now be able to run nv-ingest on a single 80GB A100 or H100 GPU. If you want to use the old pipeline, with Cached and Deplot, use the [nv-ingest 24.12.1 release](https://github.com/NVIDIA/nv-ingest/tree/24.12.1).


## What NVIDIA Ingest Is ✔️

NVIDIA Ingest is a microservice service that does the following:

- Accept a JSON job description, containing a document payload, and a set of ingestion tasks to perform on that payload.
- Allow the results of a job to be retrieved. The result is a JSON dictionary that contains a list of metadata describing objects extracted from the base document, and processing annotations and timing/trace data.
- Support multiple methods of extraction for each document type to balance trade-offs between throughput and accuracy. For example, for .pdf documents, we support extraction through pdfium, Unstructured.io, and Adobe Content Extraction Services.
- Support various types of pre- and post- processing operations, including text splitting and chunking, transform and filtering, embedding generation, and image offloading to storage.

NVIDIA Ingest supports the following file types:

- `pdf`
- `docx`
- `pptx`
- `jpeg`
- `png`
- `svg`
- `tiff`
- `txt`


## What NVIDIA Ingest Isn't ✖️

NVIDIA Ingest does not do the following:

- Run a static pipeline or fixed set of operations on every submitted document.
- Act as a wrapper for any specific document parsing library.
