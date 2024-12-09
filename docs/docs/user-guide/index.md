# What is NV-Ingest?

NV-Ingest is a scalable, performance-oriented document content and metadata extraction microservice. NV-Ingest uses specialized NVIDIA NIM microservices to find, contextualize, and extract text, tables, charts and images for use in downstream generative applications.

NV-Ingest also enables parallelization of the process of splitting documents into pages where contents are classified (such as tables, charts, images, text), extracted into discrete content, and further contextualized through optical character recognition (OCR) into a well defined JSON schema. From there, NVIDIA Ingest can optionally manage computation of embeddings for the extracted content, and optionally manage storing into a vector database [Milvus](https://milvus.io/).


## What NV-Ingest Does ✔️

NV-Ingest is a microservice service that: 

- Accepts a JSON Job description, containing a document payload, and a set of ingestion tasks to perform on that payload.
- Allows the results of a Job to be retrieved; the result is a JSON dictionary containing a list of Metadata describing objects extracted from the base document, and processing annotations and timing/trace data.
- Supports .pdf, .docx, .pptx, and images.
- Supports multiple methods of extraction for each document type to balance trade-offs between throughput and accuracy. For example, for PDF documents, we support extraction through pdfium, Unstructured.io, and Adobe Content Extraction Services.
- Supports various types of pre and post processing operations, including text splitting and chunking, transform and filtering, embedding generation, and image offloading to storage.

## What NV-Ingest Doesn't ✖️

NV-Ingest isn't a service that:

- Runs a static pipeline or fixed set of operations on every submitted document.
- Acts as a wrapper for any specific document parsing library.
