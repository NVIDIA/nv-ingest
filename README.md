<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<div align="center">
  <h1>NeMo Retriever Extraction</h1>
  <h4>Scalable, performance-oriented document content and metadata extraction microservice</h4>
</div>

NVIDIA NeMo Retriever Extraction (also known as NVIDIA Ingest or nv-ingest) is a comprehensive microservice for extracting structured content from unstructured documents. It uses specialized NVIDIA NIM microservices to find, contextualize, and extract text, tables, charts, and infographics that you can use in downstream generative applications including RAG pipelines.

NeMo Retriever Extraction enables parallelization of splitting documents into pages where artifacts are classified (such as text, tables, charts, and infographics), extracted, and further contextualized through optical character recognition (OCR) into a well-defined JSON schema. From there, NeMo Retriever Extraction can optionally compute embeddings for the extracted content and manage storing into a vector database like [Milvus](https://milvus.io/).

<p align="center">
  <img src="https://docs.nvidia.com/nemo/retriever/extraction/images/overview-extraction.png" width="800">
  <br>
  <em>NeMo Retriever Extraction pipeline architecture showing parallel processing and NIM integration.</em>
</p>

## ⚡ Quick Start

```bash
# Create a Python 3.12 environment
uv venv --python 3.12 nvingest && \
  source nvingest/bin/activate && \
  uv pip install nv-ingest==26.1.2 nv-ingest-api==26.1.2 nv-ingest-client==26.1.2 milvus-lite==2.4.12

# Set your NVIDIA API key (get one at build.nvidia.com)
export NVIDIA_API_KEY=nvapi-...

# Run the Python example below to ingest your first document
python ingest_example.py
```

## Recent News

- 02/03/2026 Updated documentation with improved benchmarking guides and MIG deployment instructions
- 01/15/2026 **v26.1.2 released!** - Transition to yolox-graphic-elements NIM, enabling single A10G (24GB) GPU deployments
- 12/10/2025 v24.12.1 - Last release with Cached and Deplot models (deprecated)
- 11/20/2025 Added support for video formats (AVI, MKV, MOV, MP4) in early access
- 10/15/2025 Helm chart updates for improved Kubernetes deployment

> [!Note]
> Cached and Deplot are deprecated. NeMo Retriever Extraction now uses the yolox-graphic-elements NIM, enabling deployment on a single 24GB A10G or better GPU. To use the old pipeline, see the [v24.12.1 release](https://github.com/NVIDIA/nv-ingest/tree/24.12.1).

## Code Overview

NeMo Retriever Extraction provides flexible deployment options and extraction methods to meet different performance and accuracy requirements. The system integrates with multiple NVIDIA NIM microservices for optimal extraction quality.

<details>
<summary><b>(Click to expand) NVIDIA NIM Microservices Integration</b></summary>
<small>

| NIM Microservice | Purpose | Model | GPU Memory | Status |
|------------------|---------|-------|------------|--------|
| **nemotron-parse** | Document parsing & layout | Nemotron Parse | 16GB+ | ✅ Stable |
| **yolox-graphic-elements** | Chart/infographic detection | YOLOX-based | 8GB+ | ✅ Stable |
| **nemoretriever-ocr-v1** | Optical character recognition | NeMo Retriever OCR | 8GB+ | ✅ Stable |
| **nemoretriever-table-structure-v1** | Table structure extraction | NeMo Retriever Table | 8GB+ | ✅ Stable |
| **llama-3.2-nv-embedqa-1b-v2** | Text embedding generation | Llama 3.2 1B | 4GB+ | ✅ Stable |
| **llama-3.2-nv-rerankqa-1b-v2** | Result reranking | Llama 3.2 1B | 4GB+ | ✅ Stable |

</small>
</details>

<details>
<summary><b>(Click to expand) Extraction Methods Support Matrix</b></summary>
<small>

| Extraction Method | Description | Throughput | Accuracy | GPU Required | Status |
|-------------------|-------------|------------|----------|--------------|--------|
| **pdfium** | Fast native PDF parsing | ⚡⚡⚡ High | ⭐⭐ Good | ❌ No | ✅ Stable |
| **nemotron-parse** | NVIDIA NIM for maximum accuracy | ⚡ Lower | ⭐⭐⭐ Excellent | ✅ Yes | ✅ Stable |
| **unstructured.io** | Community-driven extraction | ⚡⚡ Medium | ⭐⭐ Good | ❌ No | ✅ Stable |
| **Adobe PDF Services** | Enterprise PDF extraction | ⚡⚡ Medium | ⭐⭐⭐ Excellent | ❌ No | ✅ Stable |
| **yolox-graphic-elements** | Chart/infographic detection | ⚡⚡ Medium | ⭐⭐⭐ Excellent | ✅ Yes | ✅ Stable |

</small>
</details>

<details>
<summary><b>(Click to expand) Supported File Formats</b></summary>
<small>

| Format | Status | Format | Status | Format | Status |
|--------|--------|--------|--------|--------|--------|
| `pdf` | ✅ Stable | `docx` | ✅ Stable | `pptx` | ✅ Stable |
| `png` | ✅ Stable | `jpeg` | ✅ Stable | `bmp` | ✅ Stable |
| `tiff` | ✅ Stable | `html` | ✅ Stable | `txt` | ✅ Stable |
| `md` | ✅ Stable | `json` | ✅ Stable | `sh` | ✅ Stable |
| `wav` | ✅ Stable | `mp3` | ✅ Stable | | |
| `avi` | 🚧 Early Access | `mkv` | 🚧 Early Access | `mov` | 🚧 Early Access |
| `mp4` | 🚧 Early Access | | | | |

</small>
</details>

## Features

**NeMo Retriever Extraction provides:**
- 📄 **Flexible Extraction** - Multiple extraction methods per document type to balance throughput vs. accuracy (pdfium, [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse), Unstructured.io, Adobe)
- 🚀 **Parallel Processing** - Automatic parallelization and page-level splitting for maximum throughput
- 🎯 **Content Classification** - Automatic detection and extraction of text, tables, charts, and infographics
- 🔍 **OCR Integration** - Contextual optical character recognition for image-based content
- 🧮 **Embeddings & VDB** - Built-in embedding generation and vector database integration (Milvus)
- ⚙️ **Configurable Pipelines** - Customizable pre/post-processing: splitting, chunking, filtering, transforms
- 📊 **Observability** - Built-in metrics, tracing, and timing data for production monitoring

**NeMo Retriever Extraction is NOT:**
- ❌ A static pipeline with fixed operations
- ❌ A wrapper for any single document parsing library

## Documentation Resources

- **[Official Documentation](https://docs.nvidia.com/nemo/retriever/extraction/)** - Complete user guides, API references, and deployment instructions
- **[Getting Started Guide](https://docs.nvidia.com/nemo/retriever/extraction/overview/)** - Overview and prerequisites for production deployments
- **[Benchmarking Guide](https://docs.nvidia.com/nemo/retriever/extraction/benchmarking/)** - Performance testing and recall evaluation framework
- **[MIG Deployment](https://docs.nvidia.com/nemo/retriever/extraction/mig-benchmarking/)** - Multi-Instance GPU configurations for Kubernetes
- **[API Documentation](https://docs.nvidia.com/nemo/retriever/extraction/api/)** - Python client and API reference


## Getting Started

### Deployment Options

NeMo Retriever Extraction supports multiple deployment modes depending on your scale and requirements:

<details>
<summary><b>(Click to expand) Deployment Modes Support Matrix</b></summary>
<small>

| Deployment Mode | Scale | GPU Required | NIMs | Observability | MIG Support | Status |
|-----------------|-------|--------------|------|---------------|-------------|--------|
| **Library Mode** | < 100 docs | ❌ No | ✅ build.nvidia.com | ⚠️ Basic | ❌ No | ✅ Active |
| **Docker Compose** | Production | ✅ Yes | ✅ Self-hosted | ✅ Full (OTEL/Prometheus) | ❌ No | ✅ Active |
| **Kubernetes/Helm** | Enterprise | ✅ Yes | ✅ Self-hosted | ✅ Full (OTEL/Prometheus) | ✅ Yes | ✅ Active |

</small>
</details>

For production deployments, see the [prerequisites documentation](https://docs.nvidia.com/nv-ingest/user-guide/getting-started/prerequisites).

### Library Mode Quick Start

Library mode is ideal for small-scale workloads (< 100 documents) and rapid prototyping.

**Requirements:**
- Linux (Ubuntu 22.04+) or macOS
- Python 3.12
- Isolated virtual environment ([uv](https://docs.astral.sh/uv/getting-started/installation/) or [conda](https://github.com/conda-forge/miniforge) recommended)

**Step 1: Install NeMo Retriever Extraction**

```bash
# Create and activate a Python 3.12 virtual environment
uv venv --python 3.12 nvingest && source nvingest/bin/activate

# Install nv-ingest packages
uv pip install nv-ingest==26.1.2 nv-ingest-api==26.1.2 nv-ingest-client==26.1.2 milvus-lite==2.4.12

# Set your NVIDIA API key (get one at https://org.ngc.nvidia.com/setup/api-keys)
export NVIDIA_API_KEY=nvapi-...
```

**Step 2: Ingest Your First Document**

The following Python example extracts content from a PDF and stores it in a local Milvus vector database. On a 4-core laptop, this should complete in ~10 seconds.

```python
import time
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

def main():
    # Start the pipeline subprocess for library mode
    run_pipeline(block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    # Configure extraction and embedding pipeline
    ingestor = (
        Ingestor(client=client)
        .files("data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            table_output_format="markdown",
            extract_infographics=True,
            text_depth="page",
        )
        .embed()
        .vdb_upload(
            collection_name="test",
            milvus_uri="milvus.db",
            sparse=False,
            dense_dim=2048,  # Use 1024 for e5-v5 embedder
        )
    )

    print("Starting ingestion...")
    t0 = time.time()
    
    # Execute ingestion
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    
    print(f"Total time: {time.time() - t0:.2f} seconds")
    
    # Inspect results
    if results:
        print(ingest_json_results_to_blob(results[0]))
    if failures:
        print(f"Failures: {len(failures)}")

if __name__ == "__main__":
    main()
```

<details>
<summary><b>Example Output</b></summary>

```
Starting ingestion...
Total time: 9.24 seconds

TestingDocument
A sample document with headings and placeholder text
Introduction
This is a placeholder document that can be used for any purpose...
Table 1
Animal Activity Place
Giraffe Driving a car At the beach
Lion Putting on sunscreen At the park
Cat Jumping onto a laptop In a home office
Dog Chasing a squirrel In the front yard
...
```
</details>

> [!TIP]
> - For maximum accuracy on scanned PDFs, use `extract_method="nemotron_parse"` (slower but more accurate)
> - On high-CPU systems (>16 cores), use `taskset -c 0-3 python script.py` to limit CPU usage
> - Integrate with [LlamaIndex](examples/llama_index_multimodal_rag.ipynb) or [LangChain](examples/langchain_multimodal_rag.ipynb) for advanced RAG pipelines

**Step 3: Query Ingested Content**

Use the extracted content with an LLM to generate answers:

```python
import os
from openai import OpenAI
from nv_ingest_client.util.milvus import nvingest_retrieval

# Retrieve relevant content
queries = ["Which animal is responsible for the typos?"]
retrieved_docs = nvingest_retrieval(
    queries,
    collection_name="test",
    milvus_uri="milvus.db",
    hybrid=False,
    top_k=1,
)

# Generate answer using LLM
extract = retrieved_docs[0][0]["entity"]["text"]
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

prompt = f"Using the following content: {extract}\n\nAnswer: {queries[0]}"
completion = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    messages=[{"role": "user", "content": prompt}],
)

print(f"Answer: {completion.choices[0].message.content}")
```

<details>
<summary><b>Example Output</b></summary>

```
Answer: Based on the table showing animals and their activities, the **Cat** is the potential 
culprit, since it's located "In a home office" and engaged in "Jumping onto a laptop", which 
could theoretically lead to accidental keystrokes or typos!
```
</details>

> [!TIP]
> **Next Steps:**
> - Try the [live demo on build.nvidia.com](https://build.nvidia.com/nvidia/multimodal-pdf-data-extraction-for-enterprise-rag)
> - Integrate with [LlamaIndex](examples/llama_index_multimodal_rag.ipynb) or [LangChain](examples/langchain_multimodal_rag.ipynb)
> - Explore [video and audio extraction examples](examples/)


## Benchmarking & Testing

NeMo Retriever Extraction includes a comprehensive testing framework for performance benchmarking and retrieval accuracy evaluation.

### Quick Start

```bash
cd tools/harness && uv sync

# Run end-to-end performance benchmark
uv run nv-ingest-harness-run --case=e2e --dataset=bo767

# Evaluate retrieval accuracy (recall@k)
uv run nv-ingest-harness-run --case=e2e_recall --dataset=bo767
```

### Available Test Cases

| Test Case | Description | Metrics |
|-----------|-------------|---------|
| **e2e** | End-to-end performance | Throughput, latency, resource utilization |
| **e2e_recall** | Retrieval accuracy | Recall@k against ground truth |
| **mig** | Multi-Instance GPU | Performance with GPU partitioning |

### Benchmark Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| **bo767** | PDF documents with ground truth | 767 docs |
| **bo20** | Quick validation set | 20 docs |
| **single** | Single multimodal PDF | 1 doc |
| **earnings** | Earnings reports (PPT/PDF) | Varied |
| **financebench** | Financial documents | Varied |
| **custom** | Your own datasets | Custom |

**Documentation:**
- [Testing Framework Guide](https://docs.nvidia.com/nemo/retriever/extraction/benchmarking/) - Complete benchmarking documentation
- [MIG Deployment Guide](https://docs.nvidia.com/nemo/retriever/extraction/mig-benchmarking/) - Multi-tenant GPU configurations


## Repository Structure

| Directory | Description |
|-----------|-------------|
| [`api/`](api) | Core API logic and interfaces shared across Python modules |
| [`client/`](client) | Python client library (`nv-ingest-client`) with examples and CLI |
| [`src/`](src) | Core extraction pipeline and service implementation |
| [`examples/`](examples) | Jupyter notebooks and Python scripts demonstrating integration |
| [`docs/`](docs) | Source files for documentation (published to docs.nvidia.com) |
| [`helm/`](helm) | Kubernetes Helm charts for production deployment |
| [`tools/harness/`](tools/harness) | Benchmarking and testing framework |
| [`tests/`](tests) | Unit tests and integration tests |
| [`config/`](config) | Configuration files for observability (OTEL, Prometheus) |
| [`docker/`](docker) | Docker build scripts and entrypoints |
| [`ci/`](ci) | CI/CD scripts and build automation |
| [`conda/`](conda) | Conda packaging and environment definitions |
| [`.devcontainer/`](.devcontainer) | VSCode development container configuration |


## Production Deployment

For production deployments using Docker Compose or Kubernetes, see:
- **[Docker Compose Setup](https://docs.nvidia.com/nemo/retriever/extraction/docker-compose/)** - Deploy with NIMs locally
- **[Kubernetes/Helm Guide](helm/)** - Enterprise-scale deployments
- **[MIG Configuration](https://docs.nvidia.com/nemo/retriever/extraction/mig-benchmarking/)** - Multi-tenant GPU partitioning

## Third-Party Dependencies & Licenses

NeMo Retriever Extraction may optionally download and use third-party software. Review the following license terms before enabling these features:

### Adobe PDF Services SDK
- **Environment Variable:** `INSTALL_ADOBE_SDK=true`
- **Use Case:** Adobe extraction service for PDF decomposition
- **License:** Review the [Adobe PDF Services SDK License](https://github.com/adobe/pdfservices-python-sdk?tab=License-1-ov-file)
- **Package:** https://pypi.org/project/pdfservices-sdk/

### Llama Tokenizer (Built With Llama)
- **Model:** `meta-llama/Llama-3.2-1B` tokenizer (pre-downloaded in container)
- **Use Case:** Token-based text splitting
- **License:** [Llama 3.2 Community License Agreement](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt)
- **Custom Build:** Set `DOWNLOAD_LLAMA_TOKENIZER=True` and `HF_ACCESS_TOKEN=<your-token>` (requires [HuggingFace access](https://huggingface.co/meta-llama/Llama-3.2-1B))

## Contributing

We welcome contributions! All contributors must sign-off on their commits to certify that the contribution is your original work or you have rights to submit it.

**How to sign off:**

```bash
git commit --signoff --message "Add cool feature."
```

This appends `Signed-off-by: Your Name <your@email.com>` to your commit message.

> [!IMPORTANT]
> Commits without sign-off cannot be accepted. This certifies compliance with the [Developer Certificate of Origin (DCO) v1.1](https://developercertificate.org/).

<details>
<summary><b>Developer Certificate of Origin (DCO) - Full Text</b></summary>

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the 
    right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my 
    knowledge, is covered under an appropriate open source license and I 
    have the right under that license to submit that work with modifications, 
    whether created in whole or in part by me, under the same open source 
    license (unless I am permitted to submit under a different license), as 
    indicated in the file; or

(c) The contribution was provided directly to me by some other person who 
    certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public 
    and that a record of the contribution (including all personal information 
    I submit with it, including my sign-off) is maintained indefinitely and 
    may be redistributed consistent with this project or the open source 
    license(s) involved.
```
</details>

For more information, see [CONTRIBUTING.md](CONTRIBUTING.md).


## Security Considerations

> [!WARNING]
> NeMo Retriever Extraction is provided as a reference implementation. Production deployments are the responsibility of end users.

### Security Checklist

When deploying to production, ensure you:

- ✅ **Authentication & Authorization** - Implement AuthN/AuthZ to prevent unauthorized access
- ✅ **Network Security** - Secure all communication channels (TLS/mTLS recommended)
- ✅ **Access Controls** - Define trust boundaries and implement least-privilege access
- ✅ **Monitoring & Logging** - Enable comprehensive logging with secure handling of sensitive data
- ✅ **Container Security** - Use signed, scanned images free of known vulnerabilities
- ✅ **Keep Updated** - Regularly update containers and dependencies with security patches
- ✅ **Security Review** - Have security experts assess your deployment configuration

### Important Notes

- NeMo Retriever Extraction **does not** generate code requiring sandboxing
- NeMo Retriever Extraction **does not** require privileged system access
- Logs may include input/output content - handle logging securely in production
- Missing AuthN/AuthZ can expose services to cost overruns, resource exhaustion, or DoS attacks

For detailed security guidance, see [SECURITY.md](SECURITY.md).

____

## Part of the NeMo Ecosystem

NeMo Retriever Extraction is part of NVIDIA's comprehensive suite of AI and biopharma products. Stay informed about new releases, security updates, and features:

**[Subscribe to NVIDIA Biopharma Product Updates](https://www.nvidia.com/en-us/clara/biopharma/product-updates/)**

### Related Products

- **[NeMo Framework](https://developer.nvidia.com/nemo)** - Enterprise AI framework for training and deploying foundation models
- **[NeMo Retriever](https://docs.nvidia.com/nemo/retriever/)** - Complete RAG pipeline with embedding and reranking
- **[NVIDIA NIMs](https://build.nvidia.com/)** - Optimized inference microservices for AI models
- **[BioNeMo Framework](https://github.com/NVIDIA/bionemo-framework)** - GPU-optimized tools for biomolecular AI

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

Third-party licenses: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
