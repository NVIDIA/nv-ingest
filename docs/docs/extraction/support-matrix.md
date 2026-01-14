# Support Matrix for NeMo Retriever Extraction

Before you begin using [NeMo Retriever extraction](overview.md), ensure that you have the hardware for your use case.

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.


## Core and Advanced Pipeline Features

The Nemo Retriever extraction core pipeline features run on a single A10G or better GPU. 
The core pipeline features include the following:

- llama3.2-nv-embedqa-1b-v2 — Embedding model for converting text chunks into vectors.
- nemoretriever-page-elements-v3 — Detects and classifies images on a page as a table, chart or infographic.
- nemoretriever-table-structure-v1 — Detects rows, columns, and cells within a table to preserve table structure and convert to Markdown format. 
- nemoretriever-graphic-elements-v1 — Detects graphic elements within chart images such as titles, legends, axes, and numerical values. 
- nemoretriever-ocr-v1 — Image OCR model to detect and extract text from images.
- retrieval — Enables embedding and indexing into Milvus.

Advanced features require additional GPU support and disk space. 
This includes the following:

- Audio extraction — Use [Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html) for processing audio files. For more information, refer to [Audio Processing](audio.md).
- Advanced visual parsing — Use [nemotron-parse](https://docs.nvidia.com/nim/vision-language-models/latest/examples/nemotron-parse/overview.html), which adds state-of-the-art text and table extraction. For more information, refer to [Advanced Visual Parsing ](nemoretriever-parse.md).
- VLM image captioning — Use [nemotron-nano-12b-v2-vl](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl/modelcard) for experimental image captioning of unstructured images. For more information, refer to [Extract Captions from Images](nv-ingest-python-api.md#extract-captions-from-images).
- Reranker — Use [llama-3.2-nv-rerankqa-1b-v2](https://build.nvidia.com/nvidia/llama-3.2-nv-rerankqa-1b-v2) for improved retrieval accuracy.



## Hardware Requirements

NeMo Retriever extraction supports the following GPU hardware.

- [RTX Pro 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
- [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [H200 NVL](https://www.nvidia.com/en-us/data-center/h200/)
- [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/)
- [A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/)
- [A10G Tensor Core GPU](https://aws.amazon.com/ec2/instance-types/g5/)
- [L40S](https://www.nvidia.com/en-us/data-center/l40s/)


The following are the hardware requirements to run NeMo Retriever extraction.

|Feature         | GPU Option                | RTX Pro 6000  | B200          | H200 NVL      | H100        | A100 80GB   | A100 40GB     | A10G          | L40S   |
|----------------|---------------------------|---------------|---------------|---------------|-------------|-------------|---------------|---------------|--------|
| GPU            | Memory                    | 96GB          | 180GB         | 141GB         | 80GB        | 80GB        | 40GB          | 24GB          | 48GB   |
| Core Features  | Total GPUs                | 1             | 1             | 1             | 1           | 1           | 1             | 1             | 1      |
| Core Features  | Total Disk Space          | ~150GB        | ~150GB        | ~150GB        | ~150GB      | ~150GB      | ~150GB        | ~150GB        | ~150GB |
| Audio          | Additional Dedicated GPUs | 1             | 1             | 1             | 1           | 1           | 1             | 1             | 1      |
| Audio          | Additional Disk Space     | ~37GB         | ~37GB         | ~37GB         | ~37GB       | ~37GB       | ~37GB         | ~37GB         | ~37GB  |
| nemotron-parse | Additional Dedicated GPUs | Not supported | Not supported | Not supported | 1           | 1           | 1             | 1             | 1      |
| nemotron-parse | Additional Disk Space     | Not supported | Not supported | Not supported | ~16GB       | ~16GB       | ~16GB         | ~16GB         | ~16GB  |
| VLM            | Additional Dedicated GPUs | 1             | 1             | 1             | 1           | 1           | Not supported | Not supported | 1      |
| VLM            | Additional Disk Space     | ~16GB         | ~16GB         | ~16GB         | ~16GB       | ~16GB       | Not supported | Not supported | ~16GB  |
| Reranker       | With Core Pipeline        | Yes           | Yes           | Yes           | Yes         | Yes         | No*           | No*           | No*    |
| Reranker       | Standalone (recall only)  | Yes           | Yes           | Yes           | Yes         | Yes         | Yes           | Yes           | Yes    |

\* GPUs with less than 80GB VRAM cannot run the reranker concurrently with the core pipeline. 
To perform recall testing with the reranker on these GPUs, shut down the core pipeline NIM microservices 
and run only the embedder, reranker, and your vector database.



## Related Topics

- [Prerequisites](prerequisites.md)
- [Release Notes](releasenotes-nv-ingest.md)
- [NVIDIA NIM for Vision Language Models Support Matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html)
- [NVIDIA NVIDIA Riva Support Matrix](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/support-matrix/support-matrix.html)
