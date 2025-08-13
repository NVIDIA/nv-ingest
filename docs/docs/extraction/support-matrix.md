# Support Matrix for NeMo Retriever Extraction

Before you begin using [NeMo Retriever extraction](overview.md), ensure that you have the hardware for your use case.

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.


## Core and Advanced Pipeline Features

The Nemo Retriever extraction core pipeline features run on a single A10G or better GPU. 
The core pipeline features include the following:

- llama3.2-nv-embedqa-1b-v2 — Embedding model for converting text chunks into vectors.
- nemoretriever-page-elements-v2 — Detects and classifies images on a page as a table, chart or infographic. 
- nemoretriever-table-structure-v1 — Detects rows, columns, and cells within a table to preserve table structure and convert to Markdown format. 
- nemoretriever-graphic-elements-v1 — Detects graphic elements within chart images such as titles, legends, axes, and numerical values. 
- paddleocr — Image OCR model to detect and extract text from images.
- retrieval — Enables embedding and indexing into Milvus.

Advanced features require additional GPU support and disk space. 
This includes the following:

- Audio extraction — Use [Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html) for processing audio files. For more information, refer to [Audio Processing](nemoretriever-parse.md).
- `nemoretriever-parse` — Use [nemoretriever-parse](https://build.nvidia.com/nvidia/nemoretriever-parse), which adds state-of-the-art text and table extraction. For more information, refer to [Use Nemo Retriever Extraction with nemoretriever-parse](nemoretriever-parse.md).
- VLM image captioning — Use [llama 3.1 nemotron 8B Vision](https://build.nvidia.com/nvidia/llama-3.1-nemotron-nano-vl-8b-v1/modelcard) for experimental image captioning of unstructured images. For more information, refer to [Use Multimodal Embedding](vlm-embed.md).



## Hardware Requirements

NeMo Retriever extraction supports the following GPU hardware.

- [RTX Pro 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/)
- [A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/)
- [A10G Tensor Core GPU](https://aws.amazon.com/ec2/instance-types/g5/)
- [L40S](https://www.nvidia.com/en-us/data-center/l40s/)

The following are the hardware requirements to run NeMo Retriever extraction.

| GPU Option                                    | RTX Pro 6000  | B200          | H100        | A100        | A10G   | L40S   |
|-----------------------------------------------|---------------|---------------|-------------|-------------|--------|--------|
| Family                                        | PCIe          | SXM           | SXM or PCIe | SXM or PCIe | —      | —      |
| Memory                                        | 96GB          | 192GB         | 80GB        | 80GB        | 24GB   | 48GB   |
| Core Features Total GPUs                      | 1             | 1             | 1           | 1           | 1      | 1      |
| Core Features Total Disk Space                | ~150GB        | ~150GB        | ~150GB      | ~150GB      | ~150GB | ~150GB |
| Audio Additional Dedicated GPUs               | 1             | 1             | 1           | 1           | 1      | 1      |
| Audio Additional Disk Space                   | ~37GB         | ~37GB         | ~37GB       | ~37GB       | ~37GB  | ~37GB  |
| nemoretriever-parse Additional Dedicated GPUs | Not supported | Not supported | 1           | 1           | 1      | 1      |
| nemoretriever-parse Additional Disk Space     | Not supported | Not supported | ~16GB       | ~16GB       | ~16GB  | ~16GB  |
| VLM Additional Dedicated GPUs                 | 1             | 1             | 1           | 1           | 1      | 1      |
| VLM Additional Disk Space                     | ~16GB         | ~16GB         | ~16GB       | ~16GB       | ~16GB  | ~16GB  |


!!! note

    Release 25.08 does not support A10G and L40S hardware. For details, refer to [Release Notes](releasenotes-nv-ingest.md).



## Related Topics

- [Prerequisites](prerequisites.md)
- [Release Notes](releasenotes-nv-ingest.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy With Helm](helm.md)
