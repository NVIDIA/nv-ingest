# Support Matrix for NeMo Retriever Extraction

Before you begin using [NeMo Retriever extraction](overview.md), ensure that you have the hardware for your use case.


## Hardware

| GPU    | Family      | Memory | Minimum GPUs |
|--------|-------------|--------|--------------|
| H100   | SXM or PCIe | 80GB   | 1            |
| A100   | SXM or PCIe | 80GB   | 1            |
| A10G   | —           | 24GB   | 1            |
| L40S   | —           | 48GB   | 1            |


The core pipeline requires approximately 150GB disk space. 
To run the core pipeline and all optional features, you need approximately 210GB disk space.


## Advanced Feature Support

Some advanced features, such as VLM integrations and audio extraction, require additional GPU support and disk space. 
For more information, refer to [Profile Information](quickstart-guide.md#profile-information).


### VLM Integrations

- **nemoretriever-parse VLM NIM** — NeMo Retriever is compatible with [nemoretriever-parse VLM NIM](https://build.nvidia.com/nvidia/nemoretriever-parse), which adds state-of-the-art text and table extraction. To integrate this NIM into the nv-ingest pipeline, you need 1 additional GPU (H100, A100, A10G, L40S). Nemo Retriever parse requires ~16GB additional disk space.
- **Llama3.2 Vision VLM NIMs** — NeMo Retriever is compatible with the [Llama3.2 VLM NIMs](https://build.nvidia.com/meta/llama-3.2-11b-vision-instruct/modelcard) for image captioning capabilities. To integrate these NIM into the nv-ingest pipeline, you need 1 additional GPU (H100, A100, A10G, L40S). Image captioning requires ~16GB additional disk space.


### Audio Extraction (Early Access)

- **RIVA NIM** — NeMo Retriever can retrieve across audio files by using the [RIVA NIMs](https://docs.nvidia.com/nim/riva/asr/latest/overview.html). To integrate this capability into the nv-ingest pipeline, you need 1 additional GPU (H100, A100, A10G, L40S). Audio extraction requires ~37GB additional disk space.



## Related Topics

- [Prerequisites](prerequisites.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy With Helm](helm.md)
