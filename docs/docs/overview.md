# Overview of NVIDIA NeMo Retriever

NVIDIA NeMo Retriever is a collection of microservices 
for building and scaling multimodal data extraction, embedding, and reranking pipelines 
with high accuracy and maximum data privacy – built with NVIDIA NIM.

NeMo Retriever provides the following:

- **Multimodal Data Extraction** — Quickly extract documents at scale that include text, tables, charts, and infographics.
- **Embedding + Indexing** — Embed all extracted text from text chunks and images, and then insert into Milvus - accelerated with NVIDIA cuVS.
- **Retrieval** — Leverage semantic + hybrid search for high accuracy retrieval with the embedding + reranking NIM microservice.



## Enterprise-Ready Features

NVIDIA NeMo Retriever comes with enterprise-ready features, including the following:

- **High Accuracy** — NeMo Retriever exhibits a high level of accuracy when retrieving across various modalities through enterprise documents. 
- **High Throughput** — NeMo Retriever is capable of extracting, embedding, indexing and retrieving across hundreds of thousands of documents at scale with high throughput. 
- **Decomposable/Customizable** — NeMo Retriever consists of modules that can be separately used and deployed in your own environment. 
- **Enterprise-Grade Security** — NeMo Retriever NIMs come with security features such as the use of [safetensors](https://huggingface.co/docs/safetensors/index), continuous patching of CVEs, and more. 



## Applications

The following are some applications that use NVIDIA Nemo Retriever:

- [Document Research Assistant for Blog Creation](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/nvidia_document_research_assistant_for_blog_creation.ipynb) (LlamaIndex Jupyter Notebook)
- [Digital Human for Customer Service](https://github.com/NVIDIA-AI-Blueprints/digital-human) (NVIDIA AI Blueprint)
- [AI Virtual Assistant for Customer Service](https://github.com/NVIDIA-AI-Blueprints/ai-virtual-assistant) (NVIDIA AI Blueprint)
- [Building Code Documentation Agents with CrewAI](https://github.com/crewAIInc/nvidia-demo) (CrewAI Demo)
- [Video Search and Summarization](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) (NVIDIA AI Blueprint)

<!-- [Build an Enterprise RAG Pipeline](https://github.com/NVIDIA-AI-Blueprints/rag/tree/v2.0.0) -->



## Related Topics

- [NeMo Retriever Text Embedding NIM](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html)
- [NeMo Retriever Text Reranking NIM](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/overview.html)
- [NVIDIA NIM for Object Detection](https://docs.nvidia.com/nim/ingestion/object-detection/latest/overview.html)
- [NVIDIA NIM for Table Extraction](https://docs.nvidia.com/nim/ingestion/table-extraction/latest/overview.html)
