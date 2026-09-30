# Quick Start for NeMo Retriever Library

NeMo Retriever Library is a retrieval-augmented generation (RAG) ingestion pipeline for documents that can parse text, tables, charts, and infographics. NeMo Retriever Library parses documents, creates embeddings, optionally stores embeddings in LanceDB, and performs recall evaluation.

This quick start guide shows how to run NeMo Retriever Library as a library in local Python processes without containers. Choose one inference path:

- **Local GPU (Linux):** Pull and run [Nemotron RAG models from Hugging Face](https://huggingface.co/collections/nvidia/nemotron-rag) on your GPU(s). Requires CUDA 13.x and the `[local]` extra.
- **Remote NIM:** Call build.nvidia.com hosted or self-hosted NeMo Retriever NIM endpoints over the network. The base package installs on Linux, Windows x64, and macOS Apple Silicon (arm64); no local GPU is required. macOS Intel (x86_64) is not supported.

The steps below cover environment setup, installation, and a first ingestion run. For Kubernetes or container deployments, refer to [Deployment at a glance](#deployment-at-a-glance) and the [Pre-Requisites & Support Matrix](https://docs.nvidia.com/nemo/retriever/latest/extraction/prerequisites-support-matrix/).

## Deployment at a glance

For Kubernetes deployments, use the **[`nemo_retriever/helm` chart](helm/README.md)** to deploy the retriever **service** and optional in-cluster **NIM** workloads. Published Helm install and upgrade flows for the full extraction stack are documented in the **[NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/)**; use those docs together with the chart README for your release.

For standalone service-image builds and local container runs, see **[`docker.md`](docker.md)**.

## Prerequisites

Before starting, confirm requirements for your inference path (refer to the [Pre-Requisites & Support Matrix](https://docs.nvidia.com/nemo/retriever/latest/extraction/prerequisites-support-matrix/)):

- **Local GPU inference (Linux):** CUDA 13.x with `libcudart.so.13` available, and GPUs visible to the system.
- **Remote NIM inference:** Python 3.12 and network access to your NIM endpoints; CUDA is not required on the client host.

If optical character recognition (OCR) fails with a `libcudart.so.13` error on a local GPU path, install the CUDA 13 runtime for your platform and update `LD_LIBRARY_PATH` to include the CUDA lib64 directory, then rerun the pipeline.

For example, the following command can be used to update the `LD_LIBRARY_PATH` value.

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

## Setup your environment

Complete the following steps to set up your environment. You will create and activate isolated Python and project virtual environments, install the NeMo Retriever Library and its dependencies, and then run the provided ingestion snippets to validate your setup.

1. Create and activate the NeMo Retriever Library environment

Before installing NeMo Retriever Library, create an isolated Python environment so its dependencies do not conflict with other projects on your system. In this step, you set up a new virtual environment and activate it so that all subsequent installs are scoped to NeMo Retriever Library.

In your terminal, run the following commands from any location.

**Local GPU (Linux)**

Install with the `[local]` extra, which includes Nemotron model packages, transformers, and GPU tooling:

```bash
uv venv retriever --python 3.12
source retriever/bin/activate
uv pip install "nemo-retriever[local]"
```

The `[local]` extra resolves stable Nemotron extraction packages by default. To
try prerelease or nightly Nemotron packages from PyPI within the same supported
major-version windows, opt in with `--pre` and pin `nemo-retriever` to the
version you are installing.

Run the following command, and replace `<version>` with that version:

```bash
uv pip install --pre "nemo-retriever[local]==<version>"
```

**Remote NIM (no local GPU)**

The base package is sufficient when all inference runs through hosted or self-hosted NIM endpoints:

```bash
uv python install 3.12
uv venv retriever --python 3.12
source retriever/bin/activate
uv pip install nemo-retriever
```

Install matching **ingestion client** and **ingestion runtime** wheels at the same version when your workflow expects them (refer to the [NeMo Retriever Library prerequisites](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) for the exact PyPI coordinates for your release).

This creates a dedicated Python environment and installs the `nemo-retriever` PyPI package, the canonical distribution for the NeMo Retriever Library. The base install includes the lightweight tokenizer dependencies used for TXT/HTML chunking (no Transformers or local model weights).

If your PDF pipeline uses `method="nemotron_parse"`, install the Nemotron Parse client dependencies with the `nemotron-parse` extra:

```bash
uv pip install "nemo-retriever[nemotron-parse]"
```

For local GPU inference with Nemotron Parse, combine the extras as `nemo-retriever[local,nemotron-parse]`.

> **Note:** `uv python install 3.12` installs a uv-managed Python that includes development headers (`Python.h`). These headers are required by vLLM, which compiles CUDA kernels at runtime using torch inductor. If you skip this step and use a system Python without headers, vLLM actor initialization will fail with `InductorError: fatal error: Python.h: No such file or directory`.

2. Override Torch and Torchvision with CUDA 13 builds (local GPU only)

The `[local]` extra pulls PyTorch from PyPI, which defaults to a CPU build on Linux. Reinstall from the CUDA 13.0 wheel index to match the CUDA runtime required by the Nemotron model packages:

```bash
uv pip uninstall torch torchvision
uv pip install torch==2.11.0 torchvision==0.26.0 -i https://download.pytorch.org/whl/cu130
```

Skip this step if you are using remote NIM inference only.

## Run the pipeline

The [test PDF](../data/multimodal_test.pdf) contains text, tables, charts, and images. Additional test data resides [here](../data/).

> **Note:** `retriever ingest` defaults to local, in-process execution. Use `retriever ingest batch ...` for Ray Data scale-out on larger workloads.
> File formats and internal extraction stages are not separate root commands; configure supported behavior through `retriever ingest`.

The examples below use default local GPU inference (no `invoke_url` specified) and require the `[local]` extra and the CUDA 13 torch override from the setup steps above. For remote NIM inference without a local GPU, refer to [Run with remote inference](#run-with-remote-inference-no-local-gpu-required).

### Ingest a test pdf
```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.io import to_markdown, to_markdown_by_page
from pathlib import Path

documents = [str(Path("../data/multimodal_test.pdf"))]
ingestor = create_ingestor(run_mode="batch")

# ingestion tasks are chainable and defined lazily
ingestor = (
  ingestor.files(documents)
  .extract(
    # below are the default values, but content types can be controlled
    extract_text=True,
    extract_charts=True,
    extract_tables=True,
    extract_infographics=True
  )
  .embed()
  .vdb_upload()
)
```

Bare `.vdb_upload()` writes to the default LanceDB table `nemo-retriever`.
Default `Retriever()` and `retriever ingest` use that same table.

### Ingest inline text

Python callers can pass raw text documents directly to the same text splitting,
embedding, and vector database graph without creating temporary files. Inline
text is supported in `inprocess`, `batch`, and `service` run modes.

```python
from nemo_retriever import create_ingestor

texts = ["some text", "another longer text"]

chunks = (
  create_ingestor(run_mode="batch")
  .texts(texts)
  .embed()
  .vdb_upload()
  .ingest()
)
```

Each string is treated as a raw document and split with `TextChunkParams`
defaults. To override chunking for every text source in the ingest, add
`.extract(split_config={"text": {"max_tokens": 512, "overlap_tokens": 64}})`.
Inline text can be combined with
`.files(...)` and, in modes that support them, `.buffers(...)`; each source is
routed through its matching extractor before the results enter the shared
embedding and sink stages. Inline corpora remain resident in client or driver
memory, so prefer file ingestion when the corpus may exceed the available memory.

### Optional extras

- **`multimedia`** — Audio/video extraction and SVG rendering support. Install this extra when using Parakeet ASR through `extract_method="audio"` so audio decoding and resampling dependencies are available:
  ```bash
  uv pip install "nemo-retriever[multimedia]"
  # or, for local GPU inference:
  uv pip install "nemo-retriever[local,multimedia]"
  ```

Run the batch pipeline script and point it at the directory that contains your PDFs using the following command.

```bash
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py /path/to/pdfs
```

```python
# ingestor.ingest() actually executes the pipeline
chunks = ingestor.ingest()  # pandas.DataFrame (batch and inprocess)
```

### Ingest a test corpus (CLI)

`retriever ingest` accepts a file or a directory and writes a ready-to-query
LanceDB table. From a clone of this repository, `./data/multimodal_test.pdf`
is a valid first-run input. Replace `/path/to/file-or-directory` below with
that PDF or with your own file or directory.

> **Small corpora are supported.** The NeMo Retriever LanceDB adapter requests
> 16 IVF partitions by default. When the table has fewer rows, the adapter
> clamps that count to one less than the row count. An ingest that produces
> 2 through 15 rows still builds an IVF index. A one-row table is stored
> without a vector index, and ingest succeeds. An ingest that produces zero rows
> is a separate empty-result error. Use a larger corpus when you want more
> representative retrieval quality.

```bash
retriever ingest /path/to/file-or-directory \
  --lancedb-uri lancedb \
  --table-name nemo-retriever
```

You do not need to choose a retrieval index mode for the normal workflow. The
default `index_mode=auto` creates a hybrid table (dense
vectors plus BM25/full-text search), and query mode `auto` uses that table
automatically. The explicit `dense`, `hybrid`, and `sparse` modes are advanced
overrides for experiments or specialized deployments.

Chunks land at `./lancedb/nemo-retriever`, which matches the storage settings
used in [Run a recall query](#run-a-recall-query) below. Python
`.vdb_upload()` and default `Retriever()` use the same table. With the
`[local]` extra installed (refer to the setup steps above), defaults point at
local-GPU extraction and embedding.

**No local GPU?** Set [`NVIDIA_API_KEY`](https://nvidia.github.io/NeMo-Retriever/extraction/api-keys/#nvidia-api-key) (refer to [Authentication and API keys](https://nvidia.github.io/NeMo-Retriever/extraction/api-keys/)) and route extraction and embedding
through [build.nvidia.com](https://build.nvidia.com/) NIMs instead:

```bash
export NVIDIA_API_KEY=nvapi-...

retriever ingest /path/to/file-or-directory \
  --lancedb-uri lancedb \
  --table-name nemo-retriever \
  --page-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3 \
  --ocr-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2 \
  --table-structure-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1 \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

> **OCR engine default:** The default OCR engine is **Nemotron OCR v2**. Use
> `--ocr-version v1` to opt into the legacy OCR engine. Local OCR v2 defaults
> to multilingual mode (`multi`); pass `--ocr-lang english` for the English-only
> v2 selector. Remote OCR NIM endpoints decide their own model and language
> behavior, and the local OCR selectors are not added to remote request payloads.

When you use a remote embedder, the endpoint and provider prefix remain runtime
configuration. The query model is read from LanceDB metadata when available;
pass an explicit model only for an override or a legacy table without metadata.

### Inspect extracts
You can inspect how recall accuracy optimized text chunks for various content types were extracted into text representations:
```text
# page 1 raw text:
>>> chunks.iloc[0]["text"]
'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose...'

# markdown formatted table from the first page
'| Table | 1 |\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |'

# a chart from the first page
>>> chunks.iloc[2]["text"]
'Chart 1\nThis chart shows some gadgets, and some very fictitious costs.\nGadgets and their cost\n$160.00\n$140.00\n$120.00\n$100.00\nDollars\n$80.00\n$60.00\n$40.00\n$20.00\n$-\nPowerdrill\nBluetooth speaker\nMinifridge\nPremium desk fan\nHammer\nCost'

# markdown formatting for full pages or documents:
# per-page markdown is keyed by page number
>>> to_markdown_by_page(chunks).keys()
dict_keys([1, 2, 3])

>>> to_markdown_by_page(chunks)[1]
'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose. It contains some \r\nheadings and some placeholder text to fill the space. The text is not important and contains \r\nno real value, but it is useful for testing. Below, we will have some simple tables and charts \r\nthat we can use to confirm Ingest is working as expected.\r\nTable 1\r\nThis table describes some animals, and some activities they might be doing in specific \r\nlocations.\r\nAnimal Activity Place\r\nGira@e Driving a car At the beach\r\nLion Putting on sunscreen At the park\r\nCat Jumping onto a laptop In a home o@ice\r\nDog Chasing a squirrel In the front yard\r\nChart 1\r\nThis chart shows some gadgets, and some very fictitious costs.\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost\n\n### Table 1\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\n### Chart 1\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost\n\n### Table 2\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\n### Chart 2\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost\n\n### Table 3\n\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |\n\n### Chart 3\n\nChart 1 This chart shows some gadgets, and some very fictitious costs. Gadgets and their cost $160.00 $140.00 $120.00 $100.00 Dollars $80.00 $60.00 $40.00 $20.00 $- Powerdrill Bluetooth speaker Minifridge Premium desk fan Hammer Cost'

# full document markdown is a single string (or None if empty)
>>> to_markdown(chunks)[:50]
'# Extracted Content\n\n## Page 1\n\nTestingDocument\r\nA s'
```

After ingest, those chunks are in the default LanceDB table `nemo-retriever`.
You can query them to retrieve semantically relevant chunks for an LLM:

### Run a recall query

```python
from nemo_retriever.graph.retriever import Retriever

retriever = Retriever(
  vdb_kwargs={"uri": "lancedb", "table_name": "nemo-retriever"},
  top_k=5,
  rerank=False
)

query = "Given their activities, which animal is responsible for the typos in my documents?"

# you can also submit a list with retriever.queries([...])
hits = retriever.query(query)
```

Default `Retriever()` also reads `lancedb/nemo-retriever`. Pass `vdb_kwargs` only
when you wrote a different URI or table name.

If you ingested with the remote-NIM recipe above (no local GPU), point the
`Retriever` at the same embedding endpoint so query vectors are produced by the
same model that produced the stored chunk vectors:

```python
retriever = Retriever(
    vdb_kwargs={"uri": "lancedb", "table_name": "nemo-retriever"},
    embed_kwargs={
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embedding_endpoint": "https://integrate.api.nvidia.com/v1/embeddings",
    },
    top_k=5,
    rerank=False,
)
hits = retriever.query(query)
```

```text
# retrieved text from the first page
>>> hits[0]
{'text': 'TestingDocument\r\nA sample document with headings and placeholder text\r\nIntroduction\r\nThis is a placeholder document that can be used for any purpose. It contains some \r\nheadings and some placeholder text to fill the space. The text is not important and contains \r\nno real value, but it is useful for testing. Below, we will have some simple tables and charts \r\nthat we can use to confirm Ingest is working as expected.\r\nTable 1\r\nThis table describes some animals, and some activities they might be doing in specific \r\nlocations.\r\nAnimal Activity Place\r\nGira@e Driving a car At the beach\r\nLion Putting on sunscreen At the park\r\nCat Jumping onto a laptop In a home o@ice\r\nDog Chasing a squirrel In the front yard\r\nChart 1\r\nThis chart shows some gadgets, and some very fictitious costs.', 'metadata': '{"page_number": 1, "pdf_page": "multimodal_test_1", "page_elements_v3_num_detections": 9, "page_elements_v3_counts_by_label": {"table": 1, "chart": 1, "title": 3, "text": 4}, "ocr_table_detections": 1, "ocr_chart_detections": 1, "ocr_infographic_detections": 0}', 'source': '{"source_id": "/home/dev/projects/NeMo-Retriever/data/multimodal_test.pdf"}', 'page_number': 1, '_distance': 1.5822279453277588}

# retrieved text of the table from the first page
>>> hits[1]
{'text': '| Table | 1 |\n| This | table | describes | some | animals, | and | some | activities | they | might | be | doing | in | specific |\n| locations. |\n| Animal | Activity | Place |\n| Giraffe | Driving | a | car | At | the | beach |\n| Lion | Putting | on | sunscreen | At | the | park |\n| Cat | Jumping | onto | a | laptop | In | a | home | office |\n| Dog | Chasing | a | squirrel | In | the | front | yard |\n| Chart | 1 |', 'metadata': '{"page_number": 1, "pdf_page": "multimodal_test_1", "page_elements_v3_num_detections": 9, "page_elements_v3_counts_by_label": {"table": 1, "chart": 1, "title": 3, "text": 4}, "ocr_table_detections": 1, "ocr_chart_detections": 1, "ocr_infographic_detections": 0}', 'source': '{"source_id": "/home/dev/projects/NeMo-Retriever/data/multimodal_test.pdf"}', 'page_number': 1, '_distance': 1.614684820175171}
```

###  Generate a query answer using an LLM
The above retrieval results are often feedable directly to an LLM for answer generation.

To do so, first install the openai client and set your [build.nvidia.com](https://build.nvidia.com/) API key:
```bash
uv pip install openai
export NVIDIA_API_KEY=nvapi-...
```

```python
from openai import OpenAI
import os

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.environ.get("NVIDIA_API_KEY")
)

hit_texts = [hit["text"] for hit in hits]
prompt = f"""
Given the following retrieved documents, answer the question: {query}

Documents:
{hit_texts}
"""

completion = client.chat.completions.create(
  model="nvidia/nemotron-3-super-120b-a12b",
  messages=[{"role":"user","content":prompt}],
  stream=False
)

answer = completion.choices[0].message.content
print(answer)
```

Answer:
```
Cat is the animal whose activity (jumping onto a laptop) matches the location of the typos, so the cat is responsible for the typos in the documents.
```

### Run agentic retrieval

Agentic retrieval runs an LLM-driven ReAct loop over an existing LanceDB index.
It does not ingest documents. Build the index with one of the ingestion flows
above, then query the same `lancedb_uri`, `table_name`, and embedding model.
The examples below use the default table `nemo-retriever`. When you omit
`--embed-model-name`, agentic retrieval uses the selected table's model.

By default, the agent LLM runs in process with local vLLM and `nemotron-8b`
(`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`). This requires a CUDA GPU host and the
`[local]` extra. GPU placement follows process-level vLLM behavior, so set
`CUDA_VISIBLE_DEVICES` before starting the command. Embedding follows the same
local/remote split as dense retrieval; pass `--embed-invoke-url` to use a remote
embedding endpoint.

**Local in-process vLLM agent LLM.** Omit `--agentic-invoke-url` to load the
supported local agent LLM directly in the Python process. `nemotron-8b` is the
default; `super-49b` is also supported when the process has enough visible GPUs.
For `super-49b`, set `--agentic-local-tensor-parallel-size 2` with two visible
GPUs (for example `CUDA_VISIBLE_DEVICES=0,1`).

```bash
CUDA_VISIBLE_DEVICES=0 retriever query "Given their activities, which animal is responsible for the typos in my documents?" \
  --agentic \
  --lancedb-uri lancedb \
  --table-name nemo-retriever
```

When the first ``tensor_parallel_size`` CUDA-visible GPUs are not
NVLink-connected (typical dual-GPU PCIe workstations), tensor-parallel
startup automatically sets `NCCL_NVLS_ENABLE=0` and
`TORCH_SYMM_MEM_DISABLE_MULTICAST=1`, since NVLink multicast collectives abort
vLLM startup with a NCCL CUDA error there. Set either variable yourself to
override the fallback. Detection is scoped to that TP device group, not the
whole host or extra visible GPUs outside the shard.

**OpenAI-compatible agent endpoint.** Pass `--agentic-invoke-url` when you want a
custom model or a separately hosted chat-completions server, such as vLLM server
mode or a self-hosted NIM. When an invoke URL is provided, `--agentic-llm-model`
is required and is sent as the remote model ID.

```bash
retriever query "What is RAG?" \
  --agentic \
  --agentic-llm-model nvidia/nemotron-3.5-lightning-30b-a3b \
  --agentic-invoke-url http://localhost:9000/v1/chat/completions \
  --lancedb-uri lancedb \
  --table-name nemo-retriever
```

The Helm `answer_llm` Nemotron 3.5 Lightning NIM is not tool-call ready by default.
Add `--enable-auto-tool-choice --tool-call-parser qwen3_coder` to
`NIM_PASSTHROUGH_ARGS`, retaining `--reasoning-parser nemotron_v3`, before
you point `--agentic-invoke-url` at that endpoint. Refer to
[Agentic retrieval (self-hosted Nemotron 3.5 Lightning)](helm/README.md#agentic-retrieval-llm)
in the Helm chart README.

Agentic CLI output is not the five-field dense projection (`modality`,
`page_number`, `score`, `source`, and `text`). Each JSON object is the
internal hit dictionary plus `doc_id`, `rank`, and `result_source`.
`result_source` is `final_results`, `rrf`, or `selection_agent`. When no
retrieval hop returned the document, only those three keys are present.

For a quick smoke test, reduce agent work:

```bash
CUDA_VISIBLE_DEVICES=0 retriever query "What is RAG?" \
  --agentic \
  --lancedb-uri lancedb \
  --table-name nemo-retriever \
  --top-k 1 \
  --agentic-react-max-steps 1
```

You can run the same flow from Python. Omit `invoke_url` for the default local
in-process vLLM backend, or pass `invoke_url` on `QueryAgenticOptions` for a
separate OpenAI-compatible chat-completions endpoint. Omit `embed_model_name`
so the query reuses the table's recorded model.

```python
from nemo_retriever.cli.query_workflow import agentic_query_documents
from nemo_retriever.query.options import (
    QueryAgenticOptions,
    QueryRequest,
    QueryRetrievalOptions,
    QueryStorageOptions,
)

# Local in-process vLLM: set CUDA_VISIBLE_DEVICES before starting Python.
# Remote agent LLM: pass invoke_url="http://localhost:9000/v1/chat/completions".
results = agentic_query_documents(
    QueryRequest(
        query="What is RAG?",
        retrieval=QueryRetrievalOptions(top_k=10),
        storage=QueryStorageOptions(
            lancedb_uri="lancedb",
            table_name="nemo-retriever",
        ),
        agentic=QueryAgenticOptions(
            enabled=True,
            llm_model="nemotron-8b",
            local_llm_backend="vllm",
            local_gpu_memory_utilization=0.8,
            local_tensor_parallel_size=1,
        ),
    )
)
```

For CLI flags and operator details, refer to [Agentic retrieval](docs/cli/README.md#agentic-retrieval) in the CLI reference.

### Live RAG SDK (retrieve + answer in one call)

The pattern above -- retrieve hits, build a prompt, call an LLM -- is baked into the SDK as `Retriever.answer()` so live applications can skip the boilerplate. The same `Retriever` instance powers three entry points:

| Method | Input | Output | Use case |
| --- | --- | --- | --- |
| `Retriever.retrieve(query, top_k=...)` | one query | `RetrievalResult` (`chunks`, `metadata`) | Structured retrieval without an LLM. |
| `Retriever.answer(query, llm=..., judge=None, reference=None, ...)` | one query | `AnswerResult` (answer + chunks + optional scores) | One-shot RAG -- production/live. |
| `Retriever.pipeline().generate(...).judge(...).score().run(queries)` | many queries | `pandas.DataFrame` | Batch RAG over the operator graph, each step optional. |

Install the LLM client extra:
```bash
uv pip install "nemo-retriever[llm]"
export NVIDIA_API_KEY=nvapi-...
```

The default Live RAG model uses LiteLLM's `nvidia_nim` provider. LiteLLM does not
read `NVIDIA_API_KEY` for that provider. Pass `api_key="os.environ/NVIDIA_API_KEY"`
so the same key is forwarded on each request.

Single-query live RAG. Point `vdb_kwargs` at the table you ingested. The default
table is `nemo-retriever` for both Python `.vdb_upload()` and `retriever ingest`.
The embedding model in `embed_kwargs` must match the model used during ingestion.

```python
from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.models.llm import LiteLLMClient

retriever = Retriever(
    vdb_kwargs={"uri": "lancedb", "table_name": "nemo-retriever"},
    embed_kwargs={
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embedding_endpoint": "https://integrate.api.nvidia.com/v1/embeddings",
    },
    top_k=5,
)
llm = LiteLLMClient.from_kwargs(
    model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key="os.environ/NVIDIA_API_KEY",
    temperature=0.0,
    max_tokens=512,
)

result = retriever.answer("What is RAG?", llm=llm)
print(result.answer)
# 'Retrieval-augmented generation combines external context with an LLM...'
print(len(result.chunks), "chunks from", {m.get("source") for m in result.metadata})
print(f"{result.latency_s:.2f}s on {result.model}")
```

Local-GPU shortcut: if you ingested with default `retriever ingest` flags
(`[local]` extra installed), omit `embed_kwargs`. The `Retriever` reads the
logical `nvidia/nemotron-3-embed-1b` model from the LanceDB table metadata and
resolves it to a checkpoint that supports the local GPU.

Live RAG with scoring and an LLM judge (requires a ground-truth `reference`):
```python
from nemo_retriever.models.llm import LLMJudge

judge = LLMJudge.from_kwargs(
    model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key="os.environ/NVIDIA_API_KEY",
    temperature=0.1,
    max_tokens=4096,
)
result = retriever.answer(
    "What is RAG?",
    llm=llm,
    judge=judge,
    reference="RAG combines retrieved context with LLM generation.",
)
print(result.token_f1, result.judge_score, result.failure_mode)
# 0.62 4 'correct'
```

Batch RAG over the operator graph -- each builder step is optional:
```python
df = (
    retriever.pipeline()
    .generate(llm)
    .judge(judge)
    .score()
    .run(
        queries=["What is RAG?", "What is reranking?"],
        reference=["RAG combines retrieval with generation.", "Reranking re-scores retrieved passages."],
    )
)
print(df[["query", "answer", "token_f1", "judge_score", "failure_mode"]])
```

The pipeline builder runs steps in call order. `.score()` writes `failure_mode` from the `judge_score` present at that step. Call `.judge()` before `.score()` when you want `failure_mode` to reflect the judge result. If `.score()` runs first, a missing `judge_score` is classified as `judge_error`, and a later `.judge()` does not recompute `failure_mode`.

Scoring tiers on `AnswerResult`:

- **Tier 1** (`answer_in_context`) -- whether retrieval surfaced the evidence; requires `reference`.
- **Tier 2** (`token_f1`, `exact_match`) -- token-level overlap; requires `reference`.
- **Tier 3** (`judge_score`) -- dual-judge `AnswerAccuracy` LLM-as-judge score (0.0-1.0), ported from ragas onto `litellm`; requires `reference` and `judge`. `judge_reasoning` is always empty (the metric emits only a rating).
- `failure_mode` -- derived classification (`correct`, `partial`, `retrieval_miss`, `generation_miss`, `generation_error`, `refused_*`, `thinking_truncated`, `judge_error`).

If only `reference` is supplied, Tier 1 + 2 run. If only `judge` is supplied (without `reference`), a `ValueError` is raised. On generation error, scoring and judge are skipped and `AnswerResult.error` is populated.

### Ingest other types of content:

For PowerPoint and Docx files, ensure libeoffice is installed by your system's package manager. This is required to make their pages renderable as images for our [page-elements content classifier](https://huggingface.co/nvidia/nemotron-page-elements-v3).

For example, with apt-get on Ubuntu:
```bash
sudo apt install -y libreoffice
```

For SVG files, install the optional `cairosvg` dependency. SVG support is available in the NeMo Retriever Library, but not in the container deployment. `cairosvg` requires network access to install, so it will not work in air-gapped environments.
```bash
uv pip install "nemo-retriever[multimedia]"
# or to install only the SVG dependency:
uv pip install "cairosvg>=2.7.0"
```

Example usage:
```python
# docx and pptx files
documents = [str(Path(f"../data/*{ext}")) for ext in [".pptx", ".docx"]]
# mixed types of images
images = [str(Path(f"../data/*{ext}")) for ext in [".png", ".jpeg", ".bmp"]]
ingestor = (
  # above file types can be combined into a single job
  ingestor.files(documents + images)
  .extract()
)
```

*Note:* the `split_config` keyword on `.extract()` uses a tokenizer to split texts by a max_token length
### Render results as markdown

If you want a readable markdown view of extracted results, pass a single document's extraction
records to `nemo_retriever.common.io.to_markdown`. The helper returns one markdown string (or `None`
if there is no content), with per-page sections joined under a single document heading.

For multi-document runs, filter the results by `path` and pass one document at a time.
To build a filename-keyed index across many documents, use `build_page_index`.

PDF text is split at the page level.

HTML and .txt files have no natural page delimiters, so they almost always need to be paired with the `split_config` keyword.

```python
# html and text files - include split_config to prevent texts from exceeding the embedder's max sequence length
documents = [str(Path(f"../data/*{ext}")) for ext in [".txt", ".html"]]
ingestor = (
  ingestor.files(documents)
  .extract(split_config={"text": {"max_tokens": 5}, "html": {"max_tokens": 5}}) # 1024 by default, set low here to demonstrate chunking
)
results = ingestor.ingest()
document_path = results["path"].iloc[0]
document_results = results[results["path"] == document_path]
markdown_doc = to_markdown(document_results)
print(markdown_doc)
```

Use `to_markdown_by_page(document_results)` when you want a `dict[int, str]` keyed by page
number instead, where each value is the rendered markdown for that page.
For audio and video files, ensure ffmpeg is installed by your system's package manager.

For example, with apt-get on Ubuntu:
```bash
sudo apt install -y ffmpeg
```

The bundled Docker image uses the FFmpeg package provided by the base Ubuntu
image when `INSTALL_FFMPEG=true` is set. If your workflow depends on exact
FFmpeg codec or version behavior, verify the image package against those
requirements.

The bundled Dockerfile skips ffmpeg/ffprobe by default. For the service image,
set `INSTALL_FFMPEG=true` at runtime to install them during container startup:

```bash
docker run -e INSTALL_FFMPEG=true nemo-retriever-service
```

For Kubernetes deployments, set `service.installFfmpeg=true` in the Helm chart.
This runtime install requires network access to package repositories, a
writable root filesystem, and security policy that allows the image's scoped
sudo use. For locked-down environments that cannot install packages at startup,
use a custom service image that already contains ffmpeg/ffprobe.

```python
ingestor = create_ingestor(run_mode="batch")
ingestor = ingestor.files([str(INPUT_AUDIO)]).extract_audio()
```

### Store row images

Use `.store()` after `.embed()` to persist row-level image payloads to local disk or object storage (S3, MinIO, GCS via fsspec). Stored URIs are written back to the DataFrame for VDB upload and reranking.

```python
ingestor = (
  ingestor.files(documents)
  .extract()
  .embed()
  .store(
    storage_uri="s3://my-bucket/citation-assets",  # or a local path
    storage_options={"key": "...", "secret": "..."},  # fsspec auth for S3/MinIO
  )
  .vdb_upload()
)
```

### Explore Different Pipeline Options:

You can use the [Nemotron RAG VL Embedder](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2)

```python
ingestor = (
  ingestor.files(documents)
  .extract()
  .embed(
    model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
    #works with plain "text"s, "image"s, and "text_image" pairs
    embed_modality="text_image"
  )
)
```

You can use a different ingestion pipeline based on [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse) hosted on NVIDIA Build and combined with the default embedder:

```python
ingestor = ingestor.files(documents).extract(
  method="nemotron_parse",
  nemotron_parse_invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
  nemotron_parse_model="nvidia/nemotron-parse",
)
```

## Run with remote inference, no local GPU required:

For build.nvidia.com hosted inference, set [`NVIDIA_API_KEY`](https://nvidia.github.io/NeMo-Retriever/extraction/api-keys/#nvidia-api-key) as an environment variable (refer to [Authentication and API keys](https://nvidia.github.io/NeMo-Retriever/extraction/api-keys/)).

```python
ingestor = (
  ingestor.files(documents)
  .extract(
    # for self hosted NIMs, your URLs will depend on your NIM container DNS settings
    page_elements_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3",
    ocr_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2",
    table_structure_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1",
  )
  .embed(
    embed_invoke_url="https://integrate.api.nvidia.com/v1/embeddings",
    model_name="nvidia/llama-nemotron-embed-1b-v2",
    embed_modality="text",
  )
  .vdb_upload()
)
```

## Ray cluster setup

NeMo Retriever Library uses Ray Data for distributed ingestion and benchmarking. [NeMo Ray run guide](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)

### Local Ray cluster with dashboard

To start a Ray cluster with the dashboard on a single machine use the following command.

```bash
ray start --head
```

Open `http://127.0.0.1:8265` in your browser for the Ray Dashboard, and run your NeMo Retriever Library pipeline on the same machine with `--ray-address auto` to attach to this cluster. [Connecting to a remote Ray cluster on Kubernetes](https://discuss.ray.io/t/connecting-to-remote-ray-cluster-on-k8s/7460)

### Single‑GPU cluster on multi‑GPU nodes

To restrict Ray to a single GPU on a multi‑GPU node use the following command.

```bash
CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1
```
Then run your pipeline as before with `--ray-address auto` so it connects to this single‑GPU Ray cluster. [NeMo Ray run guide](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)

## Multi-GPU resource heuristics for library batch mode

In batch mode, NeMo Retriever Library sizes unspecified Ray actor pools from Ray CPU and GPU resources. The library uses the resources that Ray reports as available immediately before it submits the pipeline.

The default sizing order is:

1. Detect CPU and GPU counts from the Ray cluster after the batch runtime starts (`cluster_resources` / `available_resources`).
2. Scale unspecified worker pools from the available GPU count using built-in per-stage heuristic constants.
3. Apply explicit `BatchTuningParams` values and `node_overrides`. These take precedence over the heuristic defaults.

The library does not read environment variables to set worker counts or CPU and GPU totals. `CUDA_VISIBLE_DEVICES` still controls which GPUs Ray can see. To limit GPU count, start a Ray cluster with a restricted GPU set as shown in the previous section.

### Override worker counts

Set explicit worker counts when you need a fixed allocation. Unspecified fields still use the heuristic defaults.

The following table lists the primary batch-mode worker controls:

| CLI Flag | BatchTuningParams Field | Meaning |
|---|---|---|
| `--pdf-extract-workers` | `pdf_extract_workers` | Maximum Ray tasks for PDF extraction |
| `--page-elements-workers` | `page_elements_workers` | Ray actors for page-element detection |
| `--ocr-workers` | `ocr_workers` | Ray actors for OCR inference |
| `--embed-workers` | `embed_workers` | Ray actors for embedding |

Related batch-size, CPU, and GPU-per-actor flags are documented in the [CLI ingest options](docs/cli/README.md).

For the CLI, pass the batch-mode flags on `retriever ingest batch`. Replace
`/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest batch /path/to/your/pdfs \
  --pdf-extract-workers 4 \
  --page-elements-workers 3 \
  --ocr-workers 3 \
  --embed-workers 2
```

For the Python API, pass `BatchTuningParams` on `.extract()` and `.embed()`:

```python
from pathlib import Path

from nemo_retriever import create_ingestor
from nemo_retriever.common.params import BatchTuningParams

documents = [str(Path("../data/multimodal_test.pdf"))]

chunks = (
  create_ingestor(run_mode="batch")
  .files(documents)
  .extract(
    batch_tuning=BatchTuningParams(
      pdf_extract_workers=4,
      page_elements_workers=3,
      ocr_workers=3,
    )
  )
  .embed(
    batch_tuning=BatchTuningParams(
      embed_workers=2,
    )
  )
  .ingest()
)
```

If explicit worker counts or `node_overrides` exceed the available Ray CPU or GPU budget, the library raises an error before it submits work. Reduce `*_workers` or per-node concurrency, or wait for shared-cluster capacity.

For source-task CPU reservations and custom Ray Data graphs, refer to the [performance guide](https://docs.nvidia.com/nemo/retriever/latest/extraction/performance_guide/).

## NIM containers

For deployment of NeMo Retriever / **NIM** containers, use **Helm**:
**[`helm/README.md`](helm/README.md)** and the **NeMo Retriever Library**
documentation linked from that guide and the
[NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/).

## Troubleshooting

### vLLM engine fails to start during CUDA graph capture

When using the vLLM-based VL reranker, the engine may fail to start with errors similar to the following:

```
fatal error: Python.h: No such file or directory
...
torch._inductor.exc.InductorError: CalledProcessError: Command '['/usr/bin/gcc', '...cuda_utils.c', ...]' returned non-zero exit status 1.
...
RuntimeError: Engine core initialization failed.
```

This occurs because Triton compiles a small C extension at runtime during CUDA graph capture and requires the Python development headers. If `Python.h` is not installed, the compilation fails and the vLLM engine cannot start.

To resolve this, install the Python development headers for your Python version:

```bash
# For Python 3.12 on Ubuntu/Debian
sudo apt install python3.12-dev
```

After installing the headers, restart the pipeline.

## Benchmarking

End-to-end Retriever experiments and benchmark orchestration are maintained in
the [NeMo Retriever Benchmark (NRB) repository](https://gitlab-master.nvidia.com/charlesb/nemo-retriever-benchmark/).
This repository continues to provide the library, CLI workflows, service
implementation, and Helm chart that NRB benchmarks.

### Ingest image storage

Use root ingest to persist extracted image assets to local storage or any
fsspec-compatible URI:

```bash
retriever ingest ./data \
  --store-images-uri ./processed_docs/images
```

The store stage writes the image payloads produced by ingest. With
`--embed-granularity page`, stored assets are page images. With
`--embed-granularity element`, stored assets are element images.
