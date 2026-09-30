# QA Evaluation Pipeline

The evaluation framework lives in **`nemo_retriever.evaluation`** (install `nemo-retriever[llm]` from PyPI, or **`uv pip install -e "./nemo_retriever[llm]"` from this repo root** to exercise local changes).

Measures LLM answer quality over a RAG pipeline: retrieve context from a VDB, generate answers with one or more LLMs, and score each answer against ground-truth references using multi-tier scoring and an LLM-as-judge.

**Pluggable retrieval:** The evaluation framework does not care how you retrieved chunks -- only that you produce a JSON file that matches the **[retrieval JSON specification](#retrieval-json-format-interface-contract)** expected by `retriever eval run` / `FileRetriever`. Vector search, hybrid, agentic pipelines, or any custom system can plug in as long as the file format and query strings align with your chosen ground-truth dataset. Alternatively, set `retrieval.type: "lancedb"` to **[query LanceDB live in-memory](#in-memory-lancedb-retrieval)** -- skipping the export step entirely while optionally saving the JSON for future re-runs.

**Default ground truth:** Standalone runs default to **`data/bo767_annotations.csv`** at the repo root -- the **bo767 annotations subset** maintained for this benchmark (multi-modality Q&A over the bo767 PDFs). Override with `QA_DATASET` or another registered loader when comparing different corpora.

Designed to be **plug-and-play** -- swap retrievers, generators, or judges independently via Python Protocols without touching the orchestrator.

## Table of Contents

- [Pipeline File Map and Data Flow](#pipeline-file-map-and-data-flow)
- [Reproducing the bo767 Run](#reproducing-the-bo767-run)
  - [Which Path Should I Use?](#which-path-should-i-use)
- [Retrieval JSON Format (Interface Contract)](#retrieval-json-format-interface-contract)
- [Custom Datasets (CSV Loader)](#custom-datasets-csv-loader)
- [Architecture](#architecture)
  - [Operator Graph Chain (single-model)](#operator-graph-chain-single-model)
  - [QAEvalPipeline (multi-model sweeps)](#qaevalpipeline-multi-model-sweeps)
  - [When to Use Which](#when-to-use-which)
  - [Protocol Interfaces](#protocol-interfaces)
  - [Files](#files-nemo_retrieverevaluation)
  - [Graph Framework Integration](#graph-framework-integration)
  - [Entry Points](#entry-points)
- [Configuration](#configuration)
  - [Eval Config File (YAML / JSON)](#eval-config-file-yaml--json)
  - [In-Memory LanceDB Retrieval](#in-memory-lancedb-retrieval)
- [Scoring System (Three-Tier Hierarchy)](#scoring-system-three-tier-hierarchy)
- [Adding a New Component](#adding-a-new-component)
- [Output Format](#output-format)
- [Dataset Limitations](#dataset-limitations)

## Pipeline File Map and Data Flow

End-to-end bo767 + LanceDB + full-page markdown touches these **artifacts** and **library code**:

| Stage | Artifacts produced | Code / APIs involved |
|-------|-------------------|----------------------|
| **1. Ingest + embed** | `lancedb/<uri>/<table>/` (embedded sub-page chunks) | `retriever ingest` extracts, embeds, and writes LanceDB through the supported ingest workflow. **Table name must match** `retriever eval export` (`--lancedb-table`, default `nemo-retriever`). |
| **2. Optional full-page markdown index** | `data/bo767_page_markdown.json` (`source_id` -> page -> markdown) | If you already have extraction Parquet, `retriever eval build-page-index` calls `nemo_retriever.io.markdown.build_page_index()` to create a page index. |
| **3. Retrieval export** | `data/eval/bo767_retrieval_fullpage.json` (or sub-page JSON) | `retriever eval export` -> `nemo_retriever.export.export_retrieval_json()` queries LanceDB; if `--page-index` is provided, hits are expanded/deduped by `(source_id, page)` and replaced with full-page markdown strings. |
| **4. Ground truth** | `data/bo767_annotations.csv` (repo root) | Questions/answers for export and eval; must align with **query string normalization** in `FileRetriever` (see retrieval JSON rules). |
| **5. Evaluation** | `qa_results_*.json` | `retriever eval run` or operator graph chain -> `nemo_retriever.evaluation`: `RetrievalLoaderOperator >> QAGenerationOperator >> JudgingOperator >> ScoringOperator`, or `QAEvalPipeline` for multi-model sweeps. |

**Data flow (conceptual):** `retriever ingest` writes chunk embeddings to
LanceDB. **Export** queries that table and writes retrieval JSON. If a separate
workflow supplies extraction Parquet, an optional page index can replace hit
chunks with full-page markdown during export. In **file mode**, evaluation reads
the retrieval JSON and ground-truth CSV. In **[lancedb mode](#in-memory-lancedb-retrieval)**,
evaluation queries LanceDB directly (optionally saving JSON for later re-runs).

```
+--------------------+    +---------+    +----------------+    +------------------+
| retriever ingest   | -> | LanceDB | -> | eval export    | -> | RetrievalLoader  |
| --lancedb-uri ...  |    | table   |    | retrieval JSON |    | >> Generation    |
| --table-name ...   |    +---------+    +----------------+    | >> Judging       |
+--------------------+          ^                              | >> Scoring       |
                                | optional page index          +------------------+
                   extraction Parquet -> build-page-index

 Bring Your Own Retrieval    +---------+--------+
 ─────────────────────────   | Any pipeline that |
 Skip steps 1-3 entirely.   | outputs retrieval |
 Produce a JSON matching     | JSON (see spec)  |
 the interface contract. --> +---------+--------+
```

Steps 1-3 are one reference implementation (NeMo Retriever + LanceDB).
Any retrieval system that produces a conforming JSON can replace them.
Steps 4-5 can be re-run with different LLM configs without repeating retrieval.

**Bring your own retrieval:** Skip steps 1-3 entirely if you already have a retrieval system. Produce a JSON conforming to the [interface contract](#retrieval-json-format-interface-contract) and proceed directly to step 4.

## Reproducing the bo767 Run

The default supported path evaluates sub-page chunks from a LanceDB table. A
full-page markdown run additionally requires a separately produced extraction
Parquet dataset for `retriever eval build-page-index`.

**Working directory:** All commands below run from the **repo root** unless otherwise noted.

**Debug:** Lance index build can hit `No space left on device` when `/tmp` is a tiny tmpfs; set `export TMPDIR=/path/to/large/filesystem/tmp` and `mkdir -p "$TMPDIR"` before step 1, then re-run `retriever ingest`.

**Before running any quick reference below**, complete the one-time setup:

```bash
# 1. Create and activate a Python 3.12 environment
uv venv qa-retriever --python 3.12
source qa-retriever/bin/activate

# 2. Install nemo_retriever with LLM extras (from repo root)
cd /path/to/nemo-retriever
uv pip install -e "./nemo_retriever[llm]"

# 3. Set your API key (used by generation + judging)
export NVIDIA_API_KEY="nvapi-..."
```

See [Python environment](#python-environment) and [Prerequisites](#prerequisites-data-and-keys) for details.

<details>
<summary><strong>Quick reference -- sub-page chunks (skip full-page markdown)</strong></summary>

If you do not need full-page markdown context and want the simplest path,
skip the Parquet/index steps entirely. This uses raw sub-page chunks as
retrieval context, which may produce lower scores for structured content
(tables, charts, infographics).

```bash
cd /path/to/nemo-retriever

# 1. Ingest + embed into LanceDB
retriever ingest /path/to/bo767 \
  --lancedb-uri lancedb \
  --table-name nemo-retriever

# 2. Export retrieval (sub-page chunks, no page index)
retriever eval export \
  --lancedb-uri lancedb \
  --query-csv data/bo767_annotations.csv \
  --output data/eval/bo767_retrieval.json

# 3. Run QA evaluation
retriever eval run --config nemo_retriever/examples/eval_sweep.yaml
```

</details>

<details>
<summary><strong>Quick reference -- live LanceDB evaluation</strong></summary>

Alternative to the separable export+eval path above. Ingest first, then let the
evaluation command query LanceDB directly. Optionally save retrieval JSON so
you can re-run evaluation later without re-querying.

```bash
cd /path/to/nemo-retriever

retriever ingest /path/to/bo767 \
  --lancedb-uri lancedb \
  --table-name nemo-retriever

export LANCEDB_URI="lancedb"
export LANCEDB_TABLE="nemo-retriever"
export QA_DATASET="csv:data/bo767_annotations.csv"
export RETRIEVAL_SAVE_PATH="data/eval/bo767_retrieval.json"  # optional
retriever eval run --from-env
```

</details>

### Which path should I use?

| | **Option A: Separable (recommended)** | **Option B: Live LanceDB** |
|---|---|---|
| **Steps** | Ingest -> Export JSON -> Run eval | Ingest -> LanceDB query -> Eval |
| **Re-run eval with a different model?** | Instant -- just re-run step 4 with the same JSON | Must re-query LanceDB (or pass `save_path` to cache) |
| **Share results with teammates?** | Send the retrieval JSON file | Must share LanceDB or use `save_path` |
| **Best for** | Benchmarking, CI, reproducible comparisons | Quick end-to-end validation, development iteration |
| **Commands** | `retriever eval export` + `retriever eval run` | `LANCEDB_URI` with `retriever eval run --from-env` |

Option A is the primary design -- the retrieval JSON is the interface contract
that lets you re-run eval N times without re-querying, swap retrievers, and
share results. Option B is a convenience path for direct evaluation against an
existing LanceDB table.

### Bring your own retrieval (skip steps 1-3)

Steps 1-3 below are the **NeMo Retriever + LanceDB** reference implementation
for ingestion, extraction, indexing, and retrieval. If your team already has a
retrieval pipeline (agentic, hybrid, BM25, or any custom system), **skip
steps 1-3 entirely** and produce a retrieval JSON file that conforms to the
[Retrieval JSON Format (Interface Contract)](#retrieval-json-format-interface-contract).
Then proceed directly to [Step 4: Run QA evaluation](#step-4-run-qa-evaluation).

The only requirement is that your JSON contains a top-level `queries` object
mapping each ground-truth question string to `{ "chunks": ["...", ...] }`.
See the [interface contract](#retrieval-json-format-interface-contract) for the
full schema, required fields, and a worked example.

Any retrieval system that produces a conforming JSON can be evaluated -- see
the [interface contract](#retrieval-json-format-interface-contract) below.

### Python environment

Steps 1-3 (ingest, build index, export) require the **`nemo_retriever`** library with LanceDB, CUDA, and Ray support. Step 4 (QA eval) additionally requires **`litellm`**.

**Recommended setup:** create an isolated Python 3.12 virtual environment and install the **local** `nemo_retriever` checkout in editable mode:

```bash
uv venv qa-retriever --python 3.12
source qa-retriever/bin/activate
cd /path/to/nemo-retriever   # repo root
uv pip install -e "./nemo_retriever[llm]"
```

The `[llm]` extra installs `litellm` for LLM generation and judging (and powers both the batch-eval framework and the live-RAG SDK). If you are not using this tree, you can instead `uv pip install "nemo-retriever[llm]"` from PyPI (package name uses a hyphen).

**Eval-only path:** if you already have a retrieval JSON and only need to run `retriever eval run`, an environment with `nemo_retriever[llm]` installed is sufficient.

### Prerequisites (data and keys)

```bash
# bo767 PDFs (767 files)
ls /path/to/bo767/*.pdf | wc -l   # should be 767

# Ground truth: data/bo767_annotations.csv (1007 Q&A pairs across all modalities)
# Located at the repo root: <repo>/data/bo767_annotations.csv
```

### Step 1: Ingest and embed PDFs (NeMo Retriever)

`retriever ingest` resolves the manifest and executes the same graph ingest
implementation in local mode by default or with Ray Data in explicit `batch`
mode. It extracts, embeds, and uploads chunks to LanceDB in one workflow.

Run from the **repo root**. **Estimated time: ~45-90 min** (767 PDFs, GPU-accelerated
extraction + embedding).

```bash
retriever ingest /path/to/bo767 \
  --lancedb-uri lancedb \
  --table-name nemo-retriever
```

Output:
- `lancedb/nemo-retriever/` (~84k chunks) -- used by step 3 for retrieval queries.

Use `retriever ingest batch ...` for Ray Data scale-out.

### Step 2: Build page markdown index (NeMo Retriever)

> **Optional:** this step requires an extraction Parquet dataset produced by a
> separate workflow. Skip it to evaluate the sub-page chunks stored by
> `retriever ingest` directly.

Steps 2-4 are run from the repo root (or adjust paths accordingly).

Groups Parquet records by (document, page number) and renders each page via
`nemo_retriever.io.markdown.to_markdown_by_page()`. Outputs a JSON index
mapping `source_id -> page_number -> markdown`.
**Estimated time: ~5-10 min** (CPU-only, reads Parquet and renders markdown).

```bash
retriever eval build-page-index \
  --parquet-dir data/bo767_extracted \
  --output data/bo767_page_markdown.json
```

Output: `data/bo767_page_markdown.json` (~180 MB, ~6k pages across 767 docs).

### Step 3: Export retrieval results (NeMo Retriever)

> **Skip this step** if using the [live LanceDB path](#which-path-should-i-use),
> which queries LanceDB during evaluation instead of reading a JSON file.

Queries LanceDB for each ground-truth question via `nemo_retriever.export.export_retrieval_json()`,
then looks up the full-page markdown for each hit's page. Multiple sub-page hits
from the same page are deduplicated into a single full-page chunk.
**Estimated time: ~5-15 min** (1005 LanceDB queries + page index lookup).

```bash
retriever eval export \
  --lancedb-uri lancedb \
  --query-csv data/bo767_annotations.csv \
  --output data/eval/bo767_retrieval_fullpage.json \
  --page-index data/bo767_page_markdown.json
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--lancedb-uri` | `lancedb` | LanceDB directory |
| `--lancedb-table` | `nemo-retriever` | LanceDB table name |
| `--top-k` | `5` | Chunks per query |
| `--embedder` | `nvidia/llama-nemotron-embed-1b-v2` | Embedding model |
| `--query-csv` | _(required)_ | Ground-truth query/answer CSV |
| `--page-index` | _(unset)_ | Page markdown index JSON (enables full-page mode) |
| `--output` | _(required)_ | Output JSON path |

Output: `data/eval/bo767_retrieval_fullpage.json` (~50 MB, 1005 queries).

**Sub-page chunk mode:** omit `--page-index` to skip full-page
expansion. The export will use raw sub-page chunks directly from LanceDB, so no
Parquet or page index is needed.

### Step 4: Run QA evaluation

> **Alternative:** use the [live LanceDB path](#which-path-should-i-use) to
> query the table during `retriever eval run --from-env`.

**Estimated time: ~15 min - 45 min** (1005 queries, ~12s per query for generation + judge, 8 concurrent workers).

```bash
export NVIDIA_API_KEY="nvapi-..."
export RETRIEVAL_FILE=data/eval/bo767_retrieval_fullpage.json
export QA_DATASET="csv:data/bo767_annotations.csv"
export QA_MAX_WORKERS=8
retriever eval run --from-env
```

| Env Var | Default | Purpose |
|---------|---------|---------|
| `RETRIEVAL_FILE` | _(required)_ | Retrieval JSON from step 3 |
| `NVIDIA_API_KEY` | _(at least one key required)_ | Fallback API key for both generator and judge |
| `GEN_API_KEY` | falls back to `NVIDIA_API_KEY` | API key for the generator model |
| `JUDGE_API_KEY` | falls back to `NVIDIA_API_KEY` | API key for the judge model |
| `QA_DATASET` | _(required)_ | Ground-truth dataset (e.g. `csv:data/bo767_annotations.csv`) |
| `QA_TOP_K` | `5` | Chunks per query |
| `QA_MAX_WORKERS` | `4` | Concurrent API calls |
| `QA_LIMIT` | `0` (all) | Evaluate only first N queries |
| `RESULTS_DIR` | `data/eval` | Directory for auto-timestamped result JSONs |
| `GEN_MODEL` | `nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b` | Generator (single) |
| `GEN_MODEL_NAME` | `generator` | Short label for the generator |
| `GEN_API_BASE` | _(unset)_ | Override endpoint URL for the generator |
| `GEN_MODELS` | _(unset)_ | Multi-model sweep: `name:model,...` (overrides `GEN_MODEL`) |
| `GEN_TEMPERATURE` | `0.0` | Sampling temperature for generator |
| `JUDGE_MODEL` | `nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b` | Judge model |
| `JUDGE_API_BASE` | _(unset)_ | Override endpoint URL for the judge |
| `JUDGE_TEMPERATURE` | `0.1` | Sampling temperature for judge |
| `JUDGE_MAX_TOKENS` | `4096` | Max tokens for judge JSON output |
| `LITELLM_DEBUG` | `0` | Set `1` for full request/response logging |
| `MIN_COVERAGE` | `0.0` | Abort if retrieval covers fewer queries (0.0-1.0, e.g. `0.8`) |

**API key resolution:** `GEN_API_KEY` and `JUDGE_API_KEY` each fall back to `NVIDIA_API_KEY` in the `--from-env` CLI path. Set only `NVIDIA_API_KEY` if generator and judge share the same provider. Set individual keys when they use different providers.

For Python constructors and YAML model blocks, `api_key: null` (or an omitted `api_key`) means NeMo Retriever does not pass an explicit key; LiteLLM performs provider-native environment lookup on the worker, such as `OPENAI_API_KEY` for an OpenAI model. Use `api_key: "${VARIABLE_NAME}"` when configuration loading should require a specific variable, or `api_key="os.environ/VARIABLE_NAME"` in a persisted operator graph. For an unauthenticated local endpoint, pass `NO_API_KEY` explicitly; `None` does not mean “disable authentication.”

```python
from nemo_retriever.common.params import NO_API_KEY
from nemo_retriever.models.llm import LiteLLMClient

local_client = LiteLLMClient.from_kwargs(
    model="openai/local-model",
    api_base="http://localhost:8000/v1",
    api_key=NO_API_KEY,
)
```

**Auto-timestamped output:** Results are written to `{RESULTS_DIR}/qa_results_{dataset}_{generator}_{judge}_{YYYYMMDD_HHMMSS}.json` so consecutive runs never overwrite each other.

**Switching inference providers:** To use a different provider for the generator, set `GEN_MODEL`, `GEN_API_BASE`, and `GEN_API_KEY`:

```bash
export GEN_MODEL="openai/my-org/my-model"
export GEN_API_BASE="https://your-openai-compatible-endpoint/v1"
export GEN_API_KEY="your-api-key"
export NVIDIA_API_KEY="nvapi-..."   # still used by the judge (NIM)
retriever eval run --from-env
```

LiteLLM routes by model prefix (`nvidia_nim/`, `openai/`, `huggingface/`). See [Model Strings](#model-strings) for supported providers. Each model can target a different endpoint/key.

**API cost:** Each query costs ~$0.01-0.02 (generation + judge) on NIM pay-as-you-go. A full 1005-query run is approximately $10-20. Set `QA_LIMIT` to cap during development.

**Multi-model sweeps and repeated runs:** To evaluate multiple generator/judge combos or run each configuration N times for statistical confidence, use a YAML config file instead of env vars:

```bash
retriever eval run --config nemo_retriever/examples/eval_sweep.yaml
```

The config defines models once and composes evaluation combos with per-combo run counts. An example config is provided at `nemo_retriever/examples/eval_sweep.yaml`. See [Eval Config File](#eval-config-file-yaml--json) for the full schema.

### Results (March 2026 -- full-page markdown, bo767_annotations.csv)

This historical run predates the Nemotron 3.5 Lightning default. Its scores do not measure the current generator.

```
1005 queries evaluated (Nemotron Super 49B generator, Mixtral 8x22B judge)

Tier 1 - Retrieval Quality:
  Answer-in-Context rate:  88.2% (886/1005)

Tier 2 - Programmatic Answer Quality:
  generator            token_f1=0.120

Tier 3 - LLM Judge:
  generator            mean=3.74/5  scored=970  errors=35
                       dist: 1:251  2:44  3:28  4:26  5:621

Failure Breakdown:
  correct: 647  refused_missing_context: 95  refused_with_context: 37
  generation_miss: 89  partial: 65  retrieval_miss: 37
  thinking_truncated: 35
```

**Interpretation:** 88% of queries had the answer present in the retrieved
chunks (Tier 1). The generator answered correctly ~64% of the time. The gap
is primarily `refused_missing_context` (answer genuinely not in chunks) and
`refused_with_context` (model said "not found" when the answer was present),
plus `generation_miss` (model had the context but answered incorrectly).

### Sub-page chunk mode

To use raw sub-page chunks, skip step 2 and omit `--page-index` in step 3.
This produces smaller context windows and may
result in lower scores for queries that span structured content (tables, charts,
infographics). See the sub-page quick reference above.

## Retrieval JSON Format (Interface Contract)

The retrieval JSON is the **only interface** between your retrieval system and the
QA evaluation framework. Any retrieval method -- vector search, agentic retrieval, hybrid
pipelines, BM25, reranked, or a fully custom system -- can plug in by producing
a single JSON file that **matches this specification** (loaded via `FileRetriever`). The framework takes it from there: generates answers with one
or more LLMs and scores them with the judge. If your JSON does not match, the
eval script will not load or align queries correctly.

### Minimal format (all you need)

```json
{
  "queries": {
    "What is the range of the 767?": {
      "chunks": ["First retrieved chunk text...", "Second chunk text..."]
    },
    "How many engines does it have?": {
      "chunks": ["The 767 is powered by two..."]
    }
  }
}
```

Rules:
- **`"queries"`** (required): dict mapping query strings to result objects.
- **`"chunks"`** (required per query): list of plain-text strings, one per retrieved passage. Order matters -- put the best/most relevant chunk first. The framework uses the first `top_k` entries (default 5).
- **`"metadata"`** (optional per query): list of per-chunk provenance dicts (e.g. `{"source_id": "file.pdf", "page_number": 3}`). Carried through to the results JSON for traceability but not used for scoring.
- Top-level **`"metadata"`** (optional): free-form dict for your records (retrieval method, model, timing, etc.). Ignored by FileRetriever.
- Query matching is normalized (NFKC unicode, case-folded, whitespace-collapsed) so trivial formatting differences between the ground-truth CSV and the retrieval JSON don't cause misses.

### Full format example

```json
{
  "metadata": {
    "retrieval_method": "agentic_rag",
    "model": "nvidia/llama-nemotron-embed-1b-v2",
    "top_k": 5,
    "notes": "Used multi-step agent with query decomposition"
  },
  "queries": {
    "What percentage of infections occur without eyewear?": {
      "chunks": [
        "According to the infographic, 16% of infections...",
        "Protective eyewear reduces transmission by..."
      ],
      "metadata": [
        {"source_id": "1000360.pdf", "page_number": 3, "distance": 0.31},
        {"source_id": "1000360.pdf", "page_number": 3, "distance": 0.45}
      ]
    }
  }
}
```

### Using it with the CLI

```bash
# Point the eval at your retrieval JSON and run
export RETRIEVAL_FILE="path/to/my_retrieval_results.json"
export NVIDIA_API_KEY="nvapi-..."
retriever eval run --from-env
```

This means you can compare retrieval strategies head-to-head by running
the same eval against different retrieval JSONs -- the generator and judge
stay constant, so any score difference is purely from retrieval quality.

## Custom Datasets (CSV Loader)

Bring your own Q&A dataset without writing code. Any CSV with `query` and `answer` columns works:

```csv
query,answer,category
"What is the capital of France?","Paris","geography"
"What year was Python released?","1991","tech"
```

Point the eval at it with the `csv:` prefix:

```bash
export QA_DATASET="csv:/path/to/my_questions.csv"
export RETRIEVAL_FILE="path/to/my_retrieval.json"
retriever eval run --from-env
```

All columns beyond `query` and `answer` are preserved as metadata in the output. Rows with empty query or answer are silently skipped.

Built-in datasets: `bo767_infographic`, `vidore/<hf_dataset_id>`, `csv:/path/to/file.csv`. The default dataset is `csv:data/bo767_annotations.csv` (1007 Q&A pairs across text, table, chart, and infographic modalities).

## Architecture

The evaluation framework provides **two execution paths** that share the same
underlying components. Both paths consume an identical retrieval JSON + ground-truth
dataset and produce the same multi-tier scored output.

### Operator graph chain (single-model)

Built from `AbstractOperator` nodes connected via the `>>` operator. Each operator
inherits from `EvalOperator` (which extends `AbstractOperator` + `CPUOperator`),
accepts a pandas DataFrame, validates required columns, and returns an enriched
DataFrame. The chain is executed by `InprocessExecutor` (DFS traversal).

```
RetrievalLoaderOperator >> QAGenerationOperator >> JudgingOperator >> ScoringOperator
        |                         |                      |                  |
  Loads ground truth +     Generates answers       LLM-as-judge        Programmatic
  retrieval JSON into      via LiteLLMClient       scoring (0-1)       metrics (F1,
  a DataFrame with         (ThreadPool, N          via LLMJudge        exact match,
  query, answer,           concurrent workers)                         failure modes)
  context columns
```

Construct with `build_eval_chain(config)` from `nemo_retriever.evaluation.config`
or manually:

```python
from nemo_retriever.evaluation.retrieval_loader import RetrievalLoaderOperator
from nemo_retriever.evaluation.generation import QAGenerationOperator
from nemo_retriever.evaluation.judging import JudgingOperator
from nemo_retriever.evaluation.scoring_operator import ScoringOperator

graph = (
    RetrievalLoaderOperator(retrieval_json="retrieval.json", ground_truth_csv="gt.csv")
    >> QAGenerationOperator(model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")
    >> JudgingOperator(model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")
    >> ScoringOperator()
)
result_df = graph.execute(None)
```

### QAEvalPipeline (multi-model sweeps)

For evaluating multiple generators in a single run, `QAEvalPipeline` orchestrates
retrieval, generation, judging, and scoring across all configured models. It
accepts Protocol-based components (`RetrieverStrategy`, `LLMClient`, `AnswerJudge`)
and provides both a dict API (`evaluate()`) and a DataFrame API (`process()`).

```
nemo_retriever.evaluation.orchestrator (QAEvalPipeline)
    |
    |-- retriever  : RetrieverStrategy protocol
    |     |-- FileRetriever   (cached JSON -- recommended, in nemo_retriever.evaluation)
    |
    |-- llm_clients : dict[str, LLMClient protocol]
    |     |-- LiteLLMClient   (NVIDIA NIM, OpenAI, vLLM, Ollama)
    |
    |-- judge : AnswerJudge protocol
          |-- LLMJudge        (ragas AnswerAccuracy, 0-1 dual-judge)
```

Construct with `build_eval_pipeline(config)` from `nemo_retriever.evaluation.config`
or manually via `QAEvalPipeline(retriever=..., llm_clients=..., judge=...)`.

### When to use which

| Criterion | Operator graph chain | QAEvalPipeline |
|-----------|---------------------|----------------|
| Number of models | Single generator | Multiple generators (sweep) |
| Entry point | `build_eval_chain(config)` | `build_eval_pipeline(config)` |
| Data interchange | pandas DataFrame flowing through `>>` | `evaluate(qa_pairs)` returns dict, `process(df)` returns DataFrame |
| Extensibility | Add operators to the chain with `>>` | Swap Protocol implementations |
| Config-driven | YAML/JSON via `load_eval_config()` | YAML/JSON via `load_eval_config()` |

### Protocol interfaces

All three pluggable interfaces are Python `Protocol` classes defined in
`nemo_retriever.llm.types`. Any object that implements the right method
signature works -- no inheritance or registration required.

| Protocol | Method | Default implementation |
|----------|--------|----------------------|
| `RetrieverStrategy` | `retrieve(query, top_k) -> RetrievalResult` | `FileRetriever` (cached JSON) |
| `LLMClient` | `generate(query, chunks) -> GenerationResult` | `LiteLLMClient` (NIM, OpenAI, vLLM) |
| `AnswerJudge` | `judge(query, reference, candidate) -> JudgeResult` | `LLMJudge` (ragas AnswerAccuracy, 0-1) |

### Files (`nemo_retriever.evaluation`)

The evaluation framework lives in `nemo_retriever/src/nemo_retriever/evaluation/`.

| Module | Purpose |
|--------|---------|
| `types.py` | Protocol definitions (`RetrieverStrategy`, `LLMClient`, `AnswerJudge`) and dataclasses (`RetrievalResult`, `GenerationResult`, `JudgeResult`) |
| `eval_operator.py` | `EvalOperator` base class for all QA operators -- extends `AbstractOperator` + `CPUOperator`, adds `required_columns` / `output_columns` validation |
| `retrieval_loader.py` | `RetrievalLoaderOperator` -- entry point for graph chains; loads ground truth + retrieval JSON into a DataFrame |
| `generation.py` | `QAGenerationOperator` -- wraps `LiteLLMClient` for concurrent batch generation (ThreadPoolExecutor) |
| `judging.py` | `JudgingOperator` -- wraps `LLMJudge` for concurrent batch scoring |
| `scoring_operator.py` | `ScoringOperator` -- wraps `score_dataframe()` for programmatic metrics |
| `config.py` | Eval config loader: `load_eval_config()`, `build_eval_chain()`, `build_eval_pipeline()`; supports YAML/JSON with `${VAR}` expansion |
| `retrievers.py` | `FileRetriever` (cached JSON with normalized query matching) |
| `generators.py` | `LiteLLMClient` -- unified LLM client via litellm (NIM, OpenAI, vLLM, HF) |
| `judges.py` | `LLMJudge` -- 0-1 scoring via ragas `AnswerAccuracy` (dual-judge) |
| `scoring.py` | Programmatic scoring functions: `answer_in_context`, `token_f1`, `classify_failure`, `score_dataframe` |
| `orchestrator.py` | `QAEvalPipeline` -- multi-model orchestrator with `evaluate()` (dict API) and `process()` (DataFrame API) |
| `ground_truth.py` | Dataset loaders: `bo767_infographic`, `vidore/*`, and generic `csv:` loader |
| `text_utils.py` | Shared text processing (`strip_think_tags`) |
| `runner.py` | `run_eval_sweep()` -- reusable config-driven evaluation loop |
| `cli.py` | `retriever eval` Typer subcommands (`run`, `export`, `build-page-index`) |

### Graph framework integration

The eval operators integrate with the same graph framework used by the ingestion
pipeline (`nemo_retriever.graph`):

| Graph component | Role in eval |
|-----------------|-------------|
| `AbstractOperator` | Base class providing `preprocess` / `process` / `postprocess` lifecycle and `>>` chaining |
| `CPUOperator` | Mixin marking eval operators as CPU-only (no GPU resources required) |
| `Node` / `Graph` | Wraps operators into a traversable DAG; `Graph.execute()` runs DFS |
| `InprocessExecutor` | Runs the linear eval chain sequentially on pandas DataFrames |

The eval operators are **decoupled** from the ingestion graph. They consume
pre-computed artifacts (retrieval JSON + ground-truth CSV) and can be used
independently of any ingestion or retrieval pipeline.

### Entry Points

| Entry Point | Use Case |
|-------------|----------|
| `retriever ingest` | Ingest supported documents and media into LanceDB. |
| `retriever eval build-page-index` | Build full-page markdown index from Parquet extraction results |
| `retriever eval export` | Export retrieval from NeMo Retriever LanceDB (supports full-page markdown) |
| `retriever eval run` | QA eval runner -- env-var (`--from-env`) or config-driven (`--config`), uses `QAEvalPipeline` |
| `load_eval_config()` + `build_eval_chain()` | Config-driven single-model eval via operator graph chain |
| `load_eval_config()` + `build_eval_pipeline()` | Config-driven multi-model eval via `QAEvalPipeline` |

## Configuration

### Eval config file (YAML / JSON)

For programmatic or CI use, `nemo_retriever.evaluation.config` provides a
standalone configuration layer.  Two schema formats are supported.

#### New format: `models` + `evaluations` (recommended)

Define models once, compose generator/judge combos with per-combo run counts.
Any model can serve as generator or judge.

```yaml
dataset:
  source: "csv:data/bo767_annotations.csv"

retrieval:
  type: "file"
  file_path: "data/eval/bo767_retrieval_fullpage.json"

models:
  generator-a:
    model: "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
    api_key: "${NVIDIA_API_KEY}"

  generator-b:
    model: "openai/my-org/my-model"
    api_base: "https://your-openai-compatible-endpoint/v1"
    api_key: "${GEN_API_KEY}"

  nemotron-lightning-judge:
    model: "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
    api_key: "${NVIDIA_API_KEY}"
    temperature: 0.1
    max_tokens: 4096

evaluations:
  - generator: "generator-b"
    judge: "nemotron-lightning-judge"
    runs: 2

  - generator: "generator-a"
    judge: "nemotron-lightning-judge"
    runs: 5

execution:
  top_k: 5
  max_workers: 8
```

| Section | Purpose |
|---------|---------|
| `models` | Dict of model definitions keyed by short name. Fields: `model` (LiteLLM string), `api_base`, `api_key`, `temperature`, `top_p`, `max_tokens`, `reasoning_enabled`, `extra_params`, and `num_retries`. |
| `evaluations` | List of generator+judge combos. Each entry references model names and has an optional `runs` count (default: `execution.runs`, then 1). Per-evaluation `temperature`, `top_p`, `max_tokens`, and `reasoning_enabled` overrides are supported by the sweep runner. |
| `execution` | Shared settings: `top_k`, `max_workers`, `runs` (default for evaluations that omit it), `min_coverage` (abort if retrieval covers fewer queries than this fraction, default 0.0). |
| `output` | Optional `results_dir` (default: `data/eval`). |

**Config-driven sweep** -- run all evaluations with one command:

```bash
export GEN_API_KEY="your-api-key"    # for generators using an OpenAI-compatible endpoint
export NVIDIA_API_KEY="nvapi-..."    # for NIM models
retriever eval run --config nemo_retriever/examples/eval_sweep.yaml
```

Each run produces a timestamped output file:
`qa_results_{dataset}_{generator}_{judge}_{run}of{total}_{YYYYMMDD_HHMMSS}.json`.
Failed runs are logged and skipped; a pass/fail summary prints at the end.

#### Legacy format: `generators` + `judge`

Still accepted and auto-normalised internally.  Every generator shares the
same judge.

```yaml
generators:
  - name: "nemotron"
    model: "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
    api_key: "${NVIDIA_API_KEY}"
    temperature: 0.0
    max_tokens: 4096

judge:
  model: "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
  api_key: "${NVIDIA_API_KEY}"
  temperature: 0.1
  max_tokens: 4096

execution:
  top_k: 5
  max_workers: 8
```

Environment variables are expanded in string values: `${VAR}` resolves to
`os.environ["VAR"]` at load time. Secrets never live in the config file.

**Programmatic usage:**

```python
from nemo_retriever.evaluation.config import load_eval_config, build_eval_chain

config = load_eval_config("eval_config.yaml")
graph = build_eval_chain(config, model_name="nemotron")
result_df = graph.execute(None)
```

### Model Strings

LiteLLM routes by prefix:

| Prefix | Provider | Example |
|--------|----------|---------|
| `nvidia_nim/` | NVIDIA NIM (build.nvidia.com) | `nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b` |
| `openai/` | OpenAI or any OpenAI-compatible server | `openai/gpt-4o` |
| `huggingface/` | HuggingFace Inference Endpoints | `huggingface/meta-llama/Llama-3-70b-instruct` |

For local vLLM/Ollama, use `openai/<model>` with `api_base: http://localhost:8000/v1`.

### In-Memory LanceDB Retrieval

Instead of pre-exporting a retrieval JSON, the eval pipeline can query LanceDB
live and feed results directly to generation/judging. Set `retrieval.type` to
`"lancedb"` in your config:

```yaml
retrieval:
  type: "lancedb"
  lancedb_uri: "lancedb"
  lancedb_table: "nemo-retriever"
  embedder: "nvidia/llama-nemotron-embed-1b-v2"
  save_path: "data/eval/bo767_retrieval.json"   # optional -- persists for re-runs
  page_index: "data/bo767_page_markdown.json"    # optional -- full-page expansion
```

| Field | Default | Description |
|-------|---------|-------------|
| `type` | `"file"` | `"file"` reads a pre-exported JSON; `"lancedb"` queries live |
| `lancedb_uri` | `"lancedb"` | LanceDB directory path |
| `lancedb_table` | `"nemo-retriever"` | Table name inside the LanceDB directory |
| `embedder` | `"nvidia/llama-nemotron-embed-1b-v2"` | Embedding model for query encoding |
| `save_path` | *none* | If set, writes the retrieval JSON for later `type: "file"` re-runs |
| `page_index` | *none* | JSON page-markdown index for full-page chunk expansion |

Three usage modes from a single config:

- **End-to-end + save**: `type: lancedb, save_path: retrieval.json` -- queries live, saves for future, runs eval
- **End-to-end no save**: `type: lancedb` -- queries live, pure in-memory, runs eval
- **File-only (existing)**: `type: file, file_path: retrieval.json` -- unchanged, reads saved file

**CLI (--from-env mode):** set `LANCEDB_URI` instead of `RETRIEVAL_FILE`:

```bash
export LANCEDB_URI="lancedb"
export LANCEDB_TABLE="nemo-retriever"           # optional, default: nemo-retriever
export EMBEDDER="nvidia/llama-nemotron-embed-1b-v2"  # optional
export RETRIEVAL_SAVE_PATH="data/eval/retrieval.json" # optional
export QA_DATASET="csv:data/bo767_annotations.csv"
retriever eval run --from-env
```

**Live LanceDB evaluation:**

Ingest through the supported CLI, then run evaluation against that table.

```bash
retriever ingest /data/pdfs \
    --lancedb-uri lancedb \
    --table-name nemo-retriever

export LANCEDB_URI="lancedb"
export LANCEDB_TABLE="nemo-retriever"
export QA_DATASET="csv:data/bo767_annotations.csv"
export RETRIEVAL_SAVE_PATH="data/eval/bo767_retrieval.json"
retriever eval run --from-env
```

### Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `NVIDIA_API_KEY` | Config expansion, fallback for `GEN_API_KEY`/`JUDGE_API_KEY` | Default API key for NIM models |
| `NVIDIA_NIM_API_KEY` | litellm's `nvidia_nim` provider | Alias -- set to same value as `NVIDIA_API_KEY` |
| `GEN_API_KEY` | `retriever eval run --from-env`, config `${GEN_API_KEY}` | Generator API key (falls back to `NVIDIA_API_KEY`) |
| `JUDGE_API_KEY` | `retriever eval run --from-env`, config `${JUDGE_API_KEY}` | Judge API key (falls back to `NVIDIA_API_KEY`) |
| `LANCEDB_URI` | `retriever eval run --from-env` (lancedb mode) | LanceDB directory path (activates lancedb mode when `RETRIEVAL_FILE` is unset) |
| `LANCEDB_TABLE` | `retriever eval run --from-env` (lancedb mode) | LanceDB table name (default: `nemo-retriever`) |
| `EMBEDDER` | `retriever eval run --from-env` (lancedb mode) | Embedding model name |
| `RETRIEVAL_SAVE_PATH` | `retriever eval run --from-env` (lancedb mode) | Optional path to persist the retrieval JSON |

### Mixing providers per component

Every component (`Retriever`, `LiteLLMClient`, `LLMJudge`) takes its own `(model, api_base, api_key)` triple, so you can point different stages at different endpoints without any extra wiring. For LLM clients, an unset `api_key` is left to LiteLLM's provider-native environment lookup on the worker. Only provide an explicit `api_key` when a stage needs a distinct credential or an explicit environment reference.

```python
import os
from nemo_retriever.retriever import Retriever
from nemo_retriever.models.llm import LiteLLMClient, LLMJudge

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
    model="openai/gpt-4o-mini",
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.2,
)

judge = LLMJudge.from_kwargs(
    model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.1,
    max_tokens=4096,
)
```

Transport params redact the bearer token in their `__repr__` / `__str__` (`api_key=***`), so logging a client or letting a Pydantic validation error echo a params object back will not leak credentials.

### Batch generation failure rate

When the fluent builder ran `.generate(...)`, the returned `DataFrame` exposes the aggregate generation failure rate on `df.attrs` so batch-eval reports can surface it alongside `token_f1` / `judge_score`:

```python
df = retriever.pipeline().generate(llm).run(queries=questions)
print(f"generation_failure_rate = {df.attrs['generation_failure_rate']:.2%}")
```

`generation_failure_rate` is the fraction of rows whose `gen_error` column is non-null; it is only attached when the `.generate()` step ran.

### Generation run metadata

Sweep result JSON includes a secret-free `generation_config` object containing the effective `model`, `temperature`, `top_p`, `max_tokens`, and `reasoning_enabled` values. Per-evaluation overrides are resolved before this object is written, so result files can be compared without reopening the input config. API keys, endpoint credentials, and `extra_params` are intentionally excluded.

### Generation error codes

`gen_error` is `null` on success. The reusable text-generation lifecycle emits these stable values on failure:

| Code | Meaning |
|------|---------|
| `empty_input` | A task rejected an empty source value before calling the provider. |
| `request_error` | Input validation or provider-neutral request construction failed. |
| `transport_error` | The provider request failed after the configured client retry behavior. |
| `unsupported_response` | The provider returned tools, multiple choices, non-text content, or another unsupported text-response shape. |
| `parse_error` | The task could not convert returned text into its expected text result. |
| `empty_output` | A generic text task produced no usable text after parsing. |
| `thinking_truncated` | RAG answer generation contained only model reasoning and no final answer, usually because the token budget was exhausted. |

Errors are sanitized before they enter a DataFrame or result JSON; raw provider exception text is not persisted.

### Stable public surface

`nemo_retriever.models.llm.__all__` is the supported integration point for the client + types layer. Import the Protocols, result dataclasses, `LiteLLMClient`, `LLMJudge`, and the `LLMInferenceParams` / `LLMRemoteClientParams` models from `nemo_retriever.models.llm`; deeper submodule paths (`models.llm.clients.litellm`, `models.llm.text_utils`, ...) are implementation details and may be reorganised without notice.

The previously-provided `nemo-retriever[eval]` install extra was removed in favor of `nemo-retriever[llm]`; pin the new extra in any requirements files that still reference `[eval]`.

## Scoring System (Three-Tier Hierarchy)

Each (query, model) pair is scored by three independent tiers. Each tier tests a different layer of the RAG pipeline:

| Tier | Name | What it measures | Method | Key question |
|------|------|-----------------|--------|-------------|
| **1** | Retrieval quality | Did the retriever return chunks containing the answer? | Programmatic | "Are the reference answer's content words present in the retrieved context?" |
| **2** | Answer quality (token) | Does the generated answer overlap with the reference at the token level? | Programmatic | "How precisely did the model reproduce the expected facts?" |
| **3** | Answer quality (semantic) | Is the generated answer factually correct given the reference? | LLM-as-judge | "Did the model get the right facts, allowing paraphrasing and numeric equivalence?" |

### Tier 1: `answer_in_context` (retrieval signal)

Returns `True` if >= 50% of the reference answer's **content words** (stopwords removed, normalized) appear in the concatenated retrieved chunks. If `False`, the retriever failed to surface the answer -- any generation failure is a retrieval miss, not a model miss.

### Tier 2: `token_f1` (SQuAD-style token metrics)

After normalizing both reference and candidate (lowercase, strip punctuation, numeric equivalence like `16.00%` -> `16%`):

| Field | Formula | Meaning |
|-------|---------|---------|
| `precision` | `common_tokens / candidate_tokens` | Fraction of the model's output tokens that are relevant |
| `recall` | `common_tokens / reference_tokens` | Fraction of the reference's tokens the model captured |
| `f1` | `2 * P * R / (P + R)` | Harmonic mean of precision and recall |

### Tier 3: `judge_score` (LLM-as-judge, 0.0-1.0 scale)

`LLMJudge` is a from-scratch port of ragas'
[`AnswerAccuracy`](https://pypi.org/project/ragas/) metric -- it reproduces the
**dual-judge** logic directly on `LiteLLMClient`, with no ragas / instructor /
openai dependency:

1. **Judge 1** rates the candidate against the reference.
2. **Judge 2** repeats the rating with the candidate/reference roles swapped
   (a bidirectional equivalence check), under a paraphrased rubric.
3. Each judge returns `0` (no match), `2` (partial match), or `4` (exact
   match); each is normalised to `0.0-1.0` and the two are averaged
   (NaN-aware: a single valid judge still scores).

The two prompts and their few-shot examples are copied verbatim from ragas
(`AnswerAccuracyJudge1Prompt` / `AnswerAccuracyJudge2Prompt`), rendered through
a faithful reproduction of `BasePrompt.to_string`, and run against the judge
model via `litellm` (NVIDIA NIM by default). `instructor` constrained decoding
is replaced by the prompts' own `{"rating": X}` JSON contract plus a tolerant
parser. The dual `0/2/4` design yields one of `{0.0, 0.25, 0.5, 0.75, 1.0}`.
`AnswerAccuracy` emits **only** a numeric rating, so `judge_reasoning` is
always empty.

| `judge_score` | `classify_failure` band |
|---------------|-------------------------|
| `>= 0.75` (`0.75`, `1.0`) | `correct` |
| `0.25 - 0.5` | `partial` (or refusal variants) |
| `< 0.25` (`0.0`) | `retrieval_miss` / `generation_miss` / refusal variants |

> Migration note: the previous judge used a custom single-call `1-5` rubric.
> Scores from before this change are **not** comparable; re-baseline any saved
> judge numbers, thresholds, and cached `agent_eval` run JSON.

### Output fields per query

| Field | Source | Type | Description |
|-------|--------|------|-------------|
| `query` | Input | `str` | The ground-truth question |
| `reference_answer` | Input | `str` | Ground-truth answer from the dataset |
| `context` | Tier 1 | `list[str]` | Retrieved chunks passed to the generator |
| `answer_in_context` | **Tier 1** | `bool` | >= 50% of reference content words found in chunks |
| `answer` | Generation | `str` | The model's generated answer |
| `gen_latency_s` | Generation | `float` | Generation API call latency (seconds) |
| `gen_error` | Generation | `str?` | Error string if generation failed, else `null` |
| `token_f1` | **Tier 2** | `float` | SQuAD-style F1 (0.0-1.0) |
| `judge_score` | **Tier 3** | `float?` | ragas AnswerAccuracy score (0.0-1.0), `null` on error |
| `judge_reasoning` | **Tier 3** | `str` | Always empty -- AnswerAccuracy emits only a rating |
| `judge_error` | Tier 3 | `str?` | Error string if judging failed, else `null` |
| `failure_mode` | Derived | `str` | Classification (see below) |

### Failure mode classification

Combines Tier 1 + Tier 3 to diagnose **where** a failure occurred:

| `failure_mode` | Condition | What it means |
|----------------|-----------|---------------|
| `correct` | `judge_score >= 4` | Model answered correctly |
| `partial` | `judge_score` 2-3, answer does not claim "no context" | Model gave an incomplete but relevant answer |
| `refused_missing_context` | `judge_score` 1-3, model claims "no context", AND `answer_in_context == False` | Model correctly reported that the answer was not in the retrieved chunks |
| `refused_with_context` | `judge_score` 1-3, model claims "no context", AND `answer_in_context == True` | Model falsely claimed the answer was missing despite it being in the chunks |
| `retrieval_miss` | `judge_score <= 1`, answer does NOT claim "no context", AND `answer_in_context == False` | Retriever failed to surface the answer |
| `generation_miss` | `judge_score <= 1`, answer does NOT claim "no context", AND `answer_in_context == True` | Retriever had the answer, model missed it (wrong answer, not a refusal) |
| `thinking_truncated` | `gen_error == "thinking_truncated"` | Reasoning model exhausted token budget before producing an answer |
| `judge_error` | `judge_score` is `null` AND `gen_error != "thinking_truncated"` | Judge API failed (timeout, parse error, rate limit); query could not be scored |

The **`refused_missing_context`** vs **`refused_with_context`** split replaces the old `no_context` bucket. Both indicate the model said "context does not contain..." but they differ on whether that was true: `refused_missing_context` = correct refusal (retrieval problem), `refused_with_context` = false refusal (generation problem).

### Extending the scoring system

All scoring is applied by `score_dataframe()` in `nemo_retriever.evaluation.scoring`. To add a custom metric:

1. **Add a scoring function** in `scoring.py` (or your own module) that takes a reference answer, candidate answer, and/or context and returns a value:

```python
def my_custom_metric(reference: str, candidate: str) -> float:
    """Example: character-level Levenshtein similarity."""
    # your logic here
    return similarity_score
```

2. **Wire it into `score_dataframe()`** by appending a new column in the iteration loop, or create your own post-processing function that takes the output DataFrame and adds columns:

```python
def add_custom_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["my_metric"] = df.apply(
        lambda row: my_custom_metric(row["reference_answer"], row["answer"]),
        axis=1,
    )
    return df
```

3. **Apply after the pipeline** -- `QAEvalPipeline.process()` returns a DataFrame. Chain your scoring on top:

```python
result_df = pipeline.process(input_df)
result_df = add_custom_scores(result_df)
```

The same pattern works for custom failure classifiers, alternative judge prompts (subclass `LLMJudge` and override the system prompt), or entirely new tiers. All components follow the Protocol pattern -- no inheritance or registration required.

## Adding a New Component

### Custom Retriever

```python
from nemo_retriever.models.llm.types import RetrieverStrategy, RetrievalResult

class MyRetriever:
    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        chunks = my_search(query, top_k)
        return RetrievalResult(chunks=chunks, metadata=[])
```

### Custom LLM Client

```python
from nemo_retriever.models.llm.types import LLMClient, GenerationResult

class MyClient:
    def generate(self, query: str, chunks: list[str]) -> GenerationResult:
        answer = my_llm(query, chunks)
        return GenerationResult(answer=answer, latency_s=0.0, model="my-model")
```

### Custom Judge

```python
from nemo_retriever.models.llm.types import AnswerJudge, JudgeResult

class MyJudge:
    def judge(self, query: str, reference: str, candidate: str) -> JudgeResult:
        score = my_scoring_logic(reference, candidate)
        return JudgeResult(score=score, reasoning="...")
```

No registration step needed -- pass the instance directly to `QAEvalPipeline`.

## Output Format

`_test_results.json` structure:

```json
{
  "dataset": "csv:data/bo767_annotations.csv",
  "retrieval_file": "data/eval/bo767_retrieval_fullpage.json",
  "top_k": 5,
  "qa_results": {
    "summary": {
      "total_submitted": 1005,
      "total_completed": 1005,
      "dropped_queries": 0
    },
    "tier1_retrieval": {
      "answer_in_context_rate": 0.8816,
      "answer_in_context_count": 886,
      "total": 1005
    },
    "tier2_programmatic": {
      "generator": {
        "mean_token_f1": 0.1196
      }
    },
    "tier3_llm_judge": {
      "generator": {
        "mean_score": 0.69,
        "score_distribution": {"0.0": 251, "0.25": 44, "0.5": 28, "0.75": 26, "1.0": 621},
        "mean_latency_s": 11.7,
        "scored_count": 970,
        "error_count": 35
      }
    },
    "failure_breakdown": {
      "generator": {
        "correct": 647, "partial": 65,
        "retrieval_miss": 37, "generation_miss": 89,
        "thinking_truncated": 35, "refused_missing_context": 95,
        "refused_with_context": 37
      }
    },
    "per_query": [
      {
        "query": "How much did Pendleton County spend out of their COVID-19 fund for the month of April 2021?",
        "reference_answer": "$205.43",
        "retrieved_chunk_count": 2,
        "answer_in_context": true,
        "token_f1": {"generator": {"f1": 0.057}},
        "failure_mode": {"generator": "correct"},
        "retrieved_chunks": ["## Page 2\n\n..."],
        "retrieval_metadata": [{"source_id": "1003421.pdf", "page_number": 2, "distance": 1.037}],
        "generations": {"generator": {"answer": "...$205.43...", "latency_s": 12.2}},
        "judgements": {"generator": {"score": 5, "reasoning": "The required fact '$205.43' is present..."}}
      }
    ]
  }
}
```

## Dataset Limitations

### bo767_annotations.csv (default)

The default ground truth (`data/bo767_annotations.csv`) contains 1007 Q&A pairs
across all modalities (text, table, chart, infographic) for 767 bo767 PDFs.

1. **Short factual answers**: Most reference answers are 1-5 words (e.g., "$205.43", "5"). Tier 2 programmatic metrics (exact match, F1) carry strong signal. For open-ended datasets, Tier 3 (LLM judge) becomes primary.

2. **Retrieval =/= QA quality**: A retrieval method that returns the correct page may still get a low QA score if the extracted text is garbled or incomplete. Always check Tier 1 first -- if `answer_in_context_rate` is low, the problem is retrieval or extraction, not the generator.

3. **Full-page markdown recommended**: Sub-page chunks may split structured content (tables, charts) across multiple records. The full-page markdown pipeline (step 2 in reproduction) reconstructs complete pages, matching the research team's approach and improving generation accuracy.

4. **Reasoning model truncation**: Models with extended thinking (for example, Nemotron 3.5 Lightning) may spend their token budget reasoning and never produce a final answer. The pipeline detects this (`thinking_truncated`) and nullifies the score.

5. **Model refusal failures**: The model sometimes responds "no information found" even when the answer is in the retrieved chunks. The failure breakdown splits these into `refused_missing_context` (answer genuinely absent -- retrieval problem) vs `refused_with_context` (answer present but model refused -- generator problem). Together they account for ~13% of queries in the reference run.

## Standalone Runner (Docker / CI)

For Docker or CI environments:

```bash
export RETRIEVAL_FILE=/data/bo767_retrieval.json
export QA_DATASET="csv:data/bo767_annotations.csv"
export NVIDIA_API_KEY="nvapi-..."
retriever eval run --from-env
```

Run `retriever eval run --help` for the full list of supported environment variables.
