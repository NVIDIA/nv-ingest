<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# retrieval-bench

Pipeline evaluation framework for document retrieval.

This package depends on `vidore-benchmark` for shared framework modules that are
not vendored in this slim repository snapshot.

## Security note

All model backends load HuggingFace models with `trust_remote_code=True`, which
executes Python code shipped with the model repository. Only use model IDs from
sources you trust.

## Installation

```bash
pip install -e .
```

### Model dependencies

Most models work with `transformers==4.51.0`:

```bash
pip install transformers==4.51.0
pip install flash-attn==2.6.3 --no-build-isolation
```

The `nemotron-colembed-vl-8b-v2` model requires a newer transformers release:

```bash
pip install transformers==5.0.0rc0
pip install flash-attn==2.6.3 --no-build-isolation
```

## Dense retrieval

Retrieval-only evaluation with a pluggable backend. The `--backend` flag
selects which retriever to use.

```bash
retrieval-bench evaluate dense-retrieval \
  --dataset-name bright/biology \
  --backend llama-nv-embed-reasoning-3b

retrieval-bench evaluate dense-retrieval \
  --dataset-name vidore/vidore_v3_hr \
  --backend llama-nemotron-embed-vl-1b-v2 \
  --language english

retrieval-bench evaluate dense-retrieval \
  --dataset-name vidore/vidore_v3_hr \
  --backend nemotron-colembed-vl-8b-v2 \
  --language english
```

Backend-specific overrides can be passed via `--pipeline-args` JSON:

```bash
retrieval-bench evaluate dense-retrieval \
  --dataset-name bright/biology \
  --backend llama-nv-embed-reasoning-3b \
  --pipeline-args '{"model_id":"~/checkpoints/my_model","max_scoring_batch_size":2048}'
```

## Agentic retrieval

Dense retrieval augmented with an LLM agent that iteratively refines results.

```bash
retrieval-bench evaluate agentic-retrieval \
  --dataset-name bright/biology \
  --backend llama-nv-embed-reasoning-3b \
  --llm-model your-llm-model \
  --num-concurrent 10

retrieval-bench evaluate agentic-retrieval \
  --dataset-name vidore/vidore_v3_hr \
  --backend llama-nemotron-embed-vl-1b-v2 \
  --llm-model your-llm-model
```

By default the pipeline reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from
environment variables. Override them via `--pipeline-args`:

```bash
retrieval-bench evaluate agentic-retrieval \
  --dataset-name bright/biology \
  --backend llama-nv-embed-reasoning-3b \
  --llm-model your-llm-model \
  --pipeline-args '{"api_key":"os.environ/MY_KEY","base_url":"os.environ/MY_URL"}'
```

### Available backends

| Backend | Retriever |
|---------|-----------|
| `llama-nv-embed-reasoning-3b` | NeMo Reasoning dense retriever |
| `llama-nemoretriever-colembed-3b-v1` | ColEmbed late-interaction retriever |
| `llama-nemotron-embed-vl-1b-v2` | Nemotron Embed VL 1B v2 (multimodal dense) |
| `nemotron-colembed-vl-8b-v2` | Nemotron ColEmbed VL 8B v2 (multimodal late-interaction) |

## Utilities

```bash
retrieval-bench evaluate utils list-datasets
retrieval-bench evaluate utils list-backends
retrieval-bench evaluate utils report-results --results-dir results/my_run
retrieval-bench evaluate utils compare-results --results-dirs results/run_a --results-dirs results/run_b
```
