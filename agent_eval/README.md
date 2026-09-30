# agent_eval

An agent-agnostic harness to evaluate how well a coding agent (**claude** or **codex**)
uses the nemo-retriever skill over a document corpus. It has **two evaluation tracks**:

1. **Recall eval** (this section) — Q&A over a corpus, **with** the skill vs. a **baseline**
   with no retriever, scored on **recall@k + LLM judge**.
2. **Functional eval** ([jump ↓](#functional-tests-behavioral-passfail-eval)) — the domain-less
   `functional_corpus_variants` behavioral prompts (ingest a folder, multi-hop Q&A, counts),
   graded **pass/fail**.

Every script is built to be **re-runnable in a different environment**: the inputs (manifest,
corpus, skill, retriever CLI, judge endpoint) are all CLI flags, and the agent-running scripts
carry no `nemo_retriever` import so they copy to a clean box. Each section below lists exactly
which paths to repoint.

Three stages, three scripts (recall track):

```
extract_queries.py   manifest.json         ->  queries.json          (answer-free)
run_agent_eval.py    queries.json          ->  <save-root>/<run_id>/ (per-question artifacts)
build_report.py      run dir(s) + manifest ->  report.md/json + comparison.md
```

- **`run_agent_eval.py` is standalone** (stdlib + the local modules only — no
  `nemo_retriever` import). Copy `agent_eval/` to a clean machine to run a baseline
  with no retriever installed.
- **`build_report.py` may import `nemo_retriever`** (reuses `recall_at_k` + `LLMJudge`),
  so run it where the codebase exists.

---

## 1. `extract_queries.py` — manifest → answer-free queries

Pulls the agent-facing prompt out of an eval manifest into a `queries.json` containing
**only** `{query_id, prompt, domain}` per query. No answer, no gold pages, no
`category`/`scoring_mode` (so a `refusal` label can't tip off the agent). Gold is
recovered later by `build_report.py`, which re-reads the same manifest.

Handles both manifest shapes: the corpus-level manifest's top-level `prompt`, and the
scenario manifest's `scenario_prompt_candidates.candidates[].prompt`. An entry is
included iff it has a `primary_eval_id` and `prompt_export_status == "exported"`.

```bash
python3 extract_queries.py \
  --manifest /raid/retriever-sdg-v3/runs/.../agent_corpus_level_manifest.json \
  --out      /raid/retriever-sdg-v3/runs/.../queries.json
```

| arg | default | meaning |
|---|---|---|
| `--manifest` | *(required)* | Path to the eval manifest JSON. |
| `--out` | `queries.json` | Output path for the answer-free queries file. |
| `--categories` | none | Comma-separated category filter, e.g. `query,extract`. Filtering happens here so the label never enters the file. |
| `--domains` | none | Comma-separated domain filter. |
| `--limit` | none | Cap the number of queries (smoke tests). |

Output: `{source_manifest, extracted_at, count, queries:[{query_id, prompt, domain}]}`.

---

## 2. `run_agent_eval.py` — run an agent over the queries

For each query: builds an isolated workdir (corpus symlinked in), runs the agent for a
single turn, and saves a per-question subfolder. The agent must write `./output.json`
with `final_answer` + ranked `selected_chunks` (top-k pages).

**Profiles** (the with/without-retriever axis):
- `baseline` — no skill; `retriever` blocked (PATH-shim + Claude `settings.json` deny +
  empty HF cache). Agent answers from `./pdfs/` with native tools. Emits 0-indexed pages.
- `skill` — the nemo-retriever skill is copied in and the real `retriever` CLI is used.
  The index is built by an **agent setup turn** (`retriever ingest`, measured) unless a
  prebuilt index is supplied. Emits the retriever's raw 1-indexed pages (the report
  normalizes via `page_index_base`).

```bash
# Baseline (no GPU; crank parallelism):
python3 run_agent_eval.py --queries queries.json \
  --agent claude --model claude-opus-4-7 --profile baseline \
  --save-root /tmp/runs --domains vidore_v3_finance_en,vidore_v3_hr,vidore_v3_pharmaceuticals \
  --parallelism 48 --timeout 900

# Skill, fresh indexes, GPU-pinned (one engine per GPU):
python3 run_agent_eval.py --queries queries.json \
  --agent codex --model gpt-5.5 --profile skill \
  --save-root /tmp/runs --domains vidore_v3_finance_en,vidore_v3_hr,vidore_v3_pharmaceuticals \
  --parallelism 8 --gpu-list 0,1,2,3 --timeout 7200 \
  --retriever-bin /raid/nemo_retriever/.venv/bin/retriever

# Skill, reuse indexes from a prior run (skip GPU ingest):
python3 run_agent_eval.py --queries queries.json --agent claude --profile skill \
  --save-root /tmp/runs --prebuilt-index-root /tmp/runs/agenteval_claude_skill_<ts> --gpus 8 --parallelism 8
```

| arg | default | meaning |
|---|---|---|
| `--queries` | *(required)* | The `queries.json` from stage 1. |
| `--agent` | `claude` | `claude` or `codex`. |
| `--model` | `claude-opus-4-7` | Model id (e.g. `gpt-5.5` for codex) — editable. |
| `--profile` | `skill` | `baseline` (no retriever) or `skill`. |
| `--corpus-root` | `/raid/data/vidore_v3` | Domain corpora live at `<corpus-root>/<domain>/*.pdf`. |
| `--domains` | all | Comma-separated domain filter (each must have a corpus dir). |
| `--limit` | none | Cap query count. |
| `--save-root` | `./agent_eval_runs` | Where to write `<run_id>/`. **Keep this OUTSIDE the repo** for baseline (else the skill/`.claude` leaks in — a guard blocks it). |
| `--top-k` | `10` | Chunks the agent should return. |
| `--parallelism` | `4` | Concurrent query turns. Baseline: high (no GPU). Skill: ≈ GPUs (or 2× with pinning). |
| `--timeout` | `2400` | Per-turn wall-clock seconds (raise to ~7200 for full `retriever ingest` setups). |
| `--budget-usd` | `5.0` | Per-trial budget (Claude). |
| `--gpus` | `0` (auto) | GPUs to round-robin query processes across (skill). 0 = auto-detect via `nvidia-smi`. |
| `--gpu-list` | none | Explicit physical GPU ids, e.g. `0,1,2,3`. Overrides `--gpus`; lets two concurrent runs use disjoint halves. Pins setup ingests too. |
| `--skill-src` | `…/skills/nemo-retriever` | Skill dir to copy into skill-profile workdirs. |
| `--retriever-bin` | `…/.venv/bin/retriever` | The `retriever` CLI (skill profile). |
| `--embed-model` | `nvidia/llama-nemotron-embed-1b-v2` | Embed model for subprocess ingest. |
| `--prebuilt-index` | none | Reuse one `lancedb` dir (single-domain). |
| `--prebuilt-index-root` | none | Reuse per-domain indexes from a prior run: `<root>/_setup/skill_<domain>/lancedb`. Skips GPU ingest. |
| `--subprocess-ingest` | off | Build the index via a direct `retriever ingest` subprocess instead of an agent setup turn (no setup tokens). |
| `--skip-setup` | off | Skill: don't ingest (assume index already present). |
| `--allow-unsafe-save-root` | off | Proceed even if the save-root would un-blind a baseline. |
| `--dry-run` | off | Build workdirs + print the agent command; no agent calls. |

**Output** — `<save-root>/<run_id>/`:
- `run_config.json`, `run_metas.json`, `setup_metas.json`
- `_setup/<profile>_<domain>/` — corpus symlinks + built `lancedb/` (+ `setup_meta.json` for agent setups)
- `<query_id>/` — `output.json` (normalized chunks+answer), `agent_output.raw.json`,
  `meta.json` (status/tokens/cost/timing/session), `trace.md`, `agent_log.jsonl`, `prompt.txt`

`run_id` = `agenteval_<agent>_<profile>_<UTC-timestamp>`.

---

## 3. `build_report.py` — score runs into a report

Re-reads the manifest for gold (`relevant_pages`, `answer`), joins by `query_id`, and
writes `report.md`/`report.json` per run + a `comparison.md` across runs. Computes
`success_rate`, `recall@1/5/10` (normalized by each run's `page_index_base`), LLM
**judge**, refusal-correctness, and **retriever usage** (`attempted` /
`succeeded_clean` / `succeeded_engine`) with agent-aware log parsing (claude tool-events
vs codex `response_item` events). Re-runs reuse judge scores from an existing
`report.json` (judge cache), so re-scoring after a code change is fast.

```bash
# Compare two runs (auto-resolves the manifest from each run_config; judge on if NVIDIA_API_KEY set):
python3 build_report.py \
  /tmp/runs/agenteval_claude_baseline_<ts> /tmp/runs/agenteval_codex_baseline_<ts> \
  --manifest /raid/retriever-sdg-v3/runs/.../agent_corpus_level_manifest.json \
  --out /tmp/runs

# Recall + retriever only, no judge (fast):
python3 build_report.py /tmp/runs/agenteval_claude_skill_<ts> --no-judge
```

| arg | default | meaning |
|---|---|---|
| `run_dirs` | *(required, 1+)* | One or more run directories. >1 → also writes `comparison.md`/`json`. |
| `--manifest` | each run's `source_manifest` | Gold manifest. Override if it moved. |
| `--judge` / `--no-judge` | on if `$NVIDIA_API_KEY` set | Enable/disable LLM-as-judge. |
| `--judge-model` | `nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b` | Judge model. |
| `--judge-api-base` | `https://integrate.api.nvidia.com/v1` | Judge endpoint. |
| `--judge-api-key-env` | `NVIDIA_API_KEY` | Env var holding the judge API key. |
| `--out` | parent of first run dir | Where `comparison.md`/`json` are written. |

**Outputs:** `report.md` + `report.json` inside each run dir; `comparison.md` +
`comparison.json` in `--out` when multiple runs are passed.

Every report opens with a **`## Code version`** section: the git commit that *ran* the eval
(recorded at run time in `run_config.code_commit` — full sha, branch, dirty flag) plus the
report-time `HEAD`. Runs that predate commit-logging show the report-time `HEAD` with a
warning that the eval-time commit is unknown.

### Cost & token accounting

Cost and token totals are made apples-to-apples across agents inside `build_report.py`:

- **Claude** emits `total_cost_usd` directly (used as-is) and reports `input_tokens`
  *exclusive* of cached tokens (`cache_read` is separate).
- **Codex** emits **no dollar cost**, so it's derived from tokens via a per-model
  price table (`_PRICING`, $/1M):

  | model | input | cached_input | output |
  |---|---|---|---|
  | `gpt-5.5` | $5.00 | $0.50 | $30.00 |

  `cost = (fresh_input·input + cached·cached_input + output·output) / 1e6`, where
  `fresh_input = input_tokens − cached_input_tokens`. Codex's `output_tokens` already
  includes reasoning tokens, so they aren't added again. **To price a new model, add a
  row to `_PRICING`** (top of `build_report.py`); models not in the table fall back to
  whatever cost the CLI emitted (or `n/a`).

- **Token-total convention:** codex reports `input_tokens` *inclusive* of the cached
  subset (OpenAI style), so the report normalizes `input → input − cache_read`
  (`_norm_tokens`, codex only) **for the displayed totals** — otherwise the sum
  `input + output + cache_read` would double-count cached tokens. Cost is computed from
  the raw tokens *before* this normalization, so it's unaffected. Net: every run's
  `total` = `non-cached-input + output + cache_read` regardless of agent.

This is all report-side — no adapter change, no meta mutation — so it applies to
existing run artifacts on re-report (judge scores reuse the cache, so re-reporting is fast).

---

## End-to-end example

```bash
M=/raid/retriever-sdg-v3/runs/agent_corpus_level_batch_4/agent_corpus_level_manifest.json
DOM=vidore_v3_finance_en,vidore_v3_hr,vidore_v3_pharmaceuticals

# 1. queries
python3 extract_queries.py --manifest $M --out /tmp/queries.json

# 2. runs (baseline both agents in parallel; skill both agents on disjoint GPU halves)
python3 run_agent_eval.py --queries /tmp/queries.json --agent claude --model claude-opus-4-7 \
  --profile baseline --domains $DOM --save-root /tmp/base --parallelism 48 --timeout 900 &
python3 run_agent_eval.py --queries /tmp/queries.json --agent codex --model gpt-5.5 \
  --profile baseline --domains $DOM --save-root /tmp/base --parallelism 48 --timeout 900 &
wait

python3 run_agent_eval.py --queries /tmp/queries.json --agent claude --model claude-opus-4-7 \
  --profile skill --domains $DOM --save-root /tmp/skill --gpu-list 0,1,2,3 --parallelism 8 --timeout 7200 &
python3 run_agent_eval.py --queries /tmp/queries.json --agent codex --model gpt-5.5 \
  --profile skill --domains $DOM --save-root /tmp/skill --gpu-list 4,5,6,7 --parallelism 8 --timeout 7200 &
wait

# 3. reports
python3 build_report.py /tmp/base/agenteval_claude_baseline_* /tmp/base/agenteval_codex_baseline_* --manifest $M --out /tmp/base
python3 build_report.py /tmp/skill/agenteval_claude_skill_*   /tmp/skill/agenteval_codex_skill_*     --manifest $M --out /tmp/skill
```

## Notes / gotchas
- **Baseline save-root must be outside the repo** — otherwise the agent discovers
  `.claude/skills/nemo-retriever` by walking up the tree; a guard hard-blocks this
  (override with `--allow-unsafe-save-root`).
- **Skill GPU contention** — many concurrent query processes default to GPU 0 and OOM;
  use `--gpus`/`--gpu-list` to spread them (≈ 1–2 engines per GPU).
- **codex cost** is derived from tokens × `_PRICING` (the codex CLI emits no
  `total_cost_usd`) — see *Cost & token accounting* above. Update `_PRICING` if rates change.
- **codex `retr_succeeded_clean` is low by design** — codex's ~1s exec-yield backgrounds
  `retriever query`, so it rarely captures a clean exit even though the engine returns
  hits (see `retr_succeeded_engine`).

---

# Functional tests (behavioral pass/fail eval)

The recall pipeline above grades **recall@k**. A second track — the domain-less
`functional_corpus_variants` prompts — are **behavioral** tests (ingest a folder, answer a
multi-hop question, run a count) with **no gold pages**, graded **pass/fail**.

**Reproducibility is the point of this section:** four standalone scripts take you from a
manifest to a claude-vs-codex pass-rate report, and to recreate the experiment in a **different
environment** you only repoint a handful of paths (see *Prerequisites*).

```
extract_functional.py  manifest.json            ->  functional_queries.json  (answer-free)
run_functional.py      functional_queries.json  ->  <save-root>/<run_id>/    (per-query artifacts)
eval_functional.py     run dir + manifest       ->  functional_eval.json     (pass/fail verdicts)
functional_report.py   eval json(s)             ->  functional_report.md     (overall + by-type + per-agent)
```

Same dependency split as the recall track: **`run_functional.py` is standalone** (no
`nemo_retriever` import — copy `agent_eval/` to a clean box), **`eval_functional.py` may import
`nemo_retriever`** (reuses `LLMJudge` transport + `lancedb` row counts).

### Prerequisites (what a new environment needs, and how to repoint it)

| need | used by | repoint via (default) |
|---|---|---|
| The eval **manifest** (with `functional_corpus_variants` entries) | extract, eval | `--manifest` |
| The **corpus tree** the prompts reference at literal `test-data/...` paths | run | `--corpus-data-root` (`/raid/retriever-sdg-v3`) |
| The **nemo-retriever skill** directory | run | `--skill-src` (`…/skills/nemo-retriever`) |
| The **`retriever` CLI** venv + embed model | run | `--retriever-bin` (`…/.venv/bin/retriever`), `--embed-model` |
| **GPU(s)** for ingest/embed | run | `--gpu-list` / `--gpus` |
| **Judge** API key + endpoint | eval | `$NVIDIA_API_KEY`, `--judge-api-base` |
| `claude` and/or `codex` CLI on `PATH` | run | `--agent` |

### How tasks map to graders

Each prompt carries a `functional_type` (routing) and the manifest's `ground_truth_kind`:

| functional_type | grader |
|---|---|
| `retrieval_answer` | LLM **PASS/FAIL rubric** vs `validation_signal` + `expected_output_shape` |
| `ingest` (`action_contract`) | **programmatic gate** — a LanceDB table with rows>0 built in the workdir — AND a rubric check that the output is an ingest confirmation (not a Q&A answer) |
| `ingest_plus_answer` | both — the rows>0 gate AND the answer rubric |
| `stateful_orchestration` | **excluded** (multi-turn + mid-ingest SIGKILL; separate driver, not yet built) |

**Rubric design (this is why the numbers reproduce).** The manifest's `validation_signal`s
often reference signals the agent's *final message structurally cannot show* — tracer logs, job
IDs, backend endpoints, mid-run streaming progress. The rubrics deliberately **ignore** those
and grade only on *observable* output + the *programmatically-verified* artifact facts (rows>0).
Without this, ~80 ingest/ingest_plus tests false-FAIL. Full rationale + the artifact-vs-real
failure taxonomy is in `functional_findings.md`; the prompt→use-case mapping is in
`functional_uc_mapping.md`.

---

## F1. `extract_functional.py` — manifest → functional queries

Pulls the domain-less `functional_corpus_variants` prompts into an **answer-free**
`functional_queries.json` (`{query_id, prompt, functional_type, corpus_refs}`); pass/fail
criteria stay in the manifest (recovered later by `eval_functional.py`). `corpus_refs` are the
`test-data/...` directories parsed from each prompt (the runner mounts them); a **bare
`test-data` mention mounts the whole tree**. Excludes the 6 stateful tests by default.

```bash
python3 extract_functional.py \
  --manifest /raid/retriever-sdg-v3/runs/agent_corpus_level_batch_4/agent_corpus_level_manifest.json \
  --out      /raid/retriever-sdg-v3/runs/agent_corpus_level_batch_4/functional_queries.json
```

| arg | default | meaning |
|---|---|---|
| `--manifest` | *(required)* | Eval manifest JSON. |
| `--out` | `functional_queries.json` | Answer-free queries output. |
| `--include-stateful` | off | Include the 6 `stateful_orchestration` tests (driver not built). |
| `--limit` | none | Cap query count (smoke). |

→ 203 queries (119 `retrieval_answer` / 60 `ingest` / 24 `ingest_plus_answer`; 6 stateful excluded).

---

## F2. `run_functional.py` — run an agent (skill profile)

Per query: mounts each `corpus_ref` at its **literal** relative path (`test-data/...`), copies
the skill in, runs one agent turn, captures artifacts. `retrieval_answer` queries reuse **one
shared prebuilt index per corpus-group** (built once via subprocess `retriever ingest`);
`ingest`/`ingest_plus_answer` build the index **in-turn** (that's the graded action). Skill
profile only; GPU-pinned like the recall runner.

```bash
# claude on GPUs 0-3, codex on 4-7 (disjoint halves), all 203:
python3 run_functional.py --queries functional_queries.json \
  --agent claude --model claude-opus-4-7 --gpu-list 0,1,2,3 --parallelism 4 \
  --save-root /raid/agent_eval_func --timeout 3600 &
python3 run_functional.py --queries functional_queries.json \
  --agent codex  --model gpt-5.5      --gpu-list 4,5,6,7 --parallelism 4 \
  --save-root /raid/agent_eval_func --timeout 3600 &
wait
```

| arg | default | meaning |
|---|---|---|
| `--queries` | *(required)* | The `functional_queries.json` from F1. |
| `--agent` | `claude` | `claude` or `codex`. |
| `--model` | `claude-opus-4-7` | Model id (e.g. `gpt-5.5` for codex). |
| `--corpus-data-root` | `/raid/retriever-sdg-v3` | Root the `test-data/...` `corpus_refs` resolve under. **Repoint per environment.** |
| `--skill-src` | `…/skills/nemo-retriever` | Skill dir copied into each workdir. |
| `--retriever-bin` | `…/.venv/bin/retriever` | `retriever` CLI for shared-index builds. |
| `--embed-model` | `nvidia/llama-nemotron-embed-1b-v2` | Embed model for ingest. |
| `--save-root` | `./agent_eval_functional` | Where `<run_id>/` is written. |
| `--parallelism` | `4` | Concurrent query turns (≈ GPUs). |
| `--gpus` / `--gpu-list` | `0` / none | Round-robin GPU pinning; `--gpu-list 0,1,2,3` lets two runs use disjoint halves. |
| `--timeout` | `2400` | Per-turn seconds. **Raise to ≥3600** — financebench `ingest`/`ingest_plus` re-ingest 368 PDFs in-turn. |
| `--budget-usd` | `5.0` | Per-turn budget (Claude). |
| `--types` | all | CSV filter on `functional_type` (e.g. `ingest`). |
| `--limit` | none | Cap query count (smoke). |
| `--dry-run` | off | Mount corpora + print the agent command; no GPU/agent calls. |

**Output** — `<save-root>/<run_id>/` (`run_id = agenteval_func_<agent>_<UTC-ts>`):
`run_config.json`, `run_metas.json`, `_index/<corpus-group>/` (shared indexes), and per query
`<query_id>/` → `prompt.txt`, `response.txt`, `meta.json`
(`status`/`tokens`/`cost`/`lancedb_built`/`lancedb_tables`), `agent_log.jsonl`, `trace.md`.

> **Cost/time:** `ingest`/`ingest_plus` over financebench (368 PDFs) ingest in-turn (~30 min
> each) and may hit the timeout; those land as **non-results** (`run_timeout`), not logic fails.

---

## F3. `eval_functional.py` — grade pass/fail

Re-reads manifest criteria and grades each query by `ground_truth_kind` (table above). Judge
calls run **concurrently** (~45 s each — sequential is hours for ~200) and verdicts write
**incrementally**, so a killed run resumes from the cache.

```bash
python3 eval_functional.py \
  --run-dir /raid/agent_eval_func/agenteval_func_claude_<ts> \
  --manifest /raid/retriever-sdg-v3/runs/.../agent_corpus_level_manifest.json \
  --judge-concurrency 6
```

| arg | default | meaning |
|---|---|---|
| `--run-dir` | *(required)* | A `agenteval_func_*` run dir from F2. |
| `--manifest` | *(required)* | Manifest holding the pass/fail criteria. |
| `--out` | `<run-dir>/functional_eval.json` | Verdicts output. |
| `--min-rows` | `1` | rows>0 gate threshold for ingest tests. |
| `--judge` / `--no-judge` | on if `$NVIDIA_API_KEY` set | Enable/disable the LLM rubric. |
| `--judge-model` | `nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b` | Judge model. |
| `--judge-api-base` | `https://integrate.api.nvidia.com/v1` | Judge endpoint. |
| `--judge-api-key-env` | `NVIDIA_API_KEY` | Env var holding the judge key. |
| `--judge-concurrency` | `8` | Concurrent judge calls. |
| `--judge-timeout` / `--judge-retries` | `90` / `2` | Per-call timeout + retries (flaky endpoints). |
| `--regrade` | none | CSV of `query_id` substrings to drop from the cache and re-grade, e.g. `':e04:'` after a rubric change. |

**Output:** `functional_eval.json` = `{summary {overall, by_functional_type, status_counts},
per_query:[{query_id, passed, gate_pass, lancedb_rows, judge_verdict, reason, status}]}`.
Verdicts cache; re-runs reuse unchanged ones (use `--regrade` to force a subset).

---

## F4. `functional_report.py` — merge into a report

Merges one-or-more per-agent `functional_eval.json` into `functional_report.md`: overall
pass-rate, a `functional_type × agent` matrix, and per-agent failing-ID tables — **separating
non-results** (`run_timeout`/`judge_error`) from genuine logic FAILs so a setup outage isn't
read as a low pass-rate.

```bash
python3 functional_report.py \
  --eval claude=/raid/agent_eval_func/agenteval_func_claude_<ts>/functional_eval.json \
  --eval codex=/raid/agent_eval_func/agenteval_func_codex_<ts>/functional_eval.json \
  --out  /raid/agent_eval_func/functional_report.md
```

| arg | default | meaning |
|---|---|---|
| `--eval` | *(required, 1+)* | `label=path/to/functional_eval.json` (repeatable, one per agent/run). |
| `--out` | `functional_report.md` | Merged report path. |

The report opens with a **`## Code version`** section showing each run's eval-time commit
(from `run_config.code_commit`) plus the report-time `HEAD` — same provenance convention as the
recall `build_report.py`.

---

## End-to-end functional example

```bash
M=/raid/retriever-sdg-v3/runs/agent_corpus_level_batch_4/agent_corpus_level_manifest.json
Q=/raid/retriever-sdg-v3/runs/agent_corpus_level_batch_4/functional_queries.json
SR=/raid/agent_eval_func

# 1. queries (answer-free)
python3 extract_functional.py --manifest $M --out $Q

# 2. runs — claude (GPUs 0-3) + codex (4-7), all 203, in parallel
python3 run_functional.py --queries $Q --agent claude --model claude-opus-4-7 \
  --gpu-list 0,1,2,3 --parallelism 4 --save-root $SR --timeout 3600 &
python3 run_functional.py --queries $Q --agent codex  --model gpt-5.5 \
  --gpu-list 4,5,6,7 --parallelism 4 --save-root $SR --timeout 3600 &
wait

# 3. grade each run (concurrent judge), then merge
CL=$(ls -d $SR/agenteval_func_claude_*/ | tail -1)
CX=$(ls -d $SR/agenteval_func_codex_*/  | tail -1)
python3 eval_functional.py --run-dir $CL --manifest $M --judge-concurrency 6
python3 eval_functional.py --run-dir $CX --manifest $M --judge-concurrency 6
python3 functional_report.py --eval claude=$CL/functional_eval.json \
  --eval codex=$CX/functional_eval.json --out $SR/functional_report.md
```

## Functional notes / gotchas
- **Skill profile only** — these tests assume the retriever ingest/query pipeline (no baseline).
- **Financebench in-turn ingest is the bottleneck** — `ingest`/`ingest_plus` over its 368 PDFs
  ingest *inside the turn* (~30 min); raise `--timeout` or pre-build a shared index. Timeouts
  report as `run_timeout` non-results, not logic fails.
- **Judge throughput** — sequential grading of ~200 prompts at ~45 s/call is hours; always pass
  `--judge-concurrency 6–8`. Incremental writes make grading resumable (re-run reuses the cache).
- **Rubrics ignore unobservable internal signals** (tracer/job/endpoint/streaming) and trust the
  rows>0 gate for ingestion — by design (see `functional_findings.md`); use `--regrade` to
  re-grade a subset after any rubric edit.
- **`i12` ("the full corpus at `test-data`") mounts all ~1.7 GB** (every modality incl. the
  large vidorev3 corpora) — the heaviest query; expect it to time out without a prebuilt index.
- **Porting to a new environment:** repoint `--manifest`, `--corpus-data-root`, `--skill-src`,
  `--retriever-bin`, and set `$NVIDIA_API_KEY`. Nothing else is hard-coded.
