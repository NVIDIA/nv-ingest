# `retriever skill-eval` — benchmarking the `/nemo-retriever` skill on ViDoRe v3

`retriever skill-eval run` measures whether wiring the **`/nemo-retriever`** Claude Code skill into an agent improves retrieval and answer quality over a folder of PDFs, compared to a baseline agent that has neither the `retriever` CLI nor the skill available.

Each invocation runs the same set of questions through Claude Code under three conditions:

| Condition           | Retriever CLI on `$PATH` | Skill loaded into `.claude/` | Prompt style                       |
|---------------------|--------------------------|------------------------------|------------------------------------|
| `c1_base`           | No (shimmed + denied)    | No                           | Natural-language ("Set up search…")|
| `c2_retriever`      | Yes                      | Yes                          | Natural-language                   |
| `c3_retriever_skill`| Yes                      | Yes                          | Explicit `/nemo-retriever …` slash |

This README assumes you are targeting the **ViDoRe v3** corpus and have a copy of the per-domain PDF tree on disk (e.g. on NVIDIA infra at `/datasets/nv-ingest/vidore_v3_corpus_pdf/`, or a private mirror at any path of your choice).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Inputs at a glance](#inputs-at-a-glance)
- [1. Make the PDF tree reachable](#1-make-the-pdf-tree-reachable)
- [2. Supply an agent-eval manifest](#2-supply-an-agent-eval-manifest)
- [3. Author your `skill_eval.yaml`](#3-author-your-skill_evalyaml)
- [4. Run the benchmark](#4-run-the-benchmark)
- [CLI reference](#cli-reference)
- [Output layout](#output-layout)
- [Interpreting the summary](#interpreting-the-summary)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **`retriever` CLI** — install the project (`uv pip install -e ./nemo_retriever`) so `retriever skill-eval` is on `$PATH`.
- **`claude` CLI** — the Claude Code binary must be on `$PATH`. The runner exits with code `2` if `shutil.which("claude")` returns `None`.
- **A Claude account / API access** — `claude --print` will negotiate auth on first use.
- **Claude Code autorun / permission access** — the runner launches non-interactive Claude Code subprocesses with `--permission-mode bypassPermissions` and, for `c2_retriever` / `c3_retriever_skill`, `--allow-dangerously-skip-permissions`. Make sure your environment is allowed to run those commands before starting a sweep, and run a small `claude --print` smoke test first so any first-use auth or permission prompts are already resolved.
- **Disk** — each `(condition, domain)` builds a scratch workdir under `/tmp/skill_eval/` containing a `pdfs/` symlink farm, a `.claude/` sandbox, and any retrieval artifacts the agent creates (e.g. `lancedb/`). The workdir is deleted after the session completes, so only one LanceDB is on disk at a time.
- **(Optional) `NVIDIA_API_KEY`** — if set, the LLM-as-judge scores each `final_answer` against the manifest's ground-truth `answer` on a 0–5 scale. Unset means judging is skipped silently with a console note; recall numbers are still produced.

---

## Inputs at a glance

`skill-eval` needs three things you must supply:

1. A folder of **PDFs per domain** (e.g. `vidore_v3_finance_en/*.pdf`).
2. An **agent-eval manifest** (JSON list) describing the queries, paraphrased prompts, ground-truth pages, and ground-truth answers — see [§2](#2-supply-an-agent-eval-manifest).
3. A **`skill_eval.yaml`** config binding the manifest to the PDF directories — see [§3](#3-author-your-skill_evalyaml).

Everything else (model, budget, timeout, conditions, judge endpoint) has working defaults in the packaged config at `src/nemo_retriever/skill_eval/configs/skill_eval.yaml`.

---

## 1. Make the PDF tree reachable

ViDoRe v3 is split per-domain. The eight domains the harness recognises are:

```
vidore_v3_computer_science
vidore_v3_energy
vidore_v3_finance_en
vidore_v3_finance_fr
vidore_v3_hr
vidore_v3_industrial
vidore_v3_pharmaceuticals
vidore_v3_physics
```

Each one resolves to a directory containing only the relevant PDFs. On NVIDIA infra those live at `/datasets/nv-ingest/vidore_v3_corpus_pdf/<domain>`. If your copy lives elsewhere, substitute that path everywhere `<VIDORE_ROOT>` appears below; e.g. `<VIDORE_ROOT> = /raid/datasets/vidore_v3_corpus_pdf`.

The runner does **not** copy the PDFs — it builds per-trial symlinks into `<workdir>/pdfs/`. The PDF roots can stay on a read-only mount.

> **Tip:** if your `domain` strings in the manifest are bare (e.g. `finance_en` rather than `vidore_v3_finance_en`), the `pdf_dirs` keys in the config must match exactly what's in the manifest, not what's on the filesystem. See [§3](#3-author-your-skill_evalyaml).

---

## 2. Supply an agent-eval manifest

The manifest is a **JSON list**; each item describes one query. It is produced by an upstream SDG pipeline; the skill-eval loader is dataset-agnostic and only enforces the schema described in `dataset.py:load_eval_manifest`. The minimum required keys per entry are:

| Field                                       | Type                 | Purpose                                                                                       |
|---------------------------------------------|----------------------|-----------------------------------------------------------------------------------------------|
| `original_query`                            | string               | The raw user question. Used by the `c3_retriever_skill` slash-command prompt.                 |
| `sdg_prompt_candidates.candidates`          | list of `{variant_id, prompt}` | Paraphrased prompt variants. The runner uses the one matching `sdg_prompt_validation.selected_variant_id`, falling back to the first. |
| `sdg_prompt_validation.selected_variant_id` | int (optional)       | Chosen variant.                                                                               |
| `relevant_pages`                            | list of `{doc_id, page_number_in_doc, score}` | Ground-truth pages. `doc_id` is the PDF basename without `.pdf`; `page_number_in_doc` is 0-indexed. |
| `answer`                                    | string               | Ground-truth answer, used by the LLM judge.                                                   |
| `domain`                                    | string               | Joins the entry to a `pdf_dirs` key in the config.                                            |
| `prompt_taxonomy.domain_label`              | string               | Human-readable domain name injected into the setup-turn prompt (e.g. "energy industry reports"). |
| `primary_eval_id`                           | string (optional)    | Stable per-query id (else `eval_base_id`, else 1-indexed position).                           |

The newer scenario-format keys (`scenario_prompt_candidates`, `scenario_prompt_validation`) are accepted as aliases.

Entries with `prompt_export_status` not in `(None, "exported")` are skipped, as are entries with no usable paraphrased prompt.

**Example entry** (one item from the manifest list):

```json
{
  "primary_eval_id": "vidore_v3_finance_en:42:variant-1",
  "domain": "vidore_v3_finance_en",
  "prompt_taxonomy": { "domain_label": "English-language corporate finance filings" },
  "original_query": "What was Acme Corp's free cash flow in FY2024?",
  "sdg_prompt_candidates": {
    "candidates": [
      { "variant_id": 1, "prompt": "Look at the PDFs at ./pdfs/ and tell me Acme Corp's FY2024 free cash flow." }
    ]
  },
  "sdg_prompt_validation": { "selected_variant_id": 1 },
  "relevant_pages": [
    { "doc_id": "Acme_10K_2024", "page_number_in_doc": 47, "score": 1 }
  ],
  "answer": "$3.2B (per the FY2024 cash flow statement)."
}
```

Note any path references inside paraphrased prompts (e.g. `"the PDFs at test-data/vidore_v3/.../pdfs/"`) — they may need rewriting via `testdata_prefixes`; see [§3](#3-author-your-skill_evalyaml).

---

## 3. Author your `skill_eval.yaml`

Copy the packaged config next to your dataset checkout and edit it:

```bash
cp nemo_retriever/src/nemo_retriever/skill_eval/configs/skill_eval.yaml \
   ~/datasets/vidore_v3/skill_eval.yaml
```

A complete ViDoRe v3 config looks like:

```yaml
# ~/datasets/vidore_v3/skill_eval.yaml

# Absolute path to your agent-eval manifest (JSON list).
eval_manifest_path: ~/datasets/vidore_v3/agent_eval_manifest.json

# Per-domain PDF roots. KEY = manifest "domain" field; VALUE = directory of PDFs.
# Substitute <VIDORE_ROOT> with wherever your ViDoRe v3 corpus lives.
pdf_dirs:
  vidore_v3_computer_science:  <VIDORE_ROOT>/vidore_v3_computer_science
  vidore_v3_energy:            <VIDORE_ROOT>/vidore_v3_energy
  vidore_v3_finance_en:        <VIDORE_ROOT>/vidore_v3_finance_en
  vidore_v3_finance_fr:        <VIDORE_ROOT>/vidore_v3_finance_fr
  vidore_v3_hr:                <VIDORE_ROOT>/vidore_v3_hr
  vidore_v3_industrial:        <VIDORE_ROOT>/vidore_v3_industrial
  vidore_v3_pharmaceuticals:   <VIDORE_ROOT>/vidore_v3_pharmaceuticals
  vidore_v3_physics:           <VIDORE_ROOT>/vidore_v3_physics

# OPTIONAL — rewrite dataset-source path prefixes in paraphrased prompts to ./pdfs.
# Add one entry per prefix the manifest hard-codes.
testdata_prefixes:
  - test-data/vidore_v3/

# Agent + per-trial limits (defaults shown).
agent_model: claude-opus-4-7
per_trial_budget_usd: 5.0
per_trial_timeout_s: 600
per_trial_workdir_root: /tmp/skill_eval

# Conditions to run, in order. Each (condition, domain) workdir is deleted after
# it finishes, so only one LanceDB exists on disk at a time.
conditions:
  - c1_base
  - c2_retriever
  - c3_retriever_skill

# LLM-as-judge. Skipped silently if $NVIDIA_API_KEY is unset.
judge:
  enabled: true
  model: nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1
  api_base: https://integrate.api.nvidia.com/v1
  api_key_env: NVIDIA_API_KEY
```

**Things to double-check:**

- `pdf_dirs` keys must **exactly match** the `domain` field on each manifest entry. Mismatches cause the runner to exit with `pdf_dirs is missing an entry for domain '…'`.
- Each path under `pdf_dirs` must be a directory (not a glob). The runner symlinks every `*.pdf` inside.
- If your manifest references PDFs by paths like `test-data/vidore_v3/finance_en/pdfs/Acme.pdf`, add `test-data/vidore_v3/<domain>/pdfs/` (or the common prefix) to `testdata_prefixes` so the prompt text resolves to `./pdfs/Acme.pdf` inside the trial workdir.
- The single-path key `pdf_dir` is still honored as a fallback if you only have one domain.

---

## 4. Run the benchmark

Smoke-test on one domain first to validate config + manifest binding before paying for the whole sweep:

```bash
retriever skill-eval run \
  --config ~/datasets/vidore_v3/skill_eval.yaml \
  --domains vidore_v3_finance_en \
  --conditions c2_retriever
```

This runs one condition × one domain — one Claude session, with one setup turn followed by N query turns (one per manifest entry tagged `vidore_v3_finance_en`).

Once that succeeds, run the full sweep:

```bash
retriever skill-eval run --config ~/datasets/vidore_v3/skill_eval.yaml
```

Conditions and domains execute sequentially — three conditions × eight domains = 24 sessions in the default ViDoRe v3 setup. Each session is one Claude Code subprocess holding state across turns via `--resume <session-id>`.

The runner prints per-turn status, token usage, cost, and recall per `(condition, domain)`. Example:

```
Loaded 412 dataset entries.
Domains in this run: ['vidore_v3_computer_science', …, 'vidore_v3_physics'] (412 entries total)
Session dir: /raid/.../nemo_retriever/artifacts/skilleval_20260518_141200
Starting session for c1_base/vidore_v3_finance_en — setup + 52 query turns (pdfs=/datasets/.../vidore_v3_finance_en)
  turn 1 [vidore_v3_finance_en] setup: status=ok tokens(in/out/cache_r)=… cost=$0.041 retrieved=0
  turn 2 [vidore_v3_finance_en] entry_id=1 query_id=vidore_v3_finance_en:1:variant-1: status=ok … judge=4
  …
Recall for c1_base/vidore_v3_finance_en: recall@1=0.115  recall@5=0.327  recall@10=0.481
Cleaned up workdir for c1_base/vidore_v3_finance_en
```

---

## CLI reference

```
retriever skill-eval run [OPTIONS]
```

| Option              | Default                                              | Notes                                                                                          |
|---------------------|------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `--config PATH`     | packaged `skill_eval.yaml` (errors w/o `pdf_dirs`)   | The YAML described in [§3](#3-author-your-skill_evalyaml). Strongly recommend supplying.       |
| `--eval-manifest PATH` | `cfg.eval_manifest_path`                          | Overrides the config's manifest path.                                                          |
| `--conditions LIST` | `c1_base,c2_retriever,c3_retriever_skill`            | Comma-separated, in execution order. Unknown values exit with code `2`.                        |
| `--domains LIST`    | all domains present in the manifest                  | Comma-separated subset (e.g. `vidore_v3_finance_en,vidore_v3_finance_fr`). Unknowns exit with code `2`. |
| `--artifacts-root PATH` | `<repo>/nemo_retriever/artifacts/`               | Where the session directory is created.                                                        |

The CLI exits with code `2` for any configuration error (missing `claude` binary, missing `eval_manifest_path`, malformed `pdf_dirs`, unknown condition/domain, missing PDF directory).

---

## Output layout

Each run writes a timestamped session directory:

```
<artifacts-root>/skilleval_<timestamp>/
├── config.yaml                   # Snapshot of the resolved config
├── session_summary.json          # Machine-readable per-(condition, domain) metrics
├── session_summary.md            # Human-readable markdown report
└── trials/
    ├── c1_base/
    │   └── vidore_v3_finance_en/
    │       ├── c1_base_vidore_v3_finance_en_setup_t1.json
    │       ├── c1_base_vidore_v3_finance_en_e1_t2.json
    │       └── …
    ├── c2_retriever/
    │   └── vidore_v3_finance_en/…
    └── c3_retriever_skill/
        └── vidore_v3_finance_en/…
```

**Per-trial JSON** (`trials/<cond>/<domain>/<trial_id>.json`) is the `TrialResult` dataclass serialized: status, duration, token usage, cost, `final_answer`, `ranked_retrieved`, judge score, and `retriever_used_ever` / `skill_fired` diagnostics.

**Trial workdirs** under `per_trial_workdir_root` (`/tmp/skill_eval/` by default) are **deleted** after each `(condition, domain)` session finishes — only the session directory above survives. If you want to inspect a workdir mid-run (e.g. examine the agent's `lancedb/`), kill the run before the cleanup, or set a breakpoint.

---

## Interpreting the summary

`session_summary.md` contains, per condition (rolled up across domains) and per `(condition, domain)`:

- `success_rate` — fraction of turns that exited cleanly.
- `retr_used` — fraction of turns whose Claude Code transcript contains a Bash invocation of the `retriever` CLI. Should be near 0 for `c1_base` and near 1 for `c2/c3`.
- `recall@1 / @5 / @10` — macro-averaged recall@k over the `(doc_id, page_number)` pairs the agent wrote into `ranked_retrieved`. Comparable to `retriever harness` BEIR output.
- `judge` — mean LLM judge score on the 0–5 scale, with sample size. `—` when the judge was disabled or unreachable.
- `q_input / q_output / q_cache_read / q_cache_create` — mean per-query-turn token usage on the agent session (not the underlying retrieval pipeline's embedding/VLM calls — those aren't instrumented here).
- `q_cost` — mean per-query-turn USD cost.

A separate **"Setup turns"** table sums the one-time setup-turn cost across all domains for each condition. For `c2/c3` this captures the cost of running `retriever ingest ./pdfs/` over a domain; for `c1` it captures the cost of whatever ad-hoc scaffolding the agent invents (typically expensive and noisy).

The **"Diagnostics"** section reports `skill_fired_rate` for `c2/c3`: the fraction of turns where the agent invoked `retriever` within the first two turns (a proxy for "did the skill description auto-discover correctly").

---

## Troubleshooting

**`Error: \`claude\` CLI is not on PATH`** — install Claude Code and confirm `which claude` resolves before re-running.

**`config 'pdf_dirs' is missing an entry for domain '<X>'`** — your manifest contains a `domain` value that has no key in `pdf_dirs`. Either add the key, or use `--domains` to skip that subset.

**`PDF directory '…' for domain '…' does not exist or is not a directory`** — the value under `pdf_dirs.<domain>` was unset (`~` expansion failed, typo, etc.). Resolve the path manually with `ls "/your/configured/pdf_dirs/path"` and update the config.

**Judge prints `Judge disabled: $NVIDIA_API_KEY is not set` and exits cleanly** — that is by design. Recall and other metrics still land in the summary; only the `judge` column shows `—`. Export `NVIDIA_API_KEY` and re-run if you want the score.

**`c1_base` shows `retr_used` > 0** — the `_C1_BASH_DENY_PATTERNS` in `runner.py` are deny-globs against the assembled command line. If the agent invented a new path that those globs don't catch, the call goes through. File an issue with the offending command from the trial JSON's session-log path and extend the list.

**Per-domain run times look too long** — drop `--conditions c2_retriever,c3_retriever_skill` for a quick recall-only sweep against the skill (skipping `c1_base`), or use `--domains` to subset. Each condition × domain is independent; you can re-run any subset and the session directories don't collide (each has its own timestamped name).

**Agent failed to write `./output.json`** — the per-trial JSON will have `status="extraction_failed"` and `extraction_method` in `("missing", "invalid_json")`. The Claude Code session log path (under `~/.claude/projects/`) is reconstructable from the workdir, but the workdir has been deleted — re-run that single trial with `--conditions <cond> --domains <domain>` to capture the transcript fresh.
