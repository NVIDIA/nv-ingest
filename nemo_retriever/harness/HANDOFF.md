<!-- SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Nemo Retriever Harness Handoff

This document is a planning handoff for ongoing `nemo_retriever` harness iterations.
It captures what exists now, what was intentionally chosen, and what to iterate next.

## Current Scope and Intent

- Harness is standalone under `nemo_retriever` (not based on `tools/harness`).
- It wraps `nemo_retriever.examples.batch_pipeline`.
- Primary use case is benchmark orchestration for local/cluster runs without Docker orchestration.
- Vector DB is LanceDB only.
- Recall gating is supported and enforced by config (`recall_required`).

## Key Files

- `nemo_retriever/src/nemo_retriever/harness/run.py`
  - CLI run/sweep/nightly orchestration, subprocess execution, metrics extraction, artifact writes.
- `nemo_retriever/src/nemo_retriever/harness/config.py`
  - YAML + CLI/env merge logic and `HarnessConfig`.
- `nemo_retriever/src/nemo_retriever/harness/parsers.py`
  - Stream parsing for ingest/throughput/recall metrics.
- `nemo_retriever/src/nemo_retriever/harness/artifacts.py`
  - Artifact/session directory creation and session summary writing.
- `nemo_retriever/src/nemo_retriever/harness/recall_adapters.py`
  - Dataset-specific query normalization adapters for recall inputs.
- `nemo_retriever/harness/test_configs.yaml`
  - Active defaults, presets, dataset presets.
- `nemo_retriever/harness/nightly_config.yaml`
  - Ordered run list for sweep/nightly.

## Current Config Defaults

- Active default dataset: `jp20` (recall-required workflow).
- `bo20` remains ingestion-only (`query_csv: null`, `recall_required: false`).
- Two main presets are available:
  - `single_gpu`
  - `dgx_8gpu`
- Adapter-capable datasets:
  - `earnings` uses `recall_adapter: page_plus_one` (`page` -> `pdf_page` conversion).
  - `bo10k` wiring is included (adapter + mode), with recall disabled by default until query path is set.
  - `financebench` uses `pdf_only` matching with `financebench_json` and defaults to `data/financebench_train.json`.

## Current CLI Usage

From repo root:

```bash
source ~/setup_env.sh
source .retriever/bin/activate
uv pip install -e ./nemo_retriever
```

Single run:

```bash
retriever harness run --dataset jp20 --preset single_gpu
retriever harness run --dataset jp20 --preset single_gpu --tag nightly --tag candidate
```

Sweep:

```bash
retriever harness sweep --runs-config nemo_retriever/harness/nightly_config.yaml
```

Nightly:

```bash
retriever harness nightly --runs-config nemo_retriever/harness/nightly_config.yaml
retriever harness nightly --dry-run
retriever harness nightly --runs-config nemo_retriever/harness/nightly_config.yaml --tag nightly
```

Session inspection:

```bash
retriever harness summary nemo_retriever/artifacts/nightly_20260305_010203_UTC
retriever harness compare \
  nemo_retriever/artifacts/nightly_20260305_010203_UTC \
  nemo_retriever/artifacts/nightly_20260306_010204_UTC
```

## Artifact Contract (Current)

Per run:

- `results.json` (authoritative run record)
- `command.txt`
- `runtime_metrics/`
- `lancedb/`

Session-level:

- `session_summary.json` only

Notes:

- `detection_summary` is embedded into `results.json`.
- Standalone `detection_summary.json` is optional via `write_detection_file: true`.
- `sweep_results.json` was removed to avoid duplicated session outputs.

## Recent Cleanup Decisions

1. **Session naming cleaned**
   - Session run directories now use run name directly (for example `jp20_single`).
   - Removed redundant suffix style like `jp20_single_jp20`.

2. **Session output deduped**
   - Kept `session_summary.json`.
   - Removed `sweep_results.json` generation.

3. **TTY-backed subprocess retained**
   - Harness runs batch pipeline through a PTY so Ray progress remains rich/pretty by default.

## Known Behavior to Remember

- If a dataset has no ground-truth CSV and `recall_required` is false, run can still pass on ingest metrics.
- If `recall_required` is true and recall metrics are missing, harness marks failure (`missing_recall_metrics`).
- LanceDB directories dominate artifact size; JSON overhead is small.
- Relative `query_csv` values now resolve from the config file directory first, then from repo root, so worktree/cwd changes do not break repo-configured datasets.
- Dataset presets under `/datasets/nv-ingest/...` fall back to `/raid/$USER/...` when the dataset is local to the current cluster user.

## Recent Progress (Mar 2026)

- Dataset-specific recall adapters are implemented and wired through harness config:
  - `page_plus_one` for earnings/bo10k-style `query,pdf,page` CSVs.
  - `financebench_json` for FinanceBench JSON query data (`query,expected_pdf` output).
- Match mode is configurable and exercised end-to-end:
  - `pdf_page` for page-level recall.
  - `pdf_only` for document-level recall.
- Full e2e runs have been validated in the new harness path:
  - `earnings` with adapter-based page normalization and recall metrics emitted.
  - `financebench` with JSON adapter + `pdf_only` matching and recall metrics emitted.
- Input file discovery now supports nested directories for all input types:
  - `pdf`, `txt`, `html`, and `doc` paths recurse via `**/*` patterns.
- Recall hit-check behavior was de-duplicated:
  - `batch_pipeline` now uses shared hit-check logic from `nemo_retriever.recall.core`.
- Adapter dispatch was simplified:
  - `recall_adapters.py` uses a small adapter registry instead of `if/elif` branching.
- CI robustness improvement:
  - `nv_ingest_api` import in recall core is lazy (inside NIM embed helper), so unit tests can import recall modules in minimal environments without `nv_ingest_api` installed.
- Config usability improvement:
  - Relative `query_csv` paths are stable across cwd/worktree runs.
  - Dataset presets can resolve from `/raid/$USER/...` when `/datasets/nv-ingest/...` is unavailable.
  - `financebench` now defaults to `data/financebench_train.json` with recall enabled.
- Session UX improvements:
  - Runs, sweeps, and nightly sessions accept repeatable `--tag` values persisted into artifacts.
  - `retriever harness summary` prints a compact table from `session_summary.json`.
- Comparison utility:
  - `retriever harness compare` prints pages/sec and recall deltas by run name for two sessions.

## Current Validation Status

Harness-focused tests pass:

```bash
pytest -q nemo_retriever/tests/test_harness_parsers.py \
  nemo_retriever/tests/test_harness_config.py \
  nemo_retriever/tests/test_harness_run.py \
  nemo_retriever/tests/test_harness_reporting.py \
  nemo_retriever/tests/test_harness_recall_adapters.py \
  nemo_retriever/tests/test_recall_core.py
```

## Recommended Next Iterations

### P0 (next)

- Completed in PR 1: add run-level metadata fields useful for long-term tracking:
  - `host`, `gpu_count`, `cuda_driver`, `ray_version`, `python_version`.

### P1

- Add preset inheritance or scaling helper to reduce duplicated numeric tuning in YAML.
- Add an artifact retention helper (manual command) to prune old sessions by age/size.
- Consider a single `recall_profile` abstraction (mapping to adapter + match mode) to reduce config coupling
  and avoid invalid combinations.

### P2

- Optional matrix expansion for runs config (keep explicit run list as default UX).
- Export-to-csv summary utility for spreadsheet and trend analysis.
- Optional stricter validation mode that fails on unknown config keys.

## Planning Checklist for New Iterations

When adding harness features, keep this order:

1. Update config schema/load logic (`config.py`) and defaults in `test_configs.yaml`.
2. Update run orchestration (`run.py`) and artifact payloads (`results.json`/`session_summary.json`).
3. Add/extend unit tests in `nemo_retriever/tests/test_harness_*.py`.
4. Update `nemo_retriever/README.md` harness section.
5. Run harness tests and at least one `--dry-run` CLI check.

## Suggested "Done" Criteria Per Iteration

- No regressions in harness test suite.
- Artifact schema changes are intentional and documented.
- README command examples still execute as written.
- Session output remains concise and non-duplicative.
- At least one real run or dry run validates the new behavior.
