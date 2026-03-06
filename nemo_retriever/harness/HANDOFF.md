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
  - `financebench` wiring is included for `pdf_only` matching with `financebench_json` adapter, with recall disabled by default until query path is set.

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
```

Sweep:

```bash
retriever harness sweep --runs-config nemo_retriever/harness/nightly_config.yaml
```

Nightly:

```bash
retriever harness nightly --runs-config nemo_retriever/harness/nightly_config.yaml
retriever harness nightly --dry-run
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

## Current Validation Status

Harness-focused tests pass:

```bash
pytest -q nemo_retriever/tests/test_harness_parsers.py \
  nemo_retriever/tests/test_harness_config.py \
  nemo_retriever/tests/test_harness_run.py
```

## Recommended Next Iterations

### P0 (next)

- Completed in PR 1: add run-level metadata fields useful for long-term tracking:
  - `host`, `gpu_count`, `cuda_driver`, `ray_version`, `python_version`.
- Add optional run tag support (for example `--tag nightly`, `--tag candidate`) into `results.json`.
- Add one command to print a compact table from a session (`session_summary.json`) for quick review.

### P1

- Add a stable compare utility for two sessions (delta pages/sec and recall deltas by run name).
- Add preset inheritance or scaling helper to reduce duplicated numeric tuning in YAML.
- Add an artifact retention helper (manual command) to prune old sessions by age/size.

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
