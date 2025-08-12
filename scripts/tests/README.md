## RCI Integration tests and benchmarks (nv-ingest)

This directory contains lean, configurable end-to-end tests that run inside the main nv-ingest repo. The initial focus is the smallest Digital Corpora flow, renamed from "bo20" to "dc20" for professionalism. The underlying dataset path remains `/datasets/bo20` for now.

### Quickstart
- Default dataset: `/datasets/bo20`
- Default profiles: `retrieval,table-structure`
- Runtime API: HTTP on `localhost:7670`

Coming in small steps:
- A single runner `scripts/tests/run.py` to start services, wait for readiness, run a case, and collect minimal artifacts.
- One case to start: `dc20_e2e` (ported from `bo20_e2e.py`).

### Test catalog

| Name | What it validates | Profiles | Dataset | Notes |
|---|---|---|---|---|
| Digital Corpora (dc20) | Extract + Embed + vdb upload to Milvus + retrieval sanity | retrieval,table-structure | `/datasets/bo20` | Smallest E2E; mixed protocol as below |

Planned (from legacy scripts; not all will be ported):

| Name | What it validates | Profiles | Dataset | Status |
|---|---|---|---|---|
| multiformat_test | Mixed content extraction smoke | table-structure | repo sample | Planned |
| audio_ingest / audio_recall | Audio pipeline ingest/recall | audio | local audio | Planned |
| earnings_ingest / earnings_recall | Domain E2E ingest/recall | retrieval,table-structure | earnings | Planned |
| foundation_rag_ingest / recall | Foundation RAG flow | retrieval,table-structure | foundation | Planned |
| bo767 ingest/recall | Larger E2E scale | retrieval,table-structure | bo767 | Planned |

### Conventions
- Dataset path is configurable via `DATASET_DIR` (default `/datasets/bo20`).
- Collection naming will default to `dc20_<timestamp>` to avoid collisions (configurable).
- Artifacts will be written under `scripts/tests/artifacts/<timestamp>/` by default (configurable via `ARTIFACTS_DIR`).
- Spill directory defaults to `/tmp/spill` and is configurable via `SPILL_DIR`.

### Attach vs managed modes
- Managed: the runner starts compose services with `--build`, waits for readiness, runs a case, and optionally tears down.
- Attach: the runner just executes the case against already-running services; the user is responsible for profiles and health.


