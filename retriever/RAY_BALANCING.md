# Ray Balancing Strategy

This document describes the default Ray Data balancing strategy used by
`retriever/src/retriever/ray_balance_dag.py`, why each test family exists, and
what to try next.

## Goal

Keep the default experiment set broad enough to find bottlenecks, but small
enough to run repeatedly across multiple machines. The default matrix is now
designed to stay under 1,000 variants.

## Design Approach

The default matrix uses a practical DOE-style approach:

1. **Baseline run**  
   A stable reference point used for quick comparisons.
2. **One-factor-at-a-time (OFAT) sweeps**  
   Change one knob while keeping others at baseline to isolate sensitivity.
3. **Targeted interaction sweeps**  
   Test only high-value parameter interactions where coupling is expected.

This avoids a full Cartesian product over all knobs (which grows to millions of
runs and is usually not actionable).

## Default Matrix Definition

The script’s default matrix includes the following families.

### A) Baseline

- Single baseline config with balanced CPU/GPU and midrange batch sizes.
- Purpose: anchor for all deltas and detect regressions quickly.

### B) OFAT Sweeps

- `pdf_workers`: `[4, 8, 12, 16]`
- `pdf_num_cpus`: `[1.0, 2.0, 3.0, 4.0]`
- `pdf_split_bs`: `[1, 4, 8]`
- `pdf_bs`: `[8, 16, 24, 32]`
- `page_elements_bs`: `[8, 16, 24, 32]`
- `page_elements_workers`: `[1, 2, 3]`
- `ocr_workers`: `[1, 2, 3]`
- `ocr_bs`: `[8, 16, 24, 32]`
- `embed_workers`: `[1, 2, 3]`
- `embed_bs`: `[128, 256, 512, 768]`
- `page_elements_cpus_per_actor`: `[1.0, 2.0, 4.0]`
- `ocr_cpus_per_actor`: `[1.0, 2.0, 4.0]`
- `embed_cpus_per_actor`: `[1.0, 2.0, 4.0]`
- `gpu_page_elements`: `[0.25, 0.5, 0.75]`
- `gpu_ocr`: `[0.75, 1.0, 1.25]`
- `gpu_embed`: `[0.25, 0.5, 0.75]`

Why this matters:

- Identifies which knobs are low-impact (can be fixed) vs high-impact (worth
  deeper search).
- Narrows the search space before trying interactions.

### C) Targeted Interaction Grids

1. **OCR throughput coupling**
   - `ocr_bs x ocr_workers x gpu_ocr`
2. **Embedding throughput coupling**
   - `embed_bs x embed_workers x gpu_embed`
3. **Page-elements throughput coupling**
   - `page_elements_bs x page_elements_workers x gpu_page_elements`
4. **CPU extraction balance**
   - `pdf_workers x pdf_num_cpus x pdf_bs`
5. **Actor CPU pressure**
   - `page_elements_cpus_per_actor x ocr_cpus_per_actor x embed_cpus_per_actor`
6. **Pipeline batch-shape interaction**
   - `pdf_bs x ocr_bs x embed_bs`

Why this matters:

- These are the pairs/triples most likely to create backpressure or starvation.
- Captures non-linear behavior without exploding matrix size.

## What Has Been Tried So Far

- Full/fat sweeps were tested early and found to be too large operationally.
- Matrix generation now deduplicates repeated variants and focuses on high-signal
  combinations.
- Row-range sharding support (`--row-start`, `--row-end`) is used for distributed
  execution across machines.

## How to Generate and Run

Generate matrix CSV only:

```bash
python retriever/src/retriever/ray_balance_dag.py \
  --input-dir /path/to/pdfs \
  --write-default-matrix-csv retriever/ray_balance_variants.csv \
  --exit-after-writing-matrix
```

Run a shard:

```bash
python retriever/src/retriever/ray_balance_dag.py \
  --input-dir /path/to/pdfs \
  --matrix-csv retriever/ray_balance_variants.csv \
  --row-start 1 \
  --row-end 200 \
  --output-csv retriever/ray_balance_results_001_200.csv
```

## Recommended Next Experiments

1. **Adaptive second-pass search**
   - Keep top 10-20% by throughput and run local neighborhood sweeps.
2. **Constraint-aware optimization**
   - Add objective penalties for GPU OOM, high object-store pressure, or low
     recall to avoid fragile winners.
3. **Dataset-stratified tests**
   - Split small/medium/large PDFs and optimize per segment; mixed corpora often
     hide better settings.
4. **Stability runs**
   - Re-run top candidates 3-5 times and compare variance, not just best mean.
5. **Multi-objective scoring**
   - Rank by weighted score: throughput, recall@k, and cost (GPU-hours).
