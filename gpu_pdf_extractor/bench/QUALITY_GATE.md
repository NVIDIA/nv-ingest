# Phase-0 Quality Gate (parity co-gate)

The CUDA rasterizer cannot be bit-exact with PDFium's AGG rasterizer (different anti-aliasing,
font hinting, subpixel rounding). So "correctness" is gated on two levels:

## 1. Raster parity (fast, local) — `bench/parity.py`
Per-page metrics between the candidate backend's raster and the pypdfium baseline raster
(same page, same scale): `exact_match_frac`, `mae`, `max_abs_diff`, `psnr_db`, `ssim`.

Provisional thresholds to flag a page as a parity regression (tune in P2 against real data):
- `ssim < 0.98` OR `psnr_db < 35` → flag for review.
These are perceptual proxies, intentionally loose, because downstream models are tolerant.

## 2. Downstream recall (authoritative) — agent_eval hook
The real gate is "does the swap change retrieval/extraction quality?" measured by the existing
agent_eval harness (see memory `agent-eval-harness-plan`). Wiring:

1. Run the agent_eval functional + vidore recall suite with `NEMO_PDF_BACKEND=pypdfium` → baseline scores.
2. Re-run with `NEMO_PDF_BACKEND=gpu` → candidate scores.
3. Compare per-metric (recall@k, nDCG, table/chart extraction accuracy).
   Pass condition: no statistically significant regression (agreed tolerance, e.g. ≤0.5 pt abs
   on primary recall metric) on any bucket.

This file is the contract; the actual agent_eval invocation is run in P2/P4 once the GPU backend
exists (it requires the NIM/YOLOX services that are not stood up in P0). P0 ships only the
local raster-parity tool (#1) plus this documented hook (#2).
