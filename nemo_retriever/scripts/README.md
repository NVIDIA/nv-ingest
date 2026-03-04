# vLLM vs baseline embedding comparison scripts

Scripts for comparing **local HF (baseline)** and **vLLM offline** embedding on pre-embedded parquet: ingest time and recall. Use them to run a single comparison or a parameter sweep and plot results.

**Assumptions:** Run from the **repository root** (parent of `nemo_retriever/`) with the nemo_retriever env (e.g. `uv run`). The machine (or Ray cluster) must have **at least one GPU** for the embed service.

---

## 1. Single-run comparison (`vllm_embedding_comparison.py`)

Compares baseline (HF) and vLLM offline on a pre-embed parquet dir: starts a long-lived embed service per backend, warms it up, runs a **timed** ingest (model load excluded), then recall.

### Usage

```bash
# From repo root; unset RAY_ADDRESS so we start our own Ray cluster
unset RAY_ADDRESS

uv run python nemo_retriever/scripts/vllm_embedding_comparison.py compare-from-pre-embed \
  --pre-embed-dir /path/to/pre_embed_dir \
  --query-csv /path/to/query_gt.csv \
  --embed-model-path /path/to/llama-nemotron-embed-1b-v2/main
```

### Options (often used)

| Option | Description |
|--------|-------------|
| `--max-rows N` | Use only N rows from pre-embed (faster; e.g. 1000). |
| `--gpu-memory-utilization 0.55` | vLLM GPU memory fraction (default 0.55). |
| `--enforce-eager` | vLLM: disable CUDA graphs (slower, avoids some env issues). |
| `--sort-key COL` | Sort by column before limit so baseline and vLLM see the same rows. |
| `--output-csv FILE` | Append one row of metrics to CSV for later plotting. |

### Example (1K rows, with output for sweep)

```bash
unset RAY_ADDRESS
uv run python nemo_retriever/scripts/vllm_embedding_comparison.py compare-from-pre-embed \
  --pre-embed-dir /path/to/bo767_pre_embed \
  --query-csv data/bo767_query_gt.csv \
  --embed-model-path /path/to/llama-nemotron-embed-1b-v2/main \
  --max-rows 1000 \
  --output-csv comparison_sweep.csv
```

---

## 2. Sweep: grid of runs (`run_comparison_sweep.py`)

Runs `compare-from-pre-embed` over a grid of **gpu_memory_utilization** × **max_rows**, appending one CSV row per run. Useful for collecting data to plot ingest time vs scale and GPU util.

### Grid (defaults)

- **gpu_utils:** 0.4, 0.5, 0.6, 0.7, 0.8  
- **max_rows:** 1000, 2000, 5000, 10000  

Override with `--gpu-utils 0.4,0.6,0.8` and `--max-rows-list 1000,5000`. To sweep **embed_batch_size** (e.g. 256, 512, 768) with an env that has **flashinfer-cubin**, see [embed_batch_size_sweep_setup.md](embed_batch_size_sweep_setup.md).

### FlashInfer / flashinfer-cubin

By default the sweep **requires** an environment with **flashinfer-cubin** (e.g. `uv sync --extra vllm-attention` in `nemo_retriever`). If it is not installed, the script exits with install instructions. Use `--no-require-flashinfer-cubin` to run anyway (e.g. to collect flashinfer_cubin=false rows).

`flashinfer_cubin` is **detected at runtime** and written to the CSV. To get both true/false in one CSV:

1. **With FlashInfer:** run the sweep with `flashinfer-cubin` installed (e.g. `[vllm-attention]` extra).  
2. **Without FlashInfer:** `uv pip uninstall flashinfer-cubin`, then run the **same** command and **same** `--output-csv` (with `--no-require-flashinfer-cubin`) so rows are appended.

### Usage

```bash
unset RAY_ADDRESS

uv run python nemo_retriever/scripts/run_comparison_sweep.py \
  --pre-embed-dir /path/to/pre_embed_dir \
  --query-csv /path/to/query_gt.csv \
  --embed-model-path /path/to/llama-nemotron-embed-1b-v2/main \
  --output-csv comparison_sweep.csv
```

### Tracking progress (log file)

```bash
unset RAY_ADDRESS
uv run python nemo_retriever/scripts/run_comparison_sweep.py \
  --pre-embed-dir /path/to/pre_embed_dir \
  --query-csv /path/to/query_gt.csv \
  --embed-model-path /path/to/llama-nemotron-embed-1b-v2/main \
  --output-csv comparison_sweep.csv \
  2>&1 | tee comparison_sweep.log
```

Then in another terminal: `tail -f comparison_sweep.log`.

---

## 3. Plotting (`plot_comparison_sweep.py`)

Reads the CSV produced by the sweep and writes figures (ingest time vs max_rows, ingest vs gpu_util, recall@10, optional FlashInfer impact).

### Usage

```bash
uv run python nemo_retriever/scripts/plot_comparison_sweep.py run \
  --input-csv comparison_sweep.csv \
  --output-dir ./comparison_plots
```

Requires **pandas** and **matplotlib** (usually already in the env).

---

## 4. Optional: Ray GPU check (`check_ray_gpu.py`)

Prints Ray cluster resources and actors that use or are waiting for GPU. Useful when the comparison is “stuck pending” (no GPU placement).

- To see the comparison’s actors, the comparison must use a **persistent** Ray cluster (`ray start --head --num-gpus=1`) and you must set **RAY_ADDRESS** for both the comparison and this script.  
- If the comparison is run **without** `--ray-address` (default), it uses `ray.init("local")` and no separate Ray server exists, so this script cannot attach.

```bash
RAY_ADDRESS=127.0.0.1:6379 uv run python nemo_retriever/scripts/check_ray_gpu.py
```

---

## Running on other machines

1. **Paths:** Replace `/path/to/pre_embed_dir`, `/path/to/query_gt.csv`, and `--embed-model-path` with paths valid on that machine.  
2. **Environment:** Use the same Python env as for development (e.g. `uv run` from repo root, or activate the venv and run `python nemo_retriever/scripts/...`).  
3. **RAY_ADDRESS:** Run `unset RAY_ADDRESS` before the comparison/sweep so the script starts its own Ray cluster; otherwise you may attach to another user’s cluster and see 0 GPUs.  
4. **GPU:** The embed service needs 1 GPU. If Ray reports 0 GPUs, the script will error; ensure the node has a GPU and Ray can see it (e.g. `ray start --head --num-gpus=1` if using a persistent cluster).

---

## Script summary

| Script | Purpose |
|--------|--------|
| `vllm_embedding_comparison.py` | Single compare-from-pre-embed run (baseline + vLLM); optional `--output-csv`. |
| `run_comparison_sweep.py` | Sweep over gpu_util × max_rows; appends to `--output-csv`. |
| `plot_comparison_sweep.py` | Read sweep CSV, write figures to `--output-dir`. |
| `check_ray_gpu.py` | Print Ray GPU/resources and GPU actors (for debugging). |
