# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``retriever eval`` Typer subcommands.

All heavy imports (litellm, evaluation modules) are deferred to inside
command bodies so that ``pip install nemo-retriever`` (without ``[llm]``)
does not break the CLI at import time.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(help="QA evaluation utilities.")


# ---------------------------------------------------------------------------
# Display helpers (originally from the now-deleted run_qa_eval.py)
# ---------------------------------------------------------------------------


def _print_multi_tier_summary(eval_results: dict, total_queries: int) -> None:
    typer.echo("\n" + "=" * 60)
    typer.echo("Multi-Tier Results")
    typer.echo("=" * 60)

    tier1 = eval_results.get("tier1_retrieval", {})
    aic_rate = tier1.get("answer_in_context_rate", 0)
    aic_count = tier1.get("answer_in_context_count", 0)
    aic_total = tier1.get("total", total_queries)
    typer.echo("\nTier 1 - Retrieval Quality:")
    typer.echo(f"  Answer-in-Context rate:  {aic_rate:.1%} ({aic_count}/{aic_total})")

    tier2 = eval_results.get("tier2_programmatic", {})
    if tier2:
        typer.echo("\nTier 2 - Programmatic Answer Quality:")
        for name, metrics in tier2.items():
            f1 = metrics.get("mean_token_f1", 0)
            typer.echo(f"  {name:20s} token_f1={f1:.3f}")

    by_model = eval_results.get("by_model", {})
    if by_model:
        typer.echo("\nTier 3 - LLM Judge:")
        for name, stats in by_model.items():
            ms = stats.get("mean_score", 0)
            ml = stats.get("mean_latency_s", 0)
            sc = stats.get("scored_count", 0)
            ec = stats.get("error_count", 0)
            typer.echo(f"  {name:20s} mean={ms:.3f} (0-1)  latency={ml:.1f}s  scored={sc}  errors={ec}")
            dist = stats.get("score_distribution", {})
            if dist:
                typer.echo(f"  {'':20s} dist: " + "  ".join(f"{k}:{v}" for k, v in sorted(dist.items())))

    fb = eval_results.get("failure_breakdown", {})
    if fb:
        typer.echo("\nFailure Breakdown:")
        for name, counts in fb.items():
            total = sum(counts.values())
            parts = "  ".join(f"{k}:{v}({v * 100 / total:.1f}%)" for k, v in sorted(counts.items()))
            typer.echo(f"  {name:20s} {parts}")

    typer.echo("=" * 60)


def _print_errors(eval_results: dict) -> None:
    per_query = eval_results.get("per_query", [])
    errors_found = False
    for i, qr in enumerate(per_query):
        query_text = qr.get("query", "")[:60]
        for model_name, gen in qr.get("generations", {}).items():
            if gen.get("error"):
                if not errors_found:
                    typer.echo("\n--- Generation errors ---")
                    errors_found = True
                typer.echo(f"  [query {i}] {query_text!r}")
                typer.echo(f"    model={model_name}  error={gen['error']}")
        for model_name, jdg in qr.get("judgements", {}).items():
            if jdg.get("error"):
                if not errors_found:
                    typer.echo("\n--- Judge errors ---")
                    errors_found = True
                typer.echo(f"  [query {i}] {query_text!r}")
                typer.echo(f"    model={model_name}  error={jdg['error']}")
    if not errors_found and per_query:
        typer.echo("\nNo per-query errors.")


def _on_run(result: dict, total_queries: int) -> None:
    if result["status"] == "PASS":
        _print_multi_tier_summary(result["eval_results"], total_queries)
        _print_errors(result["eval_results"])
    else:
        typer.echo(f"\nERROR: {result.get('error', 'unknown')}", err=True)


# ---------------------------------------------------------------------------
# Env-var config builder (replaces the now-deleted _main_env)
# ---------------------------------------------------------------------------

_ENV_DEFAULTS = {
    "GEN_MODEL": "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
    "GEN_MODEL_NAME": "generator",
    "GEN_TEMPERATURE": "0.0",
    "JUDGE_MODEL": "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
    "JUDGE_TEMPERATURE": "0.1",
    "JUDGE_MAX_TOKENS": "4096",
    "QA_TOP_K": "5",
    "QA_MAX_WORKERS": "4",
    "QA_LIMIT": "0",
    "MIN_COVERAGE": "0.0",
}


def _build_env_config() -> tuple[dict, str, str, str, float]:
    """Synthesise an eval-sweep config dict from environment variables.

    Returns ``(config, qa_dataset, ground_truth_dir, results_dir, min_coverage)``.
    """
    retrieval_file = os.environ.get("RETRIEVAL_FILE", "")
    lancedb_uri = os.environ.get("LANCEDB_URI", "")
    if not retrieval_file and not lancedb_uri:
        typer.echo(
            "ERROR: set RETRIEVAL_FILE (file mode) or LANCEDB_URI (lancedb mode) with --from-env",
            err=True,
        )
        raise typer.Exit(code=1)

    qa_dataset = os.environ.get("QA_DATASET", "")
    if not qa_dataset:
        typer.echo("ERROR: QA_DATASET environment variable is required with --from-env", err=True)
        raise typer.Exit(code=1)

    ground_truth_dir = os.environ.get("GROUND_TRUTH_DIR", "data")
    qa_top_k = int(os.environ.get("QA_TOP_K", _ENV_DEFAULTS["QA_TOP_K"]))
    qa_max_workers = int(os.environ.get("QA_MAX_WORKERS", _ENV_DEFAULTS["QA_MAX_WORKERS"]))
    qa_limit = int(os.environ.get("QA_LIMIT", _ENV_DEFAULTS["QA_LIMIT"]))
    min_coverage = float(os.environ.get("MIN_COVERAGE", _ENV_DEFAULTS["MIN_COVERAGE"]))
    gen_temperature = float(os.environ.get("GEN_TEMPERATURE", _ENV_DEFAULTS["GEN_TEMPERATURE"]))

    judge_model = os.environ.get("JUDGE_MODEL", _ENV_DEFAULTS["JUDGE_MODEL"])
    judge_temperature = float(os.environ.get("JUDGE_TEMPERATURE", _ENV_DEFAULTS["JUDGE_TEMPERATURE"]))
    judge_max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", _ENV_DEFAULTS["JUDGE_MAX_TOKENS"]))
    judge_api_base = os.environ.get("JUDGE_API_BASE")
    gen_model = os.environ.get("GEN_MODEL", _ENV_DEFAULTS["GEN_MODEL"])
    gen_name = os.environ.get("GEN_MODEL_NAME", _ENV_DEFAULTS["GEN_MODEL_NAME"])
    gen_api_base = os.environ.get("GEN_API_BASE")

    fallback_key = os.environ.get("NVIDIA_API_KEY", "")
    gen_api_key = os.environ.get("GEN_API_KEY", "") or fallback_key
    judge_api_key = os.environ.get("JUDGE_API_KEY", "") or fallback_key

    gen_models_str = os.environ.get("GEN_MODELS", "")
    gen_model_pairs: list[tuple[str, str]] = []
    if gen_models_str:
        for entry in gen_models_str.split(","):
            entry = entry.strip()
            if ":" not in entry:
                typer.echo(f"ERROR: GEN_MODELS entry '{entry}' must be name:model", err=True)
                raise typer.Exit(code=1)
            n, m = entry.split(":", 1)
            gen_model_pairs.append((n.strip(), m.strip()))
    else:
        gen_model_pairs.append((gen_name, gen_model))

    models: dict[str, dict] = {}
    evaluations: list[dict] = []
    for n, m in gen_model_pairs:
        models[n] = {
            "model": m,
            "api_base": gen_api_base,
            "api_key": gen_api_key,
            "temperature": gen_temperature,
        }
        evaluations.append({"generator": n, "judge": "_judge", "runs": 1})

    models["_judge"] = {
        "model": judge_model,
        "api_base": judge_api_base,
        "api_key": judge_api_key,
        "temperature": judge_temperature,
        "max_tokens": judge_max_tokens,
    }

    results_dir = os.environ.get(
        "RESULTS_DIR",
        os.path.dirname(os.environ.get("OUTPUT_FILE", "")) or "data/test_retrieval",
    )

    if retrieval_file:
        retrieval_block: dict[str, str | None] = {"type": "file", "file_path": retrieval_file}
    else:
        retrieval_block = {
            "type": "lancedb",
            "lancedb_uri": lancedb_uri,
            "lancedb_table": os.environ.get("LANCEDB_TABLE", "nemo-retriever"),
            "embedder": os.environ.get("EMBEDDER", "nvidia/llama-nemotron-embed-1b-v2"),
            "save_path": os.environ.get("RETRIEVAL_SAVE_PATH"),
        }

    config = {
        "execution": {
            "runs": 1,
            "top_k": qa_top_k,
            "max_workers": qa_max_workers,
            "limit": qa_limit,
            "min_coverage": min_coverage,
        },
        "dataset": {"source": qa_dataset, "ground_truth_dir": ground_truth_dir},
        "retrieval": retrieval_block,
        "models": models,
        "evaluations": evaluations,
        "output": {"results_dir": results_dir},
    }
    return config, qa_dataset, ground_truth_dir, results_dir, min_coverage


def run_qa_sweep_from_config_dict(cfg: dict) -> int:
    """Run a QA evaluation sweep from a loaded config dict (0 = success, 1 = failure).

    Used by ``retriever eval run`` so the LLM sweep stays in one implementation.
    """
    from nemo_retriever.tools.evaluation.ground_truth import get_qa_dataset_loader
    from nemo_retriever.tools.evaluation.retrievers import FileRetriever
    from nemo_retriever.tools.evaluation.runner import run_eval_sweep

    if os.environ.get("LITELLM_DEBUG", "0").strip() in ("1", "true", "yes"):
        import litellm

        litellm._turn_on_debug()

    execution = cfg.get("execution", {})
    dataset_cfg = cfg.get("dataset", {})
    retrieval_cfg = cfg.get("retrieval", {})
    output_cfg = cfg.get("output", {})

    qa_dataset = dataset_cfg.get("source", "")
    if not qa_dataset:
        typer.echo("ERROR: dataset.source is required in config", err=True)
        return 1

    results_dir = output_cfg.get("results_dir", "data/test_retrieval")
    qa_limit = execution.get("limit", 0)
    min_coverage = execution.get("min_coverage", 0.0)

    loader = get_qa_dataset_loader(qa_dataset)
    ground_truth_dir = dataset_cfg.get("ground_truth_dir", "data")
    qa_pairs = loader(data_dir=ground_truth_dir)
    typer.echo(f"Loaded {len(qa_pairs)} Q&A pairs from '{qa_dataset}'")

    if qa_limit and qa_limit > 0:
        qa_pairs = qa_pairs[:qa_limit]
        typer.echo(f"limit={qa_limit}: evaluating first {len(qa_pairs)} pairs")

    retrieval_type = retrieval_cfg.get("type", "file")
    if retrieval_type == "lancedb":
        page_index_path = retrieval_cfg.get("page_index")
        page_idx = None
        if page_index_path:
            with open(page_index_path, encoding="utf-8") as f:
                page_idx = json.load(f)
            typer.echo(f"Loaded page index: {len(page_idx)} documents")

        save_path = retrieval_cfg.get("save_path")
        retriever = FileRetriever.from_lancedb(
            qa_pairs=qa_pairs,
            lancedb_uri=retrieval_cfg.get("lancedb_uri", "lancedb"),
            lancedb_table=retrieval_cfg.get("lancedb_table", "nemo-retriever"),
            embedder=retrieval_cfg.get("embedder", "nvidia/llama-nemotron-embed-1b-v2"),
            top_k=execution.get("top_k", 5),
            page_index=page_idx,
            save_path=save_path,
        )
        typer.echo("Built retriever from LanceDB (in-memory)")
    else:
        retrieval_file = retrieval_cfg.get("file_path", "")
        if not retrieval_file:
            typer.echo("ERROR: retrieval.file_path is required when type='file'", err=True)
            return 1
        retriever = FileRetriever(file_path=retrieval_file)

    coverage = retriever.check_coverage(qa_pairs)
    typer.echo(f"Coverage: {coverage:.1%}")
    if coverage < min_coverage:
        typer.echo(
            f"ERROR: retrieval covers only {coverage:.1%} of queries " f"(min_coverage={min_coverage:.0%}). Aborting.",
            err=True,
        )
        return 1

    results = run_eval_sweep(
        cfg,
        qa_pairs,
        results_dir,
        retriever=retriever,
        on_run_complete=_on_run,
    )

    passed = sum(1 for r in results if r["status"] == "PASS")
    typer.echo(f"\nSweep complete: {passed}/{len(results)} passed")
    typer.echo(f"Coverage: {coverage:.1%}")
    for r in results:
        typer.echo(f"  {r['status']}: {r['label']} -> {r.get('output_path', r.get('error', ''))}")

    return 0 if passed == len(results) else 1


@app.command("run")
def run_cmd(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to eval_sweep.yaml or .json config file.",
        dir_okay=False,
    ),
    from_env: bool = typer.Option(
        False,
        "--from-env",
        help="Build configuration from environment variables (Docker/CI mode). "
        "Requires RETRIEVAL_FILE and QA_DATASET; see retriever eval run --help.",
    ),
) -> None:
    """Run a QA evaluation sweep.

    Supply either --config (YAML/JSON file) or --from-env (reads
    RETRIEVAL_FILE, QA_DATASET, GEN_MODEL, JUDGE_MODEL, etc. from
    the environment).
    """
    if not config and not from_env:
        typer.echo("ERROR: supply --config <path> or --from-env", err=True)
        raise typer.Exit(code=1)
    if config and from_env:
        typer.echo("ERROR: --config and --from-env are mutually exclusive", err=True)
        raise typer.Exit(code=1)

    if config:
        from nemo_retriever.tools.evaluation.config import load_eval_config

        cfg = load_eval_config(str(config))
    else:
        cfg, *_ = _build_env_config()

    code = run_qa_sweep_from_config_dict(cfg)
    if code != 0:
        raise typer.Exit(code=code)


@app.command("export")
def export_cmd(
    lancedb_uri: str = typer.Option("lancedb", "--lancedb-uri", help="Path to LanceDB directory."),
    lancedb_table: str = typer.Option("nemo-retriever", "--lancedb-table", help="LanceDB table name."),
    query_csv: Path = typer.Option(
        ...,
        "--query-csv",
        help="CSV file with 'query' (and optionally 'answer') columns.",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(..., "--output", help="Output JSON path."),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks per query."),
    embedder: str = typer.Option(
        "nvidia/llama-nemotron-embed-1b-v2",
        "--embedder",
        help="Embedding model name.",
    ),
    page_index: Path = typer.Option(
        None,
        "--page-index",
        help="Page markdown index JSON (enables full-page mode).",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Export LanceDB retrieval results to FileRetriever JSON."""
    import csv as csv_mod

    from nemo_retriever.common.io.export import export_retrieval_json

    queries: list[dict] = []
    with open(query_csv, newline="", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            q = row.get("query", "").strip()
            if q:
                queries.append(row)
    typer.echo(f"Loaded {len(queries)} queries from {query_csv}")

    page_idx = None
    if page_index is not None:
        with open(page_index, encoding="utf-8") as f:
            page_idx = json.load(f)
        total_pages = sum(len(p) for p in page_idx.values())
        typer.echo(f"Page index: {len(page_idx)} documents, {total_pages} pages")

    t0 = time.monotonic()
    result = export_retrieval_json(
        lancedb_uri=str(lancedb_uri),
        lancedb_table=lancedb_table,
        queries=queries,
        output_path=str(output),
        top_k=top_k,
        embedder=embedder,
        page_index=page_idx,
    )
    elapsed = time.monotonic() - t0

    all_results = result.get("queries", {})
    typer.echo(f"Exported {len(all_results)} queries in {elapsed:.1f}s -> {output}")


@app.command("build-page-index")
def build_page_index_cmd(
    parquet_dir: Path = typer.Option(
        ...,
        "--parquet-dir",
        help="Directory containing extraction Parquet files.",
        exists=True,
        file_okay=False,
    ),
    output: Path = typer.Option(..., "--output", help="Output JSON path for the page index."),
) -> None:
    """Build a page-level markdown index from extraction Parquets."""
    from nemo_retriever.common.io.markdown import build_page_index

    typer.echo(f"Parquet dir: {parquet_dir}")
    typer.echo(f"Output:      {output}")

    t0 = time.monotonic()
    index, failures = build_page_index(parquet_dir=str(parquet_dir))
    elapsed = time.monotonic() - t0

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    total_pages = sum(len(pages) for pages in index.values())
    size_mb = output.stat().st_size / 1024 / 1024
    typer.echo(f"\nBuilt index in {elapsed:.1f}s")
    typer.echo(f"  Documents: {len(index)}")
    typer.echo(f"  Pages:     {total_pages}")
    typer.echo(f"  File size: {size_mb:.1f} MB")
    if failures:
        typer.echo(f"  Failures:  {len(failures)} documents failed rendering")
        for source_id, error in list(failures.items())[:10]:
            typer.echo(f"    {source_id}: {error}")
