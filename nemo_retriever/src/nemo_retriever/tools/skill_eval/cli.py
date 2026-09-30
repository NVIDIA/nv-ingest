# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`retriever skill-eval` benchmark."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from nemo_retriever.harness.artifacts import create_session_dir, last_commit
from nemo_retriever.harness.config import REPO_ROOT
from nemo_retriever.tools.skill_eval.dataset import DatasetEntry, load_config, load_eval_manifest
from nemo_retriever.tools.skill_eval.report import overall_recall, write_summary
from nemo_retriever.tools.skill_eval.runner import (
    CONDITIONS,
    DEFAULT_AGENT_MODELS,
    SUPPORTED_AGENTS,
    UNSCORABLE_JUDGE_ERRORS,
    ConditionRun,
    TrialResult,
    _apply_judge,
    archive_session_log,
    cleanup_condition_workdir,
    extract_compact_trace,
    run_condition,
    save_compact_trace,
    save_trial,
)

DEFAULT_ORDER = ("c1_base", "c2_retriever", "c3_retriever_skill")

app = typer.Typer(help="Benchmark coding agents with vs. without the /nemo-retriever skill on a folder of PDFs.")
logger = logging.getLogger(__name__)


def _resolve_pdf_source(
    cfg: dict,
    domain: str,
) -> Path:
    pdf_dirs = cfg.get("pdf_dirs")
    if isinstance(pdf_dirs, dict):
        if domain not in pdf_dirs:
            raise typer.BadParameter(
                f"config 'pdf_dirs' is missing an entry for domain '{domain}'. "
                f"Known domains: {sorted(pdf_dirs.keys())}"
            )
        return Path(str(pdf_dirs[domain])).expanduser().resolve()
    if cfg.get("pdf_dir"):
        return Path(str(cfg["pdf_dir"])).expanduser().resolve()
    raise typer.BadParameter("config must define either 'pdf_dirs' (per-domain map) or 'pdf_dir'.")


_LOCAL_JUDGE_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def _preflight_judge_endpoint(api_base: str, timeout: float = 5.0) -> None:
    """Probe ``/health/ready`` when the judge endpoint is local; fail fast if down.

    Cloud endpoints aren't probed (no guaranteed public health route, and a
    bad cloud config isn't actionable from the runner). A local unreachable
    endpoint nearly always means the user forgot to start the local judge,
    so we surface that hint up front instead of
    burning trials on doomed judge calls.
    """
    from urllib.parse import urlparse
    from urllib.request import urlopen

    host = (urlparse(api_base).hostname or "").lower()
    if host not in _LOCAL_JUDGE_HOSTS:
        return
    health_url = api_base.rstrip("/") + "/health/ready"
    try:
        with urlopen(health_url, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
    except Exception as exc:
        raise typer.BadParameter(
            f"Judge endpoint {api_base} is unreachable ({exc}). "
            "If you're using the bundled local judge helper, start it from the repository root:\n"
            "  docker compose -f nemo_retriever/dev/compose/judge.compose.yaml up -d judge"
        )


def _build_judge(cfg: dict) -> Optional[Any]:
    """Construct an ``LLMJudge`` from ``cfg['judge']`` or return ``None``.

    Skips silently (with a console note) when the API key env var is unset, so
    runs work end-to-end without network access. Import is deferred so the
    ``litellm`` extra isn't required when judging is disabled.
    """
    judge_cfg = cfg.get("judge") or {}
    if not judge_cfg.get("enabled", True):
        typer.echo("Judge disabled by config (judge.enabled=false).")
        return None
    api_key_env = str(judge_cfg.get("api_key_env", "NVIDIA_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        typer.echo(f"Judge disabled: ${api_key_env} is not set in the environment.")
        return None
    try:
        from nemo_retriever.models.llm.clients.judge import LLMJudge
    except ImportError as exc:
        typer.echo(f"Judge disabled: failed to import LLMJudge ({exc}). Install nemo-retriever[llm].")
        return None
    api_base = judge_cfg.get("api_base")
    if api_base:
        _preflight_judge_endpoint(str(api_base))
    judge_kwargs: dict[str, Any] = {
        "model": str(judge_cfg.get("model", "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")),
        "api_base": api_base,
        "api_key": api_key,
    }
    if judge_cfg.get("temperature") is not None:
        judge_kwargs["temperature"] = float(judge_cfg["temperature"])
    if judge_cfg.get("max_tokens") is not None:
        judge_kwargs["max_tokens"] = int(judge_cfg["max_tokens"])
    judge = LLMJudge.from_kwargs(**judge_kwargs)
    typer.echo(f"Judge enabled: model={judge.model}")
    return judge


def _build_trace_summarizer(cfg: dict) -> Optional[Any]:
    """Construct a ``TraceSummarizer`` from ``cfg['summarizer']`` or return ``None``."""
    sum_cfg = cfg.get("summarizer") or {}
    if not sum_cfg.get("enabled", True):
        typer.echo("Trace summarizer disabled by config (summarizer.enabled=false).")
        return None
    if shutil.which("claude") is None:
        typer.echo("Trace summarizer disabled: `claude` CLI is not on PATH.")
        return None
    from nemo_retriever.tools.skill_eval.trace_summarizer import TraceSummarizer

    summarizer = TraceSummarizer.from_kwargs(
        model=str(sum_cfg.get("model", "claude-opus-4-7")),
    )
    typer.echo(f"Trace summarizer enabled: model={summarizer.model}")
    return summarizer


def _resolve_agent(value: str) -> str:
    agent = value.strip().lower()
    if agent not in SUPPORTED_AGENTS:
        raise typer.BadParameter(f"agent must be one of {', '.join(SUPPORTED_AGENTS)}")
    return agent


def _resolve_agent_model(cfg: dict, agent: str, override: Optional[str]) -> str:
    if override:
        return override
    models = cfg.get("agent_models")
    if isinstance(models, dict) and models.get(agent):
        return str(models[agent])
    if cfg.get("agent_model"):
        return str(cfg["agent_model"])
    return DEFAULT_AGENT_MODELS[agent]


def _resolve_workdir_root(cfg: dict) -> Path:
    return Path(str(cfg.get("per_trial_workdir_root", "/tmp/skill_eval"))).expanduser().resolve()


def _relative_artifact(path: Path, session_dir: Path) -> str:
    try:
        return str(path.relative_to(session_dir))
    except ValueError:
        return str(path)


def _trace_label(result: TrialResult) -> str:
    if result.is_setup:
        return "setup"
    return f"query entry_id={result.entry_id} query_id={result.query_id}"


def _process_condition_logs(
    *,
    session_dir: Path,
    agent: str,
    condition: str,
    domain: str,
    condition_run: ConditionRun,
    summarizer: Optional[Any],
) -> list[Path]:
    """Archive raw logs, write compact traces, and optionally summarize them."""
    results = condition_run.results
    if not results:
        return []

    archived_by_session: dict[tuple[str, str], Path | None] = {}
    archived_paths: list[Path] = []
    trace_parts: list[str] = []

    if condition_run.execution_mode == "linear_session":
        first = results[0]
        archived = archive_session_log(
            session_dir=session_dir,
            agent=agent,
            condition=condition,
            domain=domain,
            session_uuid=first.session_id,
            workdir=condition_run.workdir,
        )
        if archived is not None:
            archived_paths.append(archived)
            rel = _relative_artifact(archived, session_dir)
            for r in results:
                r.raw_log_path = rel
        trace = extract_compact_trace(agent, condition_run.workdir, first.session_id)
        if trace:
            trace_path = save_compact_trace(first, session_dir, trace, suffix="session")
            rel_trace = _relative_artifact(trace_path, session_dir)
            for r in results:
                r.compact_trace_path = rel_trace
            trace_parts.append(trace)
    else:
        for result in results:
            workdir = condition_run.result_workdirs.get(result.trial_id, condition_run.workdir)
            session_key = (result.session_id, str(workdir))
            if session_key not in archived_by_session:
                archived = archive_session_log(
                    session_dir=session_dir,
                    agent=agent,
                    condition=condition,
                    domain=domain,
                    session_uuid=result.session_id,
                    workdir=workdir,
                )
                archived_by_session[session_key] = archived
                if archived is not None:
                    archived_paths.append(archived)
            archived = archived_by_session[session_key]
            if archived is not None:
                result.raw_log_path = _relative_artifact(archived, session_dir)

            trace = extract_compact_trace(agent, workdir, result.session_id, first_turn_label=_trace_label(result))
            if not trace:
                continue
            trace_path = save_compact_trace(result, session_dir, trace)
            result.compact_trace_path = _relative_artifact(trace_path, session_dir)
            trace_parts.append(trace)

    if summarizer is not None and trace_parts:
        narrative = summarizer.summarize(
            condition=f"{agent}/{condition}",
            domain=domain,
            trace="\n\n".join(trace_parts),
        )
        if narrative:
            for result in results:
                if result.is_setup:
                    result.tool_use_summary = narrative
                    break
    return archived_paths


def _resolve_conditions(value: Optional[str], cfg: dict) -> list[str]:
    if value is not None:
        selected = [c.strip() for c in value.split(",") if c.strip()]
    else:
        raw = cfg.get("conditions") or list(DEFAULT_ORDER)
        if isinstance(raw, str):
            selected = [c.strip() for c in raw.split(",") if c.strip()]
        elif isinstance(raw, list):
            selected = [str(c).strip() for c in raw if str(c).strip()]
        else:
            raise typer.BadParameter("config 'conditions' must be a list or comma-separated string")
    if not selected:
        raise typer.BadParameter("at least one condition must be selected")
    for c in selected:
        if c not in CONDITIONS:
            raise typer.BadParameter(f"unknown condition '{c}'. Choose from {CONDITIONS}.")
    return selected


def _resolve_domain_label(entries: list[DatasetEntry], cfg: dict, domain: str) -> str:
    """Pick a human-readable label for the setup prompt.

    Prefers the manifest-provided ``domain_label`` carried on the entry. Falls
    back to an optional ``domain_labels`` map in the config, then to ``"PDFs"``.
    """
    for e in entries:
        if e.domain == domain and e.domain_label:
            return e.domain_label
    labels = cfg.get("domain_labels")
    if isinstance(labels, dict) and domain in labels:
        return str(labels[domain])
    return "PDFs"


@app.command("run")
def run_command(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to a skill_eval.yaml; defaults to the packaged config (copy and edit it).",
    ),
    eval_manifest: Optional[Path] = typer.Option(
        None,
        "--eval-manifest",
        help="Path to an agent-eval manifest (JSON list). Overrides config.eval_manifest_path.",
    ),
    conditions: Optional[str] = typer.Option(
        None,
        "--conditions",
        help=(
            "Comma-separated conditions in execution order. Defaults to config.conditions, then "
            f"{','.join(DEFAULT_ORDER)}. Each (agent, condition, domain) workdir is deleted after it runs."
        ),
    ),
    domains: Optional[str] = typer.Option(
        None,
        "--domains",
        help="Optional comma-separated list of domains to include. Defaults to all domains present in the dataset.",
    ),
    artifacts_root: Optional[Path] = typer.Option(
        None, "--artifacts-root", help="Override the artifact root; defaults to <repo>/nemo_retriever/artifacts/"
    ),
    agent_name: Optional[str] = typer.Option(
        None,
        "--agent",
        help="Agent CLI to evaluate: claude or codex. Overrides config.agent.",
    ),
    model_override: Optional[str] = typer.Option(
        None,
        "--model",
        help="Agent model override for this run.",
    ),
    query_parallelism_override: Optional[int] = typer.Option(
        None,
        "--query-parallelism",
        min=1,
        help=(
            "Run query turns after setup in isolated parallel sessions. "
            "Defaults to config.query_parallelism, then 1 (linear session)."
        ),
    ),
) -> None:
    """Run the benchmark across the dataset's domains x selected conditions."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(config)
    agent = _resolve_agent(str(agent_name or cfg.get("agent") or "claude"))
    if shutil.which(agent) is None:
        typer.echo(f"Error: `{agent}` CLI is not on PATH.", err=True)
        raise typer.Exit(code=2)
    selected = _resolve_conditions(conditions, cfg)

    manifest_path = eval_manifest or cfg.get("eval_manifest_path")
    if not manifest_path:
        typer.echo("Error: config is missing 'eval_manifest_path' and --eval-manifest was not provided.", err=True)
        raise typer.Exit(code=2)
    entries = load_eval_manifest(Path(str(manifest_path)).expanduser().resolve())
    typer.echo(f"Loaded {len(entries)} dataset entries.")

    by_domain: dict[str, list[DatasetEntry]] = defaultdict(list)
    for e in entries:
        by_domain[e.domain].append(e)

    if domains:
        wanted = {d.strip() for d in domains.split(",") if d.strip()}
        unknown = wanted - set(by_domain)
        if unknown:
            typer.echo(
                f"Error: --domains references unknown domains {sorted(unknown)}. " f"Available: {sorted(by_domain)}",
                err=True,
            )
            raise typer.Exit(code=2)
        by_domain = {d: by_domain[d] for d in wanted}

    domain_order = sorted(by_domain.keys())
    typer.echo(f"Domains in this run: {domain_order} ({sum(len(v) for v in by_domain.values())} entries total)")

    skill_source = Path(
        str(cfg.get("skill_source_dir") or REPO_ROOT / ".claude" / "skills" / "nemo-retriever")
    ).expanduser()
    if any(c in ("c2_retriever", "c3_retriever_skill") for c in selected) and not (skill_source / "SKILL.md").is_file():
        typer.echo(f"Error: skill source '{skill_source}' does not contain SKILL.md.", err=True)
        raise typer.Exit(code=2)

    workdir_root = _resolve_workdir_root(cfg)
    workdir_root.mkdir(parents=True, exist_ok=True)
    model = _resolve_agent_model(cfg, agent, model_override)
    budget = float(cfg.get("per_trial_budget_usd", 5.0))
    timeout = int(cfg.get("per_trial_timeout_s", 600))
    query_parallelism = int(query_parallelism_override or cfg.get("query_parallelism", 1) or 1)
    if query_parallelism < 1:
        typer.echo("Error: query_parallelism must be >= 1.", err=True)
        raise typer.Exit(code=2)
    testdata_prefixes_raw = cfg.get("testdata_prefixes") or []
    if not isinstance(testdata_prefixes_raw, list):
        typer.echo("Error: config 'testdata_prefixes' must be a list of strings.", err=True)
        raise typer.Exit(code=2)
    testdata_prefixes = tuple(str(p) for p in testdata_prefixes_raw)

    judge = _build_judge(cfg)
    summarizer = _build_trace_summarizer(cfg)

    base_dir = str(artifacts_root) if artifacts_root else None
    session_dir = create_session_dir("skilleval", base_dir=base_dir)
    typer.echo(f"Session dir: {session_dir}")
    typer.echo(f"Agent: {agent}  model={model}  conditions={selected}  query_parallelism={query_parallelism}")

    resolved_cfg = dict(cfg)
    resolved_cfg["agent"] = agent
    resolved_cfg["agent_model"] = model
    resolved_cfg["conditions"] = selected
    resolved_cfg["query_parallelism"] = query_parallelism
    # Capture the commit the evaluation runs with so it survives into the
    # session summary even if the summary is regenerated later from another HEAD.
    resolved_cfg["run_commit"] = last_commit()
    (session_dir / "config.yaml").write_text(yaml.safe_dump(resolved_cfg, default_flow_style=False), encoding="utf-8")

    # Results are keyed (agent, condition, domain) so reports can compare agent runs.
    results_by_key: dict[tuple[str, str, str], list[TrialResult]] = {}
    for cond in selected:
        for domain in domain_order:
            domain_entries = by_domain[domain]
            pdf_source = _resolve_pdf_source(cfg, domain)
            if not pdf_source.is_dir():
                typer.echo(
                    f"Error: PDF directory '{pdf_source}' for domain '{domain}' does not exist or is not a directory. "
                    f"Check the 'pdf_dirs' (or 'pdf_dir') setting in your config.",
                    err=True,
                )
                raise typer.Exit(code=2)
            domain_label = _resolve_domain_label(domain_entries, cfg, domain)
            mode_hint = "linear session" if query_parallelism == 1 else "parallel isolated query sessions"
            typer.echo(
                f"Starting {agent} session for {cond}/{domain} - setup + {len(domain_entries)} query turns "
                f"({mode_hint}, parallelism={query_parallelism}, pdfs={pdf_source})"
            )
            condition_run = run_condition(
                agent=agent,
                condition=cond,
                entries=domain_entries,
                workdir_root=workdir_root,
                pdf_source=pdf_source,
                skill_source=skill_source,
                model=model,
                budget_usd=budget,
                timeout_s=timeout,
                domain=domain,
                domain_label=domain_label,
                judge=judge,
                testdata_prefixes=testdata_prefixes,
                query_parallelism=query_parallelism,
            )
            results = condition_run.results
            archived_logs = _process_condition_logs(
                session_dir=session_dir,
                agent=agent,
                condition=cond,
                domain=domain,
                condition_run=condition_run,
                summarizer=summarizer,
            )
            if summarizer is not None and results:
                setup_summary = next((r.tool_use_summary for r in results if r.is_setup and r.tool_use_summary), "")
                if setup_summary:
                    typer.echo(f"  tool-use summary: {len(setup_summary)} chars")
                else:
                    typer.echo("  tool-use summary skipped: no compact trace or summarizer returned empty")
            for archived in archived_logs:
                typer.echo(f"  archived session log: {archived.relative_to(session_dir)}")

            for r in results:
                save_trial(r, session_dir)
                kind = "setup" if r.is_setup else f"entry_id={r.entry_id} query_id={r.query_id}"
                judge_str = "" if r.is_setup or r.judge_score is None else f" judge={r.judge_score}"
                cost_str = f"${r.total_cost_usd:.3f}" if r.cost_available else "n/a"
                trace_str = f" trace={r.compact_trace_path}" if r.compact_trace_path else ""
                typer.echo(
                    f"  turn {r.num_turns} [{agent}/{domain}] {kind}: status={r.status} "
                    f"tokens(in/out/cache_r)={r.input_tokens}/{r.output_tokens}/{r.cache_read_input_tokens} "
                    f"cost={cost_str} retrieved={len(r.ranked_retrieved)}{judge_str}{trace_str}"
                )
            results_by_key[(agent, cond, domain)] = results

            entries_by_id = {e.entry_id: e for e in domain_entries}
            scores = overall_recall(results, entries_by_id)
            typer.echo(
                f"\nRecall for {agent}/{cond}/{domain}: "
                f"recall@1={scores['recall_1']:.3f}  "
                f"recall@5={scores['recall_5']:.3f}  "
                f"recall@10={scores['recall_10']:.3f}"
            )

            for scratch in sorted(set(condition_run.workdirs), key=lambda p: len(str(p)), reverse=True):
                cleanup_condition_workdir(scratch)
            typer.echo(f"Cleaned up workdirs for {agent}/{cond}/{domain}\n")

    if judge is not None:
        typer.echo("\nLLM-as-judge scores (mean over query turns, 0-5 scale):")
        for cond in selected:
            scored: list[int] = []
            errored = 0
            for domain in domain_order:
                for r in results_by_key.get((agent, cond, domain), []):
                    if r.is_setup:
                        continue
                    if r.judge_score is not None:
                        scored.append(int(r.judge_score))
                    elif r.judge_error:
                        errored += 1
            if scored:
                mean_score = sum(scored) / len(scored)
                typer.echo(f"  {agent}/{cond}: mean={mean_score:.2f}  n={len(scored)}  errors={errored}")
            else:
                typer.echo(f"  {agent}/{cond}: no scores  errors={errored} (check judge config / litellm install)")

    json_path, md_path = write_summary(
        session_dir=session_dir,
        results_by_key=results_by_key,
        entries=entries,
        config=resolved_cfg,
        agent=agent,
        model=model,
        config_path=str(config) if config else "<packaged default>",
    )
    typer.echo(f"\nWrote {json_path}")
    typer.echo(f"Wrote {md_path}")
    typer.echo("\nDone.")


def _needs_rescore(trial: dict[str, Any]) -> bool:
    """Return whether a query-turn trial needs fresh judge scoring."""
    if trial.get("is_setup"):
        return False
    judge_error = trial.get("judge_error") or ""
    if judge_error in UNSCORABLE_JUDGE_ERRORS:
        return False
    score = trial.get("judge_score")
    if score is None:
        return True
    if judge_error:
        return True
    return False


def _load_trial(path: Path) -> tuple[dict[str, Any], TrialResult] | None:
    """Load a trial JSON and reconstruct a ``TrialResult``.

    Returns ``None`` (and logs a warning) if the file is missing, truncated,
    or otherwise unparseable, so callers can skip individual corrupt trials
    without aborting the whole run.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(TrialResult)}
        ctor_kwargs = {k: v for k, v in data.items() if k in known}
        return data, TrialResult(**ctor_kwargs)
    except (OSError, ValueError, TypeError) as exc:
        typer.echo(f"  {path.name}: skip (corrupt trial: {exc})", err=True)
        return None


def _iter_trial_files(session_dir: Path) -> list[Path]:
    return sorted((session_dir / "trials").rglob("*.json"))


@app.command("rescore")
def rescore_command(
    session_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Artifact session directory from a previous `retriever skill-eval run`.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Judge/manifest config to use. Defaults to the session's own config.yaml.",
    ),
    eval_manifest: Optional[Path] = typer.Option(
        None,
        "--eval-manifest",
        help="Manifest path. Overrides eval_manifest_path from --config / session config.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rescore every query-turn trial, not just the empty/failed ones.",
    ),
) -> None:
    """Re-judge query-turn trials with missing or failed judge scores."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    session_dir = session_dir.resolve()
    trials_dir = session_dir / "trials"
    if not trials_dir.is_dir():
        typer.echo(f"Error: {trials_dir} does not exist - is this a skill_eval session dir?", err=True)
        raise typer.Exit(code=2)

    session_cfg_path = session_dir / "config.yaml"
    if config is not None:
        cfg = load_config(config)
        config_path_str = str(config)
    elif session_cfg_path.is_file():
        cfg = load_config(session_cfg_path)
        config_path_str = str(session_cfg_path)
    else:
        typer.echo(
            f"Error: no --config given and {session_cfg_path} is missing; cannot resolve judge settings.",
            err=True,
        )
        raise typer.Exit(code=2)

    manifest_path = eval_manifest or cfg.get("eval_manifest_path")
    if not manifest_path:
        typer.echo("Error: config is missing 'eval_manifest_path' and --eval-manifest was not provided.", err=True)
        raise typer.Exit(code=2)
    entries = load_eval_manifest(Path(str(manifest_path)).expanduser().resolve())
    entries_by_id = {e.entry_id: e for e in entries}

    judge = _build_judge(cfg)
    if judge is None:
        typer.echo("Error: judge is not configured (see messages above). Cannot rescore.", err=True)
        raise typer.Exit(code=2)

    trial_files = _iter_trial_files(session_dir)
    candidates = []
    for path in trial_files:
        loaded = _load_trial(path)
        if loaded is None:
            continue
        data, _ = loaded
        if data.get("is_setup"):
            continue
        if force or _needs_rescore(data):
            candidates.append(path)

    typer.echo(
        f"Rescoring {len(candidates)} trial(s) out of {len(trial_files)} on disk "
        f"(force={'on' if force else 'off'})."
    )

    rescored = 0
    unscorable = 0
    still_failed = 0
    for path in candidates:
        loaded = _load_trial(path)
        if loaded is None:
            continue
        raw, result = loaded
        entry = entries_by_id.get(result.entry_id)
        if entry is None:
            typer.echo(f"  {path.name}: skip (entry_id={result.entry_id} not in manifest)")
            continue

        result.judge_score = None
        result.judge_reasoning = ""
        result.judge_error = ""

        _apply_judge(judge, entry, result)

        raw.update(asdict(result))
        path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

        if result.judge_score is not None:
            rescored += 1
            typer.echo(f"  {path.name}: entry_id={result.entry_id} judge={result.judge_score}")
        elif result.judge_error in UNSCORABLE_JUDGE_ERRORS:
            unscorable += 1
            typer.echo(f"  {path.name}: entry_id={result.entry_id} unscorable ({result.judge_error})")
        else:
            still_failed += 1
            typer.echo(
                f"  {path.name}: entry_id={result.entry_id} still failed " f"(error={result.judge_error or 'unknown'})"
            )

    typer.echo(f"\nRescored {rescored}; unscorable {unscorable}; still failed {still_failed}.")

    results_by_key: dict[tuple[str, str, str], list[TrialResult]] = defaultdict(list)
    for path in trial_files:
        loaded = _load_trial(path)
        if loaded is None:
            continue
        _, result = loaded
        results_by_key[(result.agent, result.condition, result.domain)].append(result)

    agent = str(cfg.get("agent") or next((r.agent for rows in results_by_key.values() for r in rows), "claude"))
    model = _resolve_agent_model(cfg, agent, None)

    json_path, md_path = write_summary(
        session_dir=session_dir,
        results_by_key=dict(results_by_key),
        entries=entries,
        config=cfg,
        agent=agent,
        model=model,
        config_path=config_path_str,
    )
    typer.echo(f"Wrote {json_path}")
    typer.echo(f"Wrote {md_path}")
    typer.echo("\nDone.")
