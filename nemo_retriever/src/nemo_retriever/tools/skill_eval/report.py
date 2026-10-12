# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-trial results into a per-agent / per-condition / per-domain summary."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from nemo_retriever.tools.skill_eval.artifacts import last_commit, write_session_summary
from nemo_retriever.tools.skill_eval.dataset import DatasetEntry
from nemo_retriever.tools.skill_eval.runner import CONDITIONS, TrialResult
from nemo_retriever.tools.skill_eval.score import recall_at_k

METRIC_KS = (1, 5, 10)
RECALL_KEYS = tuple(f"recall_{k}" for k in METRIC_KS)


def _relevant_set(entry: DatasetEntry) -> set[tuple[str, int]]:
    return {(p.doc_id, p.page_number) for p in entry.ground_truth_pages}


def _ranked_pairs(result: TrialResult) -> list[tuple[str, int]]:
    items = sorted(result.ranked_retrieved, key=lambda x: x.get("rank", 999))
    return [(str(item["doc_id"]), int(item["page_number"])) for item in items]


def overall_recall(
    results: Iterable[TrialResult],
    entries_by_id: dict[int, DatasetEntry],
    ks: tuple[int, ...] = METRIC_KS,
) -> dict[str, float]:
    """Macro-averaged recall@k: mean of per-query recall@k across query turns.

    Matches the aggregation used by ``tools.recall.beir.compute_beir_metrics``,
    so skill_eval numbers are directly comparable to BEIR evaluation output.
    """
    per_query: dict[int, list[float]] = {k: [] for k in ks}
    for r in results:
        if r.is_setup:
            continue
        entry = entries_by_id.get(r.entry_id)
        if entry is None:
            continue
        relevant = _relevant_set(entry)
        if not relevant:
            continue
        ranked = _ranked_pairs(r)
        for k in ks:
            per_query[k].append(recall_at_k(ranked, relevant, k))
    return {f"recall_{k}": (sum(v) / len(v)) if v else 0.0 for k, v in per_query.items()}


def _aggregate(
    results: list[TrialResult],
    entries_by_id: dict[int, DatasetEntry],
    *,
    run_name: str,
    artifact_dir: str,
    extra_tags: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Aggregate a flat list of trial results into a single row of metrics."""
    if not results:
        return {}

    query_results = [r for r in results if not r.is_setup]
    setup_results = [r for r in results if r.is_setup]

    metrics: dict[str, Any] = dict(overall_recall(query_results, entries_by_id))
    if query_results:
        metrics["input_tokens"] = mean(r.input_tokens for r in query_results)
        metrics["output_tokens"] = mean(r.output_tokens for r in query_results)
        metrics["cache_read_input_tokens"] = mean(r.cache_read_input_tokens for r in query_results)
        metrics["cache_creation_input_tokens"] = mean(r.cache_creation_input_tokens for r in query_results)
        costed = [r.total_cost_usd for r in query_results if r.cost_available]
        metrics["total_cost_usd"] = mean(costed) if costed else None
        metrics["duration_ms"] = mean(r.duration_ms for r in query_results)
    # When aggregating across multiple sessions there may be more than one setup
    # turn (one per domain); sum them so the one-time cost reflects the full run.
    if setup_results:
        metrics["setup_input_tokens"] = sum(r.input_tokens for r in setup_results)
        metrics["setup_output_tokens"] = sum(r.output_tokens for r in setup_results)
        metrics["setup_cache_read_input_tokens"] = sum(r.cache_read_input_tokens for r in setup_results)
        metrics["setup_cache_creation_input_tokens"] = sum(r.cache_creation_input_tokens for r in setup_results)
        setup_costed = [r.total_cost_usd for r in setup_results if r.cost_available]
        metrics["setup_cost_usd"] = sum(setup_costed) if setup_costed else None
        metrics["setup_duration_ms"] = sum(r.duration_ms for r in setup_results)
        metrics["setup_status"] = (
            "ok" if all(r.status == "ok" for r in setup_results) else ",".join(r.status for r in setup_results)
        )
    metrics["session_input_tokens"] = sum(r.input_tokens for r in results)
    metrics["session_output_tokens"] = sum(r.output_tokens for r in results)
    metrics["session_cache_read_input_tokens"] = sum(r.cache_read_input_tokens for r in results)
    metrics["session_cache_creation_input_tokens"] = sum(r.cache_creation_input_tokens for r in results)
    session_costed = [r.total_cost_usd for r in results if r.cost_available]
    metrics["session_total_cost_usd"] = sum(session_costed) if session_costed else None
    metrics["num_query_turns"] = len(query_results)
    metrics["success_rate"] = sum(1 for r in results if r.status == "ok") / len(results)
    metrics["retriever_attempted_rate"] = sum(1 for r in results if r.retriever_attempted) / len(results)
    metrics["retriever_succeeded_rate"] = sum(1 for r in results if r.retriever_succeeded) / len(results)
    skill_fired = [r.skill_fired for r in results if r.skill_fired is not None]
    if skill_fired:
        metrics["skill_fired_rate"] = sum(1 for x in skill_fired if x) / len(skill_fired)
    judge_scores = [r.judge_score for r in query_results if r.judge_score is not None]
    if judge_scores:
        metrics["judge_score_mean"] = sum(judge_scores) / len(judge_scores)
        metrics["judge_score_n"] = len(judge_scores)

    tool_use_summary = next((r.tool_use_summary for r in setup_results if r.tool_use_summary), "")

    return {
        "run_name": run_name,
        "success": all(r.status == "ok" for r in results),
        "metrics": metrics,
        "tags": [results[0].agent, results[0].condition, *extra_tags, f"n_queries={len(query_results)}"],
        "artifact_dir": artifact_dir,
        "tool_use_summary": tool_use_summary,
    }


def aggregate_condition(results: Iterable[TrialResult], entries_by_id: dict[int, DatasetEntry]) -> dict[str, Any]:
    """Back-compat wrapper kept for callers that flatten per-domain results."""
    results_list = list(results)
    if not results_list:
        return {}
    agent = getattr(results_list[0], "agent", "claude")
    condition = results_list[0].condition
    return _aggregate(
        results_list,
        entries_by_id,
        run_name=f"{agent}/{condition}",
        artifact_dir=str(Path("trials") / agent / condition),
    )


def _fmt_cost(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):.3f}"


def _md_link(label: str, path: str) -> str:
    return "-" if not path else f"[{label}]({path})"


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _md_row(row: dict[str, Any]) -> str:
    m = row.get("metrics", {})
    judge_cell = f"{m['judge_score_mean']:.2f} (n={m.get('judge_score_n', 0)})" if "judge_score_mean" in m else "-"
    return (
        "| {run} | {sr:.2f} | {ra:.2f} | {rs:.2f} | {r1:.3f} | {r5:.3f} | {r10:.3f} | {judge} "
        "| {ipt:.0f} | {opt:.0f} | {cr:.0f} | {cc:.0f} | {cost} |"
    ).format(
        run=row.get("run_name", "?"),
        sr=m.get("success_rate", 0.0),
        ra=m.get("retriever_attempted_rate", 0.0),
        rs=m.get("retriever_succeeded_rate", 0.0),
        r1=m.get("recall_1", 0.0),
        r5=m.get("recall_5", 0.0),
        r10=m.get("recall_10", 0.0),
        judge=judge_cell,
        ipt=m.get("input_tokens", 0.0),
        opt=m.get("output_tokens", 0.0),
        cr=m.get("cache_read_input_tokens", 0.0),
        cc=m.get("cache_creation_input_tokens", 0.0),
        cost=_fmt_cost(m.get("total_cost_usd")),
    )


_MAIN_TABLE_HEADER = (
    "| run | success_rate | retr_attempted | retr_succeeded | recall@1 | recall@5 | recall@10 | judge "
    "| q_input | q_output | q_cache_read | q_cache_create | q_cost |"
)
_MAIN_TABLE_DIVIDER = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def write_summary_md(
    session_dir: Path,
    rows_by_domain: dict[str, list[dict[str, Any]]],
    overall_rows: list[dict[str, Any]],
    config: dict[str, Any],
    agent: str,
    model: str,
    trace_rows_by_domain: dict[str, list[TrialResult]] | None = None,
) -> Path:
    query_parallelism = int(config.get("query_parallelism") or 1)
    if query_parallelism > 1:
        execution_note = (
            f"_Execution mode: setup runs once, then query turns run in isolated sessions "
            f"with parallelism={query_parallelism}. Session totals sum setup plus all isolated query sessions._"
        )
    else:
        execution_note = (
            "_Each (agent, condition, domain) is one agent session: turn 1 = setup, turns 2..N = query turns._"
        )

    run_commit = config.get("run_commit") or last_commit()
    lines = [
        f"# skill_eval session summary - `{session_dir.name}`",
        "",
        f"- Agent: `{agent}`",
        f"- Agent model: `{model}`",
        f"- Run commit: `{run_commit}`",
        f"- Per-trial budget: ${config.get('per_trial_budget_usd', '?')}",
        f"- Per-trial timeout: {config.get('per_trial_timeout_s', '?')}s",
        "",
        "_Agent-session tokens only. Pipeline-side LLM calls (embeddings, VLM, etc.) are not instrumented._",
        execution_note,
        "",
        "## Overall (averaged across all queries in this run)",
        "",
        _MAIN_TABLE_HEADER,
        _MAIN_TABLE_DIVIDER,
    ]
    for row in overall_rows:
        lines.append(_md_row(row))

    for domain in sorted(rows_by_domain):
        rows = rows_by_domain[domain]
        if not rows:
            continue
        lines += [
            "",
            f"## Domain: {domain}",
            "",
            _MAIN_TABLE_HEADER,
            _MAIN_TABLE_DIVIDER,
        ]
        for row in rows:
            lines.append(_md_row(row))

    lines += [
        "",
        "## Setup turns (one-time cost per run, summed across domains)",
        "",
        "| run | status | setup_input | setup_output | setup_cache_read | setup_cost | setup_ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in overall_rows:
        m = row.get("metrics", {})
        lines.append(
            "| {run} | {st} | {ipt:.0f} | {opt:.0f} | {cr:.0f} | {cost} | {ms:.0f} |".format(
                run=row.get("run_name", "?"),
                st=m.get("setup_status", "?"),
                ipt=m.get("setup_input_tokens", 0),
                opt=m.get("setup_output_tokens", 0),
                cr=m.get("setup_cache_read_input_tokens", 0),
                cost=_fmt_cost(m.get("setup_cost_usd")),
                ms=m.get("setup_duration_ms", 0),
            )
        )

    lines += [
        "",
        "## Session totals (setup + all query turns)",
        "",
        "| run | query_turns | total_input | total_output | total_cache_read | total_cache_create | total_cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in overall_rows:
        m = row.get("metrics", {})
        lines.append(
            "| {run} | {n} | {ipt} | {opt} | {cr} | {cc} | {cost} |".format(
                run=row.get("run_name", "?"),
                n=m.get("num_query_turns", 0),
                ipt=m.get("session_input_tokens", 0),
                opt=m.get("session_output_tokens", 0),
                cr=m.get("session_cache_read_input_tokens", 0),
                cc=m.get("session_cache_creation_input_tokens", 0),
                cost=_fmt_cost(m.get("session_total_cost_usd")),
            )
        )

    diag_lines = []
    for row in overall_rows:
        m = row.get("metrics", {})
        if "skill_fired_rate" in m:
            diag_lines.append(f"- **{row['run_name']}**: skill_fired_rate={m['skill_fired_rate']:.2f}")
    if diag_lines:
        lines.append("")
        lines.append("## Diagnostics")
        lines.extend(diag_lines)

    trace_rows_by_domain = trace_rows_by_domain or {}
    if any(trace_rows_by_domain.values()):
        lines += [
            "",
            "## Per-query traces",
            "",
            "| domain | run | entry_id | query_id | status | compact_trace | raw_log |",
            "|---|---|---|---|---|---|---|",
        ]
        for domain in sorted(trace_rows_by_domain):
            rows = sorted(
                trace_rows_by_domain[domain],
                key=lambda r: (_condition_order(r.condition), r.condition, r.entry_id, r.query_id),
            )
            for result in rows:
                run_name = (
                    f"{result.agent}/{result.condition}/{domain}" if domain else f"{result.agent}/{result.condition}"
                )
                lines.append(
                    "| {domain} | {run} | {entry} | {query} | {status} | {trace} | {raw} |".format(
                        domain=_md_cell(domain or "default"),
                        run=_md_cell(run_name),
                        entry=result.entry_id,
                        query=_md_cell(result.query_id),
                        status=_md_cell(result.status),
                        trace=_md_link("trace", result.compact_trace_path),
                        raw=_md_link("log", result.raw_log_path),
                    )
                )

    summary_blocks: list[tuple[str, str]] = []
    for domain in sorted(rows_by_domain):
        for row in rows_by_domain[domain]:
            text = row.get("tool_use_summary") or ""
            if text:
                summary_blocks.append((str(row.get("run_name", "?")), text))
    if summary_blocks:
        lines += ["", "## Tool-use summaries", ""]
        for run_name, text in summary_blocks:
            lines.append(f"### {run_name}")
            lines.append("")
            lines.append(text)
            lines.append("")

    out = session_dir / "session_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _condition_order(condition: str) -> int:
    try:
        return CONDITIONS.index(condition)
    except ValueError:
        return len(CONDITIONS)


def write_summary(
    session_dir: Path,
    results_by_key: dict[tuple[str, str, str], list[TrialResult]],
    entries: list[DatasetEntry],
    config: dict[str, Any],
    agent: str,
    model: str,
    config_path: str,
) -> tuple[Path, Path]:
    entries_by_id = {e.entry_id: e for e in entries}

    # Per-(agent, condition, domain) rows.
    domain_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Roll-up per agent/condition across all domains.
    by_run: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)

    for (agent_name, cond, domain), results in results_by_key.items():
        if not results:
            continue
        if domain:
            artifact_dir = str(Path("trials") / agent_name / cond / domain)
        else:
            artifact_dir = str(Path("trials") / agent_name / cond)
        domain_rows[domain].append(
            _aggregate(
                results,
                entries_by_id,
                run_name=f"{agent_name}/{cond}/{domain}" if domain else f"{agent_name}/{cond}",
                artifact_dir=artifact_dir,
                extra_tags=(f"agent={agent_name}", f"domain={domain}") if domain else (f"agent={agent_name}",),
            )
        )
        by_run[(agent_name, cond)].extend(results)

    overall_rows: list[dict[str, Any]] = []
    for agent_name, cond in sorted(by_run, key=lambda x: (x[0], _condition_order(x[1]), x[1])):
        results = by_run[(agent_name, cond)]
        if not results:
            continue
        overall_rows.append(
            _aggregate(
                results,
                entries_by_id,
                run_name=f"{agent_name}/{cond}",
                artifact_dir=str(Path("trials") / agent_name / cond),
                extra_tags=(f"agent={agent_name}",),
            )
        )

    for rows in domain_rows.values():
        rows.sort(key=lambda row: tuple(str(row.get("run_name", "")).split("/", 2)[:2]))

    flat_rows = overall_rows + [r for rows in domain_rows.values() for r in rows]
    json_path = write_session_summary(
        session_dir=session_dir,
        run_results=flat_rows,
        session_type="skill_eval",
        config_path=config_path,
        run_commit=config.get("run_commit"),
    )
    trace_rows_by_domain: dict[str, list[TrialResult]] = defaultdict(list)
    for (_agent_name, _cond, domain), results in results_by_key.items():
        for result in results:
            if result.is_setup:
                continue
            if result.compact_trace_path or result.raw_log_path:
                trace_rows_by_domain[domain].append(result)

    md_path = write_summary_md(
        session_dir,
        dict(domain_rows),
        overall_rows,
        config,
        agent=agent,
        model=model,
        trace_rows_by_domain=dict(trace_rows_by_domain),
    )
    return json_path, md_path
