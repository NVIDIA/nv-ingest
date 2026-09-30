#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script C: build a report from agent_eval run artifact(s).

Re-reads the original manifest for gold (relevant_pages / answer / scoring_mode),
joins it onto each run's per-question ``output.json`` by ``query_id``, and emits
a ``report.md`` + ``report.json`` per run plus a baseline-vs-skill comparison
when several runs are passed.

Unlike the runner, this script MAY import ``nemo_retriever`` to reuse
``score.recall_at_k`` and ``models.llm.clients.judge.LLMJudge`` (it runs where the
codebase exists). Both imports degrade gracefully if unavailable.

Usage:
    python build_report.py RUN_DIR [RUN_DIR ...] [--manifest M.json] \
        [--judge/--no-judge] [--out report_dir]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import gitinfo  # noqa: E402

# ---- optional reuse of nemo_retriever (report-side only) ------------------- #
try:
    from nemo_retriever.skill_eval.score import recall_at_k  # type: ignore
except Exception:  # noqa: BLE001 - fall back to an inline copy

    def recall_at_k(ranked, relevant, k):  # type: ignore
        rel = set(relevant)
        if not rel:
            return 0.0
        return len(set(ranked[:k]) & rel) / len(rel)


# --------------------------------------------------------------------------- #
# Gold from the manifest
# --------------------------------------------------------------------------- #
@dataclass
class Gold:
    query_id: str
    domain: str
    pages: set[tuple[str, int]]
    answer: str
    scoring_mode: str
    category: str
    question: str = ""


def _norm_doc(doc_id: str) -> str:
    d = str(doc_id).strip()
    return d[:-4] if d.lower().endswith(".pdf") else d


# Per-model token prices ($/1M) for agents whose CLI doesn't emit a dollar cost
# (e.g. codex). Claude emits total_cost_usd directly, so it's not priced here.
# `input` is the FRESH (non-cached) input rate; `cached_input` the cache-read rate.
_PRICING = {
    "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
}


def _compute_cost(tokens: dict, model: str | None) -> float | None:
    """Derive cost from a tokens dict {input, output, cache_read, ...} using
    _PRICING. ``input`` is OpenAI-style inclusive of the cached subset, and
    ``output`` already includes reasoning tokens, so neither is double-counted.
    Returns None if the model isn't priced."""
    p = _PRICING.get(str(model or ""))
    if not p or not tokens:
        return None
    inp = int(tokens.get("input", 0) or 0)
    cached = int(tokens.get("cache_read", 0) or 0)
    out = int(tokens.get("output", 0) or 0)
    fresh = max(0, inp - cached)
    return round((fresh * p["input"] + cached * p["cached_input"] + out * p["output"]) / 1_000_000, 4)


def _norm_tokens(tokens: dict, agent: str | None) -> dict:
    """Normalize a tokens dict to Claude's convention (``input`` = non-cached) so
    totals are apples-to-apples. Codex reports ``input_tokens`` inclusive of the
    cached subset, which would double-count ``cache_read`` in the sum. Cost is
    computed from the raw dict beforehand, so this only affects displayed totals."""
    if str(agent or "") == "codex" and tokens:
        t = dict(tokens)
        t["input"] = max(0, int(t.get("input", 0) or 0) - int(t.get("cache_read", 0) or 0))
        return t
    return tokens


def load_gold(manifest_path: Path) -> dict[str, Gold]:
    raw = json.loads(manifest_path.read_text())
    entries = raw if isinstance(raw, list) else (raw.get("entries") or list(raw.values()))
    gold: dict[str, Gold] = {}
    for e in entries:
        qid = e.get("primary_eval_id")
        if not qid:
            continue
        pages = set()
        for p in e.get("relevant_pages") or []:
            doc, pg = p.get("doc_id"), p.get("page_number_in_doc")
            if doc is not None and pg is not None:
                pages.add((_norm_doc(doc), int(pg)))
        cand = ((e.get("scenario_prompt_candidates") or {}).get("candidates") or [{}])[0]
        gold[str(qid)] = Gold(
            query_id=str(qid),
            domain=str(e.get("domain") or ""),
            pages=pages,
            answer=str(e.get("answer") or ""),
            scoring_mode=str(cand.get("scoring_mode") or e.get("scoring_mode") or ""),
            category=str(e.get("category") or ""),
            question=str(e.get("original_query") or ""),
        )
    return gold


# --------------------------------------------------------------------------- #
# Retriever-usage detection (FIXED: handles `timeout DURATION ...`, env prefixes,
# and pipeline-into-parser failures). Applied to a claude agent_log.jsonl.
# --------------------------------------------------------------------------- #
_PIPELINE_SEP = re.compile(r"(?:;|&&|\|\||\||\n|\$\(|`)")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPERS = {"sudo", "time", "nice", "nohup", "exec", "env", "command", "builtin"}
# Leading shell keywords that precede a command inside control flow, e.g.
# `if [ -n "$RETRIEVER_VENV" ]; then "$RETRIEVER_VENV/bin/retriever" query ...`
# or `... fi "$RETRIEVER_BIN" query ...`. After splitting, the retriever segment may
# start with `then`/`else`/`fi`/`done` etc., which must be stripped or the head token
# reads as the keyword, not the command.
_SHELL_KW = {"then", "do", "else", "elif", "fi", "done", "{"}
_TIMEOUT_VAL_FLAGS = {"-k", "--kill-after", "-s", "--signal"}
_PARSE_ERR = re.compile(r"pdf_basename|JSONDecodeError|Extra data|_default_decoder|KeyError", re.I)
# The baseline profile installs a PATH shim that prints this and exits 127. A
# retriever command that hits it was blocked, not used — don't count it (and don't
# let a pipeline tail like `| head` exiting 0 read as a clean retriever exit).
_SHIM_BLOCKED = re.compile(r"command not found \(baseline profile\)|exited with code 127", re.I)


def _strip_wrappers(seg: str) -> list[str]:
    toks = seg.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if _ENV_ASSIGN.match(t):
            i += 1
            continue
        if t == "timeout":
            i += 1
            while i < len(toks) and toks[i].startswith("-"):
                i += 2 if toks[i] in _TIMEOUT_VAL_FLAGS else 1
            if i < len(toks):  # the DURATION token
                i += 1
            continue
        if t in _WRAPPERS or t in _SHELL_KW:
            i += 1
            continue
        break
    return toks[i:]


def _seg_is_retriever(seg: str) -> bool:
    toks = _strip_wrappers(seg.strip())
    if not toks:
        return False
    # Strip surrounding quotes so a guarded/quoted binary path like
    # `"$RETRIEVER_VENV/bin/retriever"` is recognized (the var stays unexpanded,
    # but the literal still ends in `/retriever`).
    h = toks[0].strip("'\"")
    if h == "retriever" or h.endswith("/retriever"):
        return True
    if len(toks) >= 3 and toks[0] == "uv" and toks[1] == "run" and toks[2] == "retriever":
        return True
    if len(toks) >= 3 and toks[0].startswith("python") and toks[1] == "-m" and toks[2].startswith("nemo_retriever"):
        return True
    return False


# Var names assigned from a resolved retriever binary, e.g.
# `RETRIEVER_BIN="$(command -v retriever)"`, `RETRIEVER_BIN="$RETRIEVER_VENV/bin/retriever"`,
# or `RETRIEVER=retriever`. The agent builds these to harden against an unset
# RETRIEVER_VENV, then invokes `"$RETRIEVER_BIN" query ...` — whose head is a $var,
# not a `retriever` literal, so it slips past _seg_is_retriever.
_RETR_VAR_ASSIGN = re.compile(r"\b([A-Za-z_]\w*)=[^\n;]*\bretriever\b")
_VAR_REF = re.compile(r"^\$\{?([A-Za-z_]\w*)\}?$")


def _retriever_bin_vars(cmd: str) -> set[str]:
    return set(_RETR_VAR_ASSIGN.findall(cmd or ""))


def cmd_uses_retriever(cmd: str) -> bool:
    segs = _PIPELINE_SEP.split(cmd or "")
    if any(_seg_is_retriever(s) for s in segs):
        return True
    # Variable indirection: VAR=<...retriever...> earlier, then `"$VAR" <subcommand>`.
    rvars = _retriever_bin_vars(cmd)
    if rvars:
        for s in segs:
            toks = _strip_wrappers(s.strip())
            if len(toks) >= 2:  # head is the binary, plus at least a subcommand
                m = _VAR_REF.match(toks[0].strip("'\""))
                if m and m.group(1) in rvars:
                    return True
    return False


def _retriever_piped_to_parser(cmd: str) -> bool:
    m = re.search(r"retriever\s+(query|recall|ingest|pipeline)", cmd or "")
    return m is not None and "|" in cmd[m.start() :]


_CODEX_EXIT_RE = re.compile(r"exited with code (\d+)")
_HITS_JSON_RE = re.compile(r'"page_number"')
# Codex backgrounds a slow command (~1s yield) → output says "Process running with
# session ID <n>"; the agent then polls it via a function_call whose arguments carry
# {"session_id": <n>}. The clean exit lands on that poll, not the original query call.
_BG_SESSION_RE = re.compile(r"running with session(?:\s+ID)?\s+(\d+)", re.I)


def detect_retriever_usage_codex(agent_log: Path) -> dict[str, bool]:
    """Return {attempted, clean, engine} for a codex rollout log.

    Codex logs ``response_item`` events: ``function_call`` (exec_command, with the
    shell command in ``arguments.cmd``) and ``function_call_output`` (whose text
    carries ``Process exited with code N``). ``clean`` = a retriever command shows
    a clean exit; ``engine`` also counts turns where retriever-hit JSON
    (``page_number``+``source``/``text``) appears in any output — i.e. the engine
    returned results even if codex's 1s-yield exec never captured a clean exit.
    """
    if not agent_log or not agent_log.exists():
        return {"attempted": False, "clean": False, "engine": False}
    calls: dict[str, str] = {}
    outs: dict[str, str] = {}
    polls: dict[str, str] = {}  # call_id -> polled session_id (background-continue calls)
    hits_seen = False
    for line in agent_log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "response_item":
            continue
        p = ev.get("payload") or {}
        if p.get("type") == "function_call":
            try:
                a = json.loads(p.get("arguments") or "{}")
            except Exception:
                a = {}
            cmd = a.get("cmd") or a.get("command") or ""
            if isinstance(cmd, list):
                cmd = " ".join(str(x) for x in cmd)
            calls[p.get("call_id")] = str(cmd)
            sid = a.get("session_id")
            if sid is not None:
                polls[p.get("call_id")] = str(sid)
        elif p.get("type") == "function_call_output":
            o = p.get("output")
            if isinstance(o, dict):
                o = o.get("output") or json.dumps(o)
            o = str(o or "")
            outs[p.get("call_id")] = o
            if _HITS_JSON_RE.search(o) and ('"source"' in o or '"text"' in o):
                hits_seen = True
    attempted = clean = engine = False
    retr_sessions: set[str] = set()  # session IDs opened by a backgrounded retriever query
    for cid, cmd in calls.items():
        if not cmd_uses_retriever(cmd):
            continue
        out = outs.get(cid, "")
        if _SHIM_BLOCKED.search(out):  # baseline shim blocked it — not real retriever use
            continue
        attempted = True
        m = _CODEX_EXIT_RE.search(out)
        if m and m.group(1) == "0":
            clean = True
            engine = True
        bg = _BG_SESSION_RE.search(out)  # query backgrounded → remember its session
        if bg:
            retr_sessions.add(bg.group(1))
    # A backgrounded `retriever query` finishes in a later session-poll, not the original
    # call — credit that poll's clean exit to the query. Follow re-yields (a poll may
    # background again under a new session id) to a fixpoint, bounded by #polls.
    if retr_sessions and not clean:
        changed = True
        while changed and not clean:
            changed = False
            for cid, sid in polls.items():
                if sid not in retr_sessions:
                    continue
                out = outs.get(cid, "")
                m = _CODEX_EXIT_RE.search(out)
                if m and m.group(1) == "0":
                    clean = engine = True
                    break
                nb = _BG_SESSION_RE.search(out)
                if nb and nb.group(1) not in retr_sessions:
                    retr_sessions.add(nb.group(1))
                    changed = True
    # Hits-JSON fallback (query returned results even if no clean exit was captured),
    # gated on a real retriever attempt so direct LanceDB reads aren't miscounted.
    if attempted and hits_seen:
        engine = True
    return {"attempted": attempted, "clean": clean, "engine": engine}


def detect_retriever_usage(agent_log: Path) -> dict[str, bool]:
    """Return {attempted, clean, engine} for a claude session log."""
    if not agent_log or not agent_log.exists():
        return {"attempted": False, "clean": False, "engine": False}
    cmd_by_id: dict[str, str] = {}
    res_by_id: dict[str, tuple[bool, str]] = {}
    for line in agent_log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "assistant":
            for b in (ev.get("message", {}) or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash":
                    cmd_by_id[b.get("id")] = (b.get("input") or {}).get("command") or ""
        elif ev.get("type") == "user":
            for b in (ev.get("message", {}) or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    res_by_id[b.get("tool_use_id")] = (bool(b.get("is_error")), str(c or ""))
    attempted = clean = engine = False
    for tid, cmd in cmd_by_id.items():
        if not cmd_uses_retriever(cmd):
            continue
        iserr, txt = res_by_id.get(tid, (True, ""))
        if _SHIM_BLOCKED.search(txt):  # baseline shim blocked it — not real retriever use
            continue
        attempted = True
        if tid not in res_by_id:
            continue
        if not iserr:
            clean = True
            engine = True
        elif _retriever_piped_to_parser(cmd) and _PARSE_ERR.search(txt):
            engine = True  # retriever produced output; downstream parser failed
    return {"attempted": attempted, "clean": clean, "engine": engine}


# --------------------------------------------------------------------------- #
# Per-query scoring
# --------------------------------------------------------------------------- #
_REFUSAL_RE = re.compile(
    r"\b(not|cannot|can't|couldn't|could not|unable|no information|not (found|available|in|present|"
    r"mentioned|provided|contained|included)|does not (appear|contain|mention)|out[- ]of[- ]corpus)\b",
    re.I,
)


@dataclass
class QueryScore:
    query_id: str
    domain: str
    category: str
    scoring_mode: str
    status: str
    has_gold_pages: bool
    recall: dict[int, float] = field(default_factory=dict)
    judge_score: float | None = None
    judge_error: str = ""
    refusal_expected: bool = False
    refusal_correct: bool | None = None
    retr_attempted: bool = False
    retr_clean: bool = False
    retr_engine: bool = False
    cost_usd: float | None = None
    duration_ms: int | None = None
    tokens: dict = field(default_factory=dict)


def _ranked_pairs(selected_chunks: list[dict], page_base: int = 0) -> list[tuple[str, int]]:
    """Ranked (doc, page) pairs normalized to the 0-indexed gold convention by
    subtracting ``page_base`` (1 for retriever-emitted 1-indexed pages, 0 for
    native 0-indexed)."""
    pairs: list[tuple[str, int]] = []
    for c in sorted(selected_chunks, key=lambda x: x.get("rank", 999)):
        loc = c.get("locator") or {}
        pg = loc.get("page_number")
        if pg is None:
            continue
        pairs.append((_norm_doc(c.get("doc_id", "")), int(pg) - page_base))
    return pairs


def score_query(
    qdir: Path, gold: Gold | None, ks=(1, 5, 10), page_base: int = 0, agent: str = "claude"
) -> QueryScore | None:
    out_path = qdir / "output.json"
    meta_path = qdir / "meta.json"
    if not out_path.exists():
        return None
    out = json.loads(out_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("dry_run"):
        return None
    qid = out.get("query_id") or qdir.name
    final_answer = str(out.get("final_answer", ""))
    chunks = out.get("selected_chunks") or []

    qs = QueryScore(
        query_id=qid,
        domain=(gold.domain if gold else meta.get("domain", "")),
        category=(gold.category if gold else ""),
        scoring_mode=(gold.scoring_mode if gold else ""),
        status=meta.get("status", "ok"),
        has_gold_pages=bool(gold and gold.pages),
        cost_usd=meta.get("cost_usd"),
        duration_ms=meta.get("duration_ms"),
        tokens=meta.get("tokens") or {},
    )
    if qs.cost_usd is None:  # agent CLI emitted no $ (e.g. codex) — derive from RAW tokens
        qs.cost_usd = _compute_cost(qs.tokens, meta.get("model"))
    qs.tokens = _norm_tokens(qs.tokens, agent)  # then normalize for apples-to-apples totals
    # recall (only where gold pages exist)
    if gold and gold.pages:
        ranked = _ranked_pairs(chunks, page_base)
        for k in ks:
            qs.recall[k] = recall_at_k(ranked, gold.pages, k)
    # refusal
    qs.refusal_expected = (qs.scoring_mode == "refusal") or (qs.category == "refusal")
    if qs.refusal_expected:
        qs.refusal_correct = bool(_REFUSAL_RE.search(final_answer))
    # retriever usage (agent-specific log format)
    _detect = detect_retriever_usage_codex if agent == "codex" else detect_retriever_usage
    usage = _detect(qdir / "agent_log.jsonl")
    qs.retr_attempted, qs.retr_clean, qs.retr_engine = usage["attempted"], usage["clean"], usage["engine"]
    return qs


# --------------------------------------------------------------------------- #
# Judge (optional, reuse nemo_retriever LLMJudge)
# --------------------------------------------------------------------------- #
def build_judge(model: str, api_base: str | None, api_key_env: str):
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"  judge disabled: ${api_key_env} not set", file=sys.stderr)
        return None
    try:
        from nemo_retriever.models.llm.clients.judge import LLMJudge  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"  judge disabled: cannot import LLMJudge ({exc})", file=sys.stderr)
        return None
    return LLMJudge.from_kwargs(model=model, api_base=api_base, api_key=api_key, temperature=0.0)


def _load_judge_cache(run_dir: Path) -> dict[str, tuple]:
    """Reuse judge scores from a prior report.json so re-runs (e.g. after a
    detector fix) don't re-call the judge for unchanged answers."""
    rp = run_dir / "report.json"
    if not rp.exists():
        return {}
    try:
        r = json.loads(rp.read_text())
    except Exception:
        return {}
    # Only reuse SUCCESSFUL judgements; a prior run that scored None (e.g. judge
    # import broken / disabled) must not poison re-judging into skipping forever.
    return {
        q["query_id"]: (q.get("judge_score"), q.get("judge_error", ""))
        for q in r.get("per_query", [])
        if q.get("judge_score") is not None
    }


def apply_judge(
    judge,
    qscores: list[QueryScore],
    qdir_by_id: dict[str, Path],
    gold: dict[str, Gold],
    workers=8,
    cache: dict[str, tuple] | None = None,
):
    cache = cache or {}

    def _one(qs: QueryScore):
        if qs.query_id in cache:  # reuse prior judge result
            qs.judge_score, qs.judge_error = cache[qs.query_id]
            return
        g = gold.get(qs.query_id)
        if not g or not g.answer:
            qs.judge_error = "no_ground_truth"
            return
        out = json.loads((qdir_by_id[qs.query_id] / "output.json").read_text())
        cand = str(out.get("final_answer", ""))
        if not cand.strip():
            qs.judge_error = "empty_candidate"
            return
        try:
            res = judge.judge(query=g.question or qs.query_id, reference=g.answer, candidate=cand)
            qs.judge_score = res.score
            qs.judge_error = res.error or ""
        except Exception as exc:  # noqa: BLE001
            qs.judge_error = f"judge_error: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, qscores))


# --------------------------------------------------------------------------- #
# Aggregation + rendering
# --------------------------------------------------------------------------- #
def _agg(scores: list[QueryScore]) -> dict[str, Any]:
    n = len(scores)
    if not n:
        return {}

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(mean(vals), 4) if vals else None

    recall = {k: _mean([s.recall.get(k) for s in scores if s.has_gold_pages]) for k in (1, 5, 10)}
    judged = [s.judge_score for s in scores if s.judge_score is not None]
    refusals = [s for s in scores if s.refusal_expected]
    return {
        "n": n,
        "success_rate": round(sum(1 for s in scores if s.status == "ok") / n, 4),
        "retr_attempted": round(sum(s.retr_attempted for s in scores) / n, 4),
        "retr_succeeded_clean": round(sum(s.retr_clean for s in scores) / n, 4),
        "retr_succeeded_engine": round(sum(s.retr_engine for s in scores) / n, 4),
        "recall_1": recall[1],
        "recall_5": recall[5],
        "recall_10": recall[10],
        "n_with_gold_pages": sum(1 for s in scores if s.has_gold_pages),
        "judge_mean": round(mean(judged), 3) if judged else None,
        "judge_n": len(judged),
        "refusal_n": len(refusals),
        "refusal_correct_rate": (
            round(sum(1 for s in refusals if s.refusal_correct) / len(refusals), 4) if refusals else None
        ),
        "cost_usd_total": round(sum(s.cost_usd for s in scores if s.cost_usd), 4),
        "duration_ms_mean": (
            round(mean([s.duration_ms for s in scores if s.duration_ms]), 0)
            if any(s.duration_ms for s in scores)
            else None
        ),
    }


def _fmt(v):
    return "n/a" if v is None else (f"{v}")


def _render_tokens_md(tokens: dict, setup_metas: list[dict]) -> str:
    cols = ["n_turns", "input", "output", "cache_read", "cache_creation", "total", "cost_usd"]
    head = "| phase | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    md = [
        "\n## Tokens (full run, split by setup vs query)\n",
        "_setup = agent-driven index-build turn(s); query = per-question turns. "
        "Pipeline-side embedding/OCR compute is not token-instrumented._\n",
        head,
    ]
    for phase in ("setup", "query", "full_run"):
        t = tokens.get(phase, {})
        md.append("| " + phase + " | " + " | ".join(_fmt(t.get(c)) for c in cols) + " |\n")
    if setup_metas:
        md.append(
            "\nSetup turns: "
            + ", ".join(f"{m.get('domain')}={m.get('status')}(index_built={m.get('index_built')})" for m in setup_metas)
            + "\n"
        )
    return "".join(md)


def _render_code_version_md(code_commit: dict | None) -> str:
    """A '## Code version' block: the commit that RAN the eval (run_config.code_commit) if
    recorded, plus the report-time HEAD for reference."""
    report_info = gitinfo.git_commit()
    lines = ["\n## Code version\n"]
    if code_commit:
        lines.append(f"- eval-time code commit: {gitinfo.format_commit(code_commit)}\n")
    else:
        lines.append("- eval-time code commit: _not recorded — this run predates commit logging_\n")
    lines.append(f"- report generated from working tree: {gitinfo.format_commit(report_info)}\n")
    if not code_commit:
        lines.append(
            "- ⚠️ eval-time commit unknown; the report-time version above may differ "
            "from the code that produced this run.\n"
        )
    return "".join(lines)


def render_md(
    run_label: str,
    scores: list[QueryScore],
    tokens: dict | None = None,
    setup_metas: list[dict] | None = None,
    code_commit: dict | None = None,
) -> str:
    overall = _agg(scores)
    by_domain: dict[str, list[QueryScore]] = defaultdict(list)
    by_cat: dict[str, list[QueryScore]] = defaultdict(list)
    for s in scores:
        by_domain[s.domain].append(s)
        by_cat[s.category or "?"].append(s)
    cols = [
        "n",
        "success_rate",
        "retr_attempted",
        "retr_succeeded_clean",
        "retr_succeeded_engine",
        "recall_1",
        "recall_5",
        "recall_10",
        "judge_mean",
        "judge_n",
        "refusal_correct_rate",
        "cost_usd_total",
    ]
    head = "| group | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"

    def row(label, agg):
        return "| " + label + " | " + " | ".join(_fmt(agg.get(c)) for c in cols) + " |\n"

    md = [
        f"# agent_eval report — `{run_label}`\n",
        "_recall computed only over queries with gold pages (`n_with_gold_pages`); "
        "retr_succeeded_clean = retriever command exited 0; retr_succeeded_engine also counts "
        "retriever calls whose only failure was a downstream parser._\n",
        f"- queries: {overall.get('n')}  | with gold pages: {overall.get('n_with_gold_pages')}\n",
        _render_code_version_md(code_commit),
        "\n## Overall\n",
        head,
        row("overall", overall),
    ]
    if tokens:
        md.append(_render_tokens_md(tokens, setup_metas or []))
    md.append("\n## By domain\n" + head)
    for dom in sorted(by_domain):
        md.append(row(dom, _agg(by_domain[dom])))
    md.append("\n## By category\n" + head)
    for cat in sorted(by_cat):
        md.append(row(cat, _agg(by_cat[cat])))
    return "".join(md)


_TOK_FIELDS = ("input", "output", "cache_read", "cache_creation")


def _sum_tokens(token_dicts: list[dict]) -> dict[str, int]:
    out = {f: 0 for f in _TOK_FIELDS}
    for t in token_dicts:
        for f in _TOK_FIELDS:
            out[f] += int((t or {}).get(f, 0) or 0)
    out["total"] = sum(out[f] for f in _TOK_FIELDS)
    return out


def _load_setup_metas(run_dir: Path) -> list[dict]:
    """Collect agent setup-turn metas (one per domain) from the _setup tree."""
    metas: list[dict] = []
    setup_root = run_dir / "_setup"
    if setup_root.exists():
        for sm in sorted(setup_root.glob("*/setup_meta.json")):
            try:
                metas.append(json.loads(sm.read_text()))
            except Exception:  # noqa: BLE001
                pass
    return metas


def _token_summary(scores: list[QueryScore], setup_metas: list[dict]) -> dict[str, Any]:
    query_tok = _sum_tokens([s.tokens for s in scores])
    setup_tok = _sum_tokens([_norm_tokens(m.get("tokens") or {}, m.get("agent")) for m in setup_metas])
    full = {f: setup_tok[f] + query_tok[f] for f in (*_TOK_FIELDS, "total")}
    query_cost = round(sum(s.cost_usd for s in scores if s.cost_usd), 4)
    setup_cost = round(
        sum(
            (
                m.get("cost_usd")
                if m.get("cost_usd") is not None
                else (_compute_cost(m.get("tokens") or {}, m.get("model")) or 0)
            )
            for m in setup_metas
        ),
        4,
    )
    return {
        "setup": setup_tok | {"cost_usd": setup_cost, "n_turns": len(setup_metas)},
        "query": query_tok | {"cost_usd": query_cost, "n_turns": len(scores)},
        "full_run": full | {"cost_usd": round(setup_cost + query_cost, 4)},
    }


def report_for_run(run_dir: Path, gold: dict[str, Gold], judge, ks=(1, 5, 10)) -> dict[str, Any]:
    cfg_path = run_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    page_base = int(cfg.get("page_index_base", 0))
    agent = str(cfg.get("agent", "claude"))
    qdirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith("_") and (d / "output.json").exists()]
    scores: list[QueryScore] = []
    qdir_by_id: dict[str, Path] = {}
    for d in sorted(qdirs):
        qs = score_query(
            d, gold.get(json.loads((d / "output.json").read_text()).get("query_id", "")), ks, page_base, agent
        )
        if qs is None:
            continue
        scores.append(qs)
        qdir_by_id[qs.query_id] = d
    if judge:
        apply_judge(judge, scores, qdir_by_id, gold, cache=_load_judge_cache(run_dir))
    label = cfg.get("run_id", run_dir.name)
    setup_metas = _load_setup_metas(run_dir)
    tokens = _token_summary(scores, setup_metas)
    code_commit = cfg.get("code_commit")
    md = render_md(label, scores, tokens, setup_metas, code_commit=code_commit)
    (run_dir / "report.md").write_text(md)
    report = {
        "run_id": label,
        "agent": cfg.get("agent"),
        "model": cfg.get("model"),
        "profile": cfg.get("profile"),
        "code_commit": code_commit,
        "overall": _agg(scores),
        "tokens": tokens,
        "setup_turns": setup_metas,
        "per_query": [vars(s) | {"recall": {str(k): v for k, v in s.recall.items()}} for s in scores],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def render_comparison(reports: list[dict]) -> str:
    cols = [
        "profile",
        "agent",
        "model",
        "success_rate",
        "retr_attempted",
        "retr_succeeded_clean",
        "retr_succeeded_engine",
        "recall_1",
        "recall_5",
        "recall_10",
        "judge_mean",
        "refusal_correct_rate",
    ]
    head = "| run | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    md = ["# agent_eval comparison\n\n"]
    # Code version per run (eval-time commit) + report-time HEAD.
    md.append("## Code version\n\n| run | eval-time code commit |\n|---|---|\n")
    for r in reports:
        ci = r.get("code_commit")
        md.append(
            f"| {r['run_id']} | "
            + (gitinfo.format_commit(ci) if ci else "_not recorded — predates commit logging_")
            + " |\n"
        )
    md.append(f"\nReport generated from working tree at {gitinfo.format_commit(gitinfo.git_commit())}.\n\n")
    md.append(head)
    for r in reports:
        o = r["overall"]
        vals = [r.get("profile"), r.get("agent"), r.get("model")] + [o.get(c) for c in cols[3:]]
        md.append("| " + r["run_id"] + " | " + " | ".join(_fmt(v) for v in vals) + " |\n")
    # token totals comparison
    md.append("\n## Tokens (setup / query / full-run total)\n")
    md.append("| run | setup_total | query_total | full_run_total | full_run_cost_usd |\n|---|---|---|---|---|\n")
    for r in reports:
        t = r.get("tokens", {})
        md.append(
            f"| {r['run_id']} | {_fmt(t.get('setup', {}).get('total'))} | "
            f"{_fmt(t.get('query', {}).get('total'))} | {_fmt(t.get('full_run', {}).get('total'))} | "
            f"{_fmt(t.get('full_run', {}).get('cost_usd'))} |\n"
        )
    return "".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path, help="One or more agent_eval run directories.")
    ap.add_argument(
        "--manifest", type=Path, default=None, help="Gold manifest. Defaults to each run's run_config.source_manifest."
    )
    ap.add_argument("--judge", dest="judge", action="store_true", default=None)
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--judge-model", default="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")
    ap.add_argument("--judge-api-base", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--judge-api-key-env", default="NVIDIA_API_KEY")
    ap.add_argument("--out", type=Path, default=None, help="Dir for the comparison report (default: first run dir).")
    args = ap.parse_args(argv)

    # Resolve manifest (shared) — prefer flag, else first run's config.
    manifest = args.manifest
    if manifest is None:
        cfg0 = args.run_dirs[0] / "run_config.json"
        if cfg0.exists():
            sm = json.loads(cfg0.read_text()).get("source_manifest")
            manifest = Path(sm) if sm else None
    if not manifest or not manifest.exists():
        ap.error("could not resolve a gold manifest; pass --manifest")
    gold = load_gold(manifest)
    print(f"Loaded gold for {len(gold)} queries from {manifest}")

    want_judge = args.judge if args.judge is not None else True
    judge = build_judge(args.judge_model, args.judge_api_base, args.judge_api_key_env) if want_judge else None
    if want_judge and judge:
        print(f"Judge enabled: {judge.model}")

    reports = []
    for rd in args.run_dirs:
        print(f"Reporting {rd} ...")
        reports.append(report_for_run(rd, gold, judge))
        o = reports[-1]["overall"]
        print(
            f"  -> success={o.get('success_rate')} recall@10={o.get('recall_10')} "
            f"retr_clean={o.get('retr_succeeded_clean')} retr_engine={o.get('retr_succeeded_engine')} "
            f"judge={o.get('judge_mean')}  (report.md/json written)"
        )

    if len(reports) > 1:
        out_dir = args.out or args.run_dirs[0].parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "comparison.md").write_text(render_comparison(reports))
        (out_dir / "comparison.json").write_text(json.dumps(reports, indent=2, default=str))
        print(f"\nComparison -> {out_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
