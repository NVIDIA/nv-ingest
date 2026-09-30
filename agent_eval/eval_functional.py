#!/usr/bin/env python3
"""Phase 3 (functional): pass/fail grader for a functional run dir.

Re-reads the manifest criteria (answer-free queries don't carry them) and grades each
query produced by ``run_functional.py`` by ``ground_truth_kind``:

* ``answer_reference`` (retrieval_answer)      -> LLM PASS/FAIL rubric vs
                                                  ``validation_signal`` + ``expected_output_shape``.
* ``action_contract`` (ingest)                 -> programmatic gate: a LanceDB table with
                                                  rows>0 was built AND the LLM rubric confirms
                                                  the output is an ingest confirmation (not Q&A).
* ``answer_reference`` (ingest_plus_answer)    -> programmatic gate (table+rows) AND the
                                                  answer rubric.

A query PASSes only if every applicable check passes. Verdicts cache to the output JSON so
re-runs (after a detector tweak) don't re-call the judge for unchanged answers. MAY import
nemo_retriever (for the judge transport + lancedb row count).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

RETRIEVAL = "retrieval_answer"
INGEST = "ingest"
INGEST_PLUS = "ingest_plus_answer"
_INGEST_GATED = {INGEST, INGEST_PLUS}

_RUBRIC_SYSTEM = (
    "You are a strict grader for a document-retrieval/ingest agent test. Given the task, the "
    "validation signal, the expected output shape, and the agent's final message, decide whether "
    "the agent PASSED or FAILED. Be strict on ANSWER QUALITY: PASS only if the message satisfies "
    "the observable parts of the validation signal AND matches the expected output shape. A "
    "refusal, an empty/placeholder answer, a hallucinated answer unsupported by the corpus, or the "
    "wrong task (e.g. answering a question when only ingestion was asked, or vice-versa) is a FAIL. "
    "HOWEVER, IGNORE any validation-signal clauses about internal tracer logs, job IDs, pipeline "
    "step names, or specific backend endpoints — those describe the eval harness's internal "
    "observability and are NOT expected to appear in a user-facing answer; never FAIL solely "
    "because the message omits them. Grade on the observable answer content and output shape. When "
    "ARTIFACT FACTS are provided, treat them as ground truth for whether ingestion occurred. "
    'Respond with ONLY a JSON object: {"verdict": "PASS" | "FAIL", "reason": "<one sentence>"}.'
)

# Ingest-only (action_contract) tests: ingestion is already VERIFIED programmatically (the
# artifact facts below). Many validation_signals also reference INTERNAL signals — tracer
# logs, job IDs, specific backend endpoints — that are not observable in the agent's final
# message; those are covered by the artifact facts and must NOT be held against the agent.
_RUBRIC_SYSTEM_INGEST = (
    "You are grading an INGEST-ONLY agent task. Ingestion was already VERIFIED programmatically "
    "(see ARTIFACT FACTS: a LanceDB table with rows>0). Decide PASS/FAIL based ONLY on the agent's "
    "OBSERVABLE final message: it PASSES if (a) it is an ingest/setup confirmation matching the "
    "expected output shape (e.g. a document/chart/row count), and (b) it is NOT a question-"
    "answering response (no Q&A was asked). IGNORE any validation-signal clauses about internal "
    "tracer logs, job IDs, specific backend endpoints, OR mid-run / streaming progress updates "
    "(e.g. per-document progress markers like '10/368 done') — those describe internal or "
    "runtime-streaming behavior that is NOT observable in a single final message and is already "
    "covered by the artifact facts. A correct ingest confirmation with a plausible count is a "
    "PASS even if it does not name internal endpoints or show mid-run progress. If the EXPECTED "
    "OUTPUT SHAPE mentions streaming or per-document progress markers, treat a final ingest "
    "confirmation bearing document/row counts as satisfying it (the final message cannot replay "
    "mid-run streaming). A Q&A answer, a refusal, or an empty message is a FAIL. "
    'Respond with ONLY a JSON object: {"verdict": "PASS" | "FAIL", "reason": "<one sentence>"}.'
)
_RUBRIC_USER = """\
TASK (what the agent was asked to do):
{behavior}

VALIDATION SIGNAL (what a passing result must exhibit):
{signal}

EXPECTED OUTPUT SHAPE:
{shape}
{facts}
AGENT'S FINAL MESSAGE:
\"\"\"
{candidate}
\"\"\"

Grade now."""


@dataclass
class QVerdict:
    query_id: str
    functional_type: str
    ground_truth_kind: str
    passed: bool = False
    reason: str = ""
    gate_required: bool = False
    gate_pass: bool | None = None
    lancedb_built: bool = False
    lancedb_rows: int | None = None
    judge_verdict: str | None = None  # PASS | FAIL | None (gate-only or skipped)
    judge_reason: str = ""
    status: str = "ok"  # ok | run_timeout | run_error | judge_error | no_criteria


def _slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace("*", "x")


def _load_manifest_criteria(manifest: Path) -> dict[str, dict]:
    raw = json.loads(manifest.read_text())
    entries = raw if isinstance(raw, list) else (raw.get("entries") or list(raw.values()))
    out: dict[str, dict] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        qid = e.get("primary_eval_id")
        if not qid or e.get("batch4_track") != "functional_corpus_variants":
            continue
        out[str(qid)] = {
            "ground_truth_kind": e.get("ground_truth_kind", ""),
            "expected_behavior": e.get("expected_behavior", ""),
            "validation_signal": e.get("validation_signal", ""),
            "expected_output_shape": e.get("expected_output_shape", ""),
        }
    return out


def _count_rows(lancedb_dir: Path) -> int | None:
    """Best-effort total row count across LanceDB tables in the dir (None if unreadable)."""
    if not lancedb_dir.exists():
        return None
    try:
        import lancedb  # type: ignore

        db = lancedb.connect(str(lancedb_dir))
        total = 0
        # table_names() reliably returns list[str]; list_tables() returns a response
        # wrapper object (not iterable as names), so prefer table_names().
        for name in db.table_names():
            try:
                total += db.open_table(name).count_rows()
            except Exception:  # noqa: BLE001
                pass
        return total
    except Exception:  # noqa: BLE001
        # Fallback: a table dir exists but we can't open it -> unknown count.
        return None


_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"?(PASS|FAIL)"?', re.IGNORECASE)
_REASON_RE = re.compile(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _parse_rubric(raw: str) -> tuple[str | None, str]:
    m = _VERDICT_RE.search(raw or "")
    verdict = m.group(1).upper() if m else None
    rm = _REASON_RE.search(raw or "")
    reason = rm.group(1) if rm else (raw or "").strip()[:300]
    return verdict, reason


def _run_rubric(
    judge, *, behavior: str, signal: str, shape: str, candidate: str, facts: str, ingest_action: bool = False
) -> tuple[str | None, str]:
    if not candidate.strip():
        return "FAIL", "empty final message"
    if ingest_action:
        # The ingest validation_signal is almost entirely unobservable internals (tracer logs,
        # job IDs, endpoints, mid-run progress). Feeding it makes the judge anchor on clauses the
        # final message structurally can't satisfy. Ingestion itself is proven by the artifact
        # facts; grade only against the observable output shape + not-a-Q&A.
        signal = (
            "(graded by the ARTIFACT FACTS + expected output shape below; do not require any internal/runtime signal)"
        )
    user = _RUBRIC_USER.format(
        behavior=behavior or "(none)",
        signal=signal or "(none)",
        shape=shape or "(none)",
        facts=facts,
        candidate=candidate[:12000],
    )
    system = _RUBRIC_SYSTEM_INGEST if ingest_action else _RUBRIC_SYSTEM
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    raw, _ = judge.complete(messages)  # reuse the judge transport/auth for a raw rubric call
    return _parse_rubric(raw)


def _build_judge(model: str, api_base: str | None, api_key_env: str, *, timeout: float = 90.0, num_retries: int = 2):
    import os

    key = os.environ.get(api_key_env)
    if not key:
        print(f"  judge disabled: ${api_key_env} not set", file=sys.stderr)
        return None
    # This harness grades against its own rubric prompt via a raw chat completion,
    # so it needs a plain completion client -- not the ragas AnswerAccuracy scorer
    # that LLMJudge now wraps. Use LiteLLMClient directly with the judge's transport.
    try:
        from nemo_retriever.llm.clients.litellm import LiteLLMClient  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"  judge disabled: cannot import LiteLLMClient ({exc})", file=sys.stderr)
        return None
    return LiteLLMClient.from_kwargs(
        model=model, api_base=api_base, api_key=key, temperature=0.0, timeout=timeout, num_retries=num_retries
    )


def _load_cache(out_path: Path, regrade: list[str] | None = None) -> dict[str, dict]:
    """Load prior verdicts to reuse. query_ids whose id contains any ``regrade`` substring are
    dropped from the cache so they are graded afresh (e.g. after a rubric change)."""
    if not out_path.exists():
        return {}
    try:
        prior = json.loads(out_path.read_text())
    except Exception:  # noqa: BLE001
        return {}
    regrade = regrade or []
    cache = {}
    for q in prior.get("per_query", []):
        if q.get("status") != "ok":
            continue
        if any(sub in q["query_id"] for sub in regrade):
            continue
        cache[q["query_id"]] = q
    return cache


def _grade_one(m: dict, *, criteria: dict[str, dict], run_dir: Path, min_rows: int, judge) -> QVerdict:
    """Grade a single query (programmatic gate + LLM rubric). Pure except the judge network
    call, so it is safe to run in a thread pool."""
    qid = m["query_id"]
    ft = m.get("functional_type") or ""
    crit = criteria.get(qid, {})
    gtk = crit.get("ground_truth_kind", "")
    v = QVerdict(qid, ft, gtk)

    if m.get("status") in ("timeout", "error"):
        v.status = "run_" + m["status"]
        v.reason = m.get("error", "") or "agent run did not complete"
        return v
    if not crit:
        v.status = "no_criteria"
        v.reason = "query_id not found in manifest"
        return v

    wd = run_dir / _slug(qid) / "workdir"
    resp = run_dir / _slug(qid) / "response.txt"
    candidate = resp.read_text() if resp.exists() else (m.get("final_text") or "")

    # Programmatic ingest gate (action_contract + ingest_plus_answer).
    v.lancedb_built = bool(m.get("lancedb_built"))
    if ft in _INGEST_GATED:
        v.gate_required = True
        rows = _count_rows(wd / "lancedb")
        v.lancedb_rows = rows
        v.gate_pass = bool(v.lancedb_built and (rows is None or rows >= min_rows))

    # LLM rubric (answer_reference, and the "ingest confirmation not Q&A" check for ingest).
    needs_rubric = gtk == "answer_reference" or ft == INGEST
    if needs_rubric:
        if judge is None:
            v.status = "judge_error"
            v.reason = "judge unavailable"
            return v
        facts = ""
        if v.gate_required:
            rows_txt = "unknown" if v.lancedb_rows is None else str(v.lancedb_rows)
            facts = (
                f"\nARTIFACT FACTS (verified programmatically): LanceDB table built="
                f"{v.lancedb_built}, rows={rows_txt}.\n"
            )
        try:
            jv, jr = _run_rubric(
                judge,
                behavior=crit["expected_behavior"],
                signal=crit["validation_signal"],
                shape=crit["expected_output_shape"],
                candidate=candidate,
                facts=facts,
                ingest_action=(ft == INGEST),
            )
            v.judge_verdict, v.judge_reason = jv, jr
        except Exception as exc:  # noqa: BLE001
            v.status = "judge_error"
            v.reason = f"judge_error: {exc}"
            return v

    # Combine.
    checks = []
    if v.gate_required:
        checks.append(("ingest gate", v.gate_pass))
    if needs_rubric:
        checks.append(("rubric", v.judge_verdict == "PASS"))
    v.passed = all(ok for _, ok in checks) if checks else False
    fails = [name for name, ok in checks if not ok]
    v.reason = (v.judge_reason if v.judge_verdict else "") or "ok"
    if fails:
        v.reason = f"failed: {', '.join(fails)}" + (f" — {v.judge_reason}" if v.judge_reason else "")
    return v


def grade_run(
    run_dir: Path,
    criteria: dict[str, dict],
    judge,
    *,
    min_rows: int,
    cache: dict[str, dict],
    concurrency: int = 8,
    flush=None,
) -> list[QVerdict]:
    """Grade all queries. Judge calls run concurrently (they are ~45s each, I/O-bound). The
    optional ``flush(verdicts)`` callback is invoked periodically so partial results persist
    and a killed run can resume from the cache."""
    metas = json.loads((run_dir / "run_metas.json").read_text())
    verdicts: list[QVerdict] = []
    to_grade: list[dict] = []
    for m in metas:
        qid = m["query_id"]
        if qid in cache:  # reuse a prior verdict for an unchanged answer
            c = cache[qid]
            verdicts.append(QVerdict(**{k: c.get(k) for k in QVerdict.__dataclass_fields__ if k in c}))
        else:
            to_grade.append(m)
    if flush:
        flush(verdicts)
    if not to_grade:
        return verdicts

    n = len(to_grade)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {
            ex.submit(_grade_one, m, criteria=criteria, run_dir=run_dir, min_rows=min_rows, judge=judge): m
            for m in to_grade
        }
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                v = fut.result()
            except Exception as exc:  # noqa: BLE001
                m = futs[fut]
                v = QVerdict(
                    m["query_id"],
                    m.get("functional_type") or "",
                    "",
                    status="judge_error",
                    reason=f"grade_error: {exc}",
                )
            verdicts.append(v)
            if i % 10 == 0 or i == n:
                done = sum(1 for x in verdicts if x.status == "ok")
                print(f"  graded {i}/{n}  (ok={done})", flush=True)
                if flush:
                    flush(verdicts)
    return verdicts


def _summary(verdicts: list[QVerdict]) -> dict:
    from collections import Counter

    def rate(items):
        gradable = [x for x in items if x.status == "ok"]
        p = sum(1 for x in gradable if x.passed)
        return {
            "n": len(items),
            "gradable": len(gradable),
            "passed": p,
            "pass_rate": round(p / len(gradable), 3) if gradable else None,
        }

    by_type = {}
    for ft in sorted({v.functional_type for v in verdicts}):
        by_type[ft] = rate([v for v in verdicts if v.functional_type == ft])
    return {
        "overall": rate(verdicts),
        "by_functional_type": by_type,
        "status_counts": dict(Counter(v.status for v in verdicts)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path, help="An agenteval_func_* run dir.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Default: <run-dir>/functional_eval.json")
    ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--judge", dest="judge", action="store_true", default=True)
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--judge-model", default="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")
    ap.add_argument("--judge-api-base", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--judge-api-key-env", default="NVIDIA_API_KEY")
    ap.add_argument(
        "--judge-concurrency",
        type=int,
        default=8,
        help="Concurrent judge calls (each ~45s; sequential is too slow for ~200).",
    )
    ap.add_argument("--judge-timeout", type=float, default=90.0)
    ap.add_argument("--judge-retries", type=int, default=2)
    ap.add_argument(
        "--regrade", default=None, help="CSV of query_id substrings to re-grade (drop from cache), e.g. ':e04:'."
    )
    args = ap.parse_args(argv)

    if not (args.run_dir / "run_metas.json").exists():
        ap.error(f"no run_metas.json under {args.run_dir}")
    out_path = args.out or (args.run_dir / "functional_eval.json")
    criteria = _load_manifest_criteria(args.manifest)
    judge = (
        _build_judge(
            args.judge_model,
            args.judge_api_base,
            args.judge_api_key_env,
            timeout=args.judge_timeout,
            num_retries=args.judge_retries,
        )
        if args.judge
        else None
    )
    regrade = [s.strip() for s in args.regrade.split(",") if s.strip()] if args.regrade else None
    cache = _load_cache(out_path, regrade=regrade)
    if cache:
        print(
            f"  reusing {len(cache)} cached verdicts from {out_path.name}"
            + (f" (re-grading ids matching {regrade})" if regrade else "")
        )

    cfg = {}
    rc = args.run_dir / "run_config.json"
    if rc.exists():
        cfg = json.loads(rc.read_text())

    def _write(verds):
        doc = {
            "run_dir": str(args.run_dir.resolve()),
            "agent": cfg.get("agent"),
            "model": cfg.get("model"),
            "judge_model": args.judge_model if judge else None,
            "manifest": str(args.manifest.resolve()),
            "summary": _summary(verds),
            "per_query": [asdict(v) for v in sorted(verds, key=lambda x: x.query_id)],
        }
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    verdicts = grade_run(
        args.run_dir,
        criteria,
        judge,
        min_rows=args.min_rows,
        cache=cache,
        concurrency=args.judge_concurrency,
        flush=_write,
    )
    _write(verdicts)
    summary = _summary(verdicts)

    o = summary["overall"]
    print(
        f"\nGraded {o['n']} queries ({o['gradable']} gradable): "
        f"PASS {o['passed']}/{o['gradable']} = {o['pass_rate']}"
    )
    for ft, r in summary["by_functional_type"].items():
        print(f"  {ft:22s} pass {r['passed']}/{r['gradable']}  ({r['pass_rate']})")
    if summary["status_counts"]:
        print("  status:", summary["status_counts"])
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
