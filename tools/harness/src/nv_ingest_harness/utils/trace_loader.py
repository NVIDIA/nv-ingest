from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .trace_summary import summarize_traces


def _load_traces_from_dir(
    trace_dir: Path,
) -> Optional[Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]]:
    """
    Read every *.json trace payload, normalizing indices and source ids.
    Returns (traces_list, results_stub) tuples that summarize_traces() expects.
    """
    trace_files = sorted(trace_dir.glob("*.json"))
    if not trace_files:
        return None

    traces: List[Optional[Dict[str, Any]]] = []
    results_stub: List[List[Dict[str, Any]]] = []

    for trace_file in trace_files:
        payload = json.loads(trace_file.read_text())
        trace_payload = payload.get("trace") or payload
        doc_index = payload.get("document_index")
        if doc_index is None:
            doc_index = len(traces)
        while len(traces) <= doc_index:
            traces.append(None)
            results_stub.append([])
        traces[doc_index] = trace_payload
        source_id = payload.get("source_id")
        results_stub[doc_index] = [
            {
                "metadata": {
                    "source_metadata": {
                        "source_id": source_id,
                    }
                }
            }
        ]

    clean_traces: List[Dict[str, Any]] = [trace or {} for trace in traces]
    return clean_traces, results_stub


def ensure_trace_summary(
    data: Dict[str, Any], results_path: Path, trace_dir_override: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Ensure data["trace_summary"] exists and contains per-document totals.
    Attempts to rebuild from raw traces if missing.
    """
    trace_summary = data.get("trace_summary")
    if isinstance(trace_summary, dict) and trace_summary.get("document_totals"):
        return trace_summary

    candidate_dirs: List[Path] = []
    if trace_dir_override:
        candidate_dirs.append(trace_dir_override)
    if isinstance(trace_summary, dict):
        output_dir = trace_summary.get("output_dir")
        if output_dir:
            candidate_dirs.append(Path(output_dir))
    candidate_dirs.append(results_path.parent / "traces")

    visited: set[Path] = set()
    for candidate in candidate_dirs:
        if not candidate:
            continue
        candidate = candidate.expanduser().resolve()
        if candidate in visited or not candidate.is_dir():
            continue
        visited.add(candidate)
        loaded = _load_traces_from_dir(candidate)
        if not loaded:
            continue
        traces_list, results_stub = loaded
        rebuilt = summarize_traces(traces_list, results_stub, trace_dir=None)
        if rebuilt:
            rebuilt.setdefault("output_dir", str(candidate))
            data["trace_summary"] = rebuilt
            return rebuilt

    print("Unable to locate trace_summary data; stage/wall plots may be skipped.")
    return None
