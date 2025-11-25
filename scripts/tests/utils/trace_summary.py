import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

NS_IN_SECOND = 1_000_000_000


def _ns_to_seconds(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / NS_IN_SECOND


def _safe_trace_filename(source_id: str | None, index: int) -> str:
    base = source_id or f"document_{index}"
    base = os.path.basename(str(base))
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:80]
    if not sanitized:
        sanitized = f"document_{index}"
    return f"{index:03d}_{sanitized}"


def summarize_traces(
    traces_list: List[dict],
    results: List,
    trace_dir: str | None,
) -> Optional[dict]:
    if not traces_list:
        return None

    valid_traces = [(idx, trace) for idx, trace in enumerate(traces_list) if trace]
    if not valid_traces:
        return None

    stage_totals = defaultdict(list)
    document_totals = []

    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    for doc_index, trace_payload in valid_traces:
        if not isinstance(trace_payload, dict):
            continue

        source_id = _extract_source_id(results, doc_index)
        stage_summary = {}
        total_resident = 0.0
        doc_first_entry_ns = None
        doc_last_exit_ns = None
        doc_submission_ts_ns = _parse_submission_ts(trace_payload)
        queue_wait_totals: Dict[str, float] = defaultdict(float)

        for key, value in trace_payload.items():
            if not key.startswith("trace::resident_time::"):
                continue
            stage = key.replace("trace::resident_time::", "")
            resident_s = _ns_to_seconds(value) or 0.0
            entry = trace_payload.get(f"trace::entry::{stage}")
            exit_value = trace_payload.get(f"trace::exit::{stage}")
            wall_s = None
            if entry is not None and exit_value is not None:
                wall_s = _ns_to_seconds(exit_value - entry)
                doc_first_entry_ns = _min_val(doc_first_entry_ns, entry)
                doc_last_exit_ns = _max_val(doc_last_exit_ns, exit_value)

            stage_entry = {"resident_s": round(resident_s, 6)}
            if wall_s is not None:
                stage_entry["wall_s"] = round(wall_s, 6)

            stage_summary[stage] = stage_entry
            total_resident += resident_s
            stage_totals[stage].append(resident_s)
            if stage.endswith("_channel_in"):
                queue_wait_totals[stage] += resident_s

        doc_record = {
            "document_index": doc_index,
            "source_id": source_id,
            "total_resident_s": round(total_resident, 6),
        }
        if queue_wait_totals:
            doc_record["in_ray_queue_s"] = round(sum(queue_wait_totals.values()), 6)
        if doc_submission_ts_ns is not None:
            doc_record["submission_ts_s"] = round(_ns_to_seconds(doc_submission_ts_ns), 6)
        if doc_first_entry_ns is not None and doc_last_exit_ns is not None and doc_last_exit_ns >= doc_first_entry_ns:
            doc_record["ray_start_ts_s"] = round(_ns_to_seconds(doc_first_entry_ns), 6)
            doc_record["ray_end_ts_s"] = round(_ns_to_seconds(doc_last_exit_ns), 6)
            doc_record["total_wall_s"] = round(_ns_to_seconds(doc_last_exit_ns - doc_first_entry_ns), 6)
            if doc_submission_ts_ns is not None and doc_first_entry_ns >= doc_submission_ts_ns:
                doc_record["ray_wait_s"] = round(
                    _ns_to_seconds(doc_first_entry_ns - doc_submission_ts_ns),
                    6,
                )

        document_totals.append(doc_record)
        if trace_dir:
            _write_trace_payload(trace_dir, source_id, doc_index, trace_payload, stage_summary)

    if not stage_totals:
        return None

    stage_summary = _build_stage_summary(stage_totals)
    return {
        "documents": len(document_totals),
        "output_dir": trace_dir,
        "stage_totals": stage_summary,
        "document_totals": document_totals,
    }


def _extract_source_id(results, doc_index: int) -> str:
    source_id = None
    if results and doc_index < len(results):
        source_id = _extract_source_from_results(results[doc_index])
    if not source_id:
        source_id = f"document_{doc_index}"
    return source_id


def _extract_source_from_results(doc_results) -> Optional[str]:
    if doc_results is None:
        return None
    try:
        first_entry = doc_results[0]
    except (IndexError, KeyError, TypeError):
        return None
    except Exception:
        try:
            iterator = iter(doc_results)
            first_entry = next(iterator)
        except Exception:
            return None
    metadata = first_entry.get("metadata", {}) if isinstance(first_entry, dict) else {}
    source_meta = metadata.get("source_metadata", {})
    return source_meta.get("source_id")


def _parse_submission_ts(payload: dict) -> Optional[int]:
    submission_ts_ns = payload.get("submission_ts_ns")
    if isinstance(submission_ts_ns, (int, float)):
        return int(submission_ts_ns)
    if isinstance(submission_ts_ns, str):
        try:
            return int(submission_ts_ns)
        except ValueError:
            return None
    return None


def _write_trace_payload(trace_dir: str, source_id: str, doc_index: int, payload: dict, stage_summary: dict) -> None:
    trace_payload_path = os.path.join(trace_dir, f"{_safe_trace_filename(source_id, doc_index)}.json")
    trace_record = {
        "document_index": doc_index,
        "source_id": source_id,
        "trace": payload,
        "stage_summary": stage_summary,
    }
    try:
        with open(trace_payload_path, "w") as fp:
            json.dump(trace_record, fp, indent=2)
    except OSError as err:
        print(f"Failed to write trace file {trace_payload_path}: {err}")


def _build_stage_summary(stage_totals: Dict[str, List[float]]) -> dict:
    summary = {}
    for stage, values in stage_totals.items():
        total = sum(values)
        count = len(values)
        summary[stage] = {
            "documents": count,
            "total_resident_s": round(total, 6),
            "avg_resident_s": round(total / count, 6),
            "max_resident_s": round(max(values), 6),
            "min_resident_s": round(min(values), 6),
        }
    return summary


def _min_val(current: Optional[int], candidate: int) -> int:
    if current is None or candidate < current:
        return candidate
    return current


def _max_val(current: Optional[int], candidate: int) -> int:
    if current is None or candidate > current:
        return candidate
    return current
