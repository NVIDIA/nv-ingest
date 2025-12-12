import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - best effort import
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover
    pdfium = None

NS_IN_SECOND = 1_000_000_000
PDFIUM_STAGE_PREFIX = "pdf_extractor::pdf_extraction::"
PDFIUM_STAGE_SUFFIXES = ("render_page", "bitmap_to_numpy", "scale_image", "pad_image")
PDFIUM_STAGE_NAMES = {f"{PDFIUM_STAGE_PREFIX}{suffix}" for suffix in PDFIUM_STAGE_SUFFIXES}
_DEDUP_SUFFIX_RE = re.compile(r"_(\d+)$")
_PAGE_COUNT_CACHE: Dict[str, int] = {}


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
    nested_stage_totals = defaultdict(list)
    pdfium_doc_breakdown: List[dict] = []
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

        nested_stage_ns, nested_stage_counts, _ = _collect_pdfium_nested_spans(trace_payload)
        for stage, duration_ns in nested_stage_ns.items():
            resident_s = _ns_to_seconds(duration_ns) or 0.0
            if resident_s <= 0:
                continue
            nested_stage_totals[stage].append(round(resident_s, 6))
        if nested_stage_ns:
            stage_seconds = {
                stage: round(_ns_to_seconds(duration_ns) or 0.0, 6) for stage, duration_ns in nested_stage_ns.items()
            }
            stage_counts = {stage: nested_stage_counts.get(stage, 0) for stage in nested_stage_ns}
            inferred_page_count = stage_counts.get(f"{PDFIUM_STAGE_PREFIX}render_page", 0)
            true_page_count = _resolve_page_count(source_id, inferred_page_count)
            pdfium_doc_breakdown.append(
                {
                    "document_index": doc_index,
                    "source_id": source_id,
                    "page_count": true_page_count,
                    "stage_seconds": stage_seconds,
                    "stage_counts": stage_counts,
                }
            )

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

    if not stage_totals and not nested_stage_totals:
        return None

    stage_summary = _build_stage_summary(stage_totals)
    nested_summary = _build_stage_summary(nested_stage_totals) if nested_stage_totals else {}
    nested_samples = {stage: values[:] for stage, values in nested_stage_totals.items()} if nested_stage_totals else {}

    result = {
        "documents": len(document_totals),
        "output_dir": trace_dir,
        "stage_totals": stage_summary,
        "document_totals": document_totals,
    }
    if nested_summary:
        result["nested_stage_totals"] = nested_summary
    if nested_samples:
        result["nested_stage_samples"] = nested_samples
    if pdfium_doc_breakdown:
        result["pdfium_document_breakdown"] = pdfium_doc_breakdown
    return result


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


def _collect_pdfium_nested_spans(trace_payload: dict) -> Tuple[Dict[str, int], Dict[str, int], int]:
    durations: Dict[str, int] = defaultdict(int)
    counts: Dict[str, int] = defaultdict(int)
    for key, entry in trace_payload.items():
        if not key.startswith("trace::entry::"):
            continue
        stage = key.replace("trace::entry::", "", 1)
        normalized = _strip_dedupe_suffix(stage)
        if normalized not in PDFIUM_STAGE_NAMES:
            continue
        exit_key = f"trace::exit::{stage}"
        exit_value = trace_payload.get(exit_key)
        if exit_value is None:
            continue
        try:
            duration_ns = int(exit_value) - int(entry)
        except (TypeError, ValueError):
            continue
        if duration_ns <= 0:
            continue
        durations[normalized] += duration_ns
        counts[normalized] += 1

    page_count = counts.get(f"{PDFIUM_STAGE_PREFIX}render_page", 0)
    return durations, counts, page_count


def _resolve_page_count(source_id: Optional[str], fallback: int) -> int:
    if not source_id:
        return fallback
    if source_id in _PAGE_COUNT_CACHE:
        return _PAGE_COUNT_CACHE[source_id]
    if pdfium is None:
        return fallback
    path = Path(source_id)
    if not path.exists():
        return fallback
    try:
        doc = pdfium.PdfDocument(str(path))
        page_count = len(doc)
        doc.close()
    except Exception:
        page_count = fallback
    if page_count <= 0:
        page_count = fallback
    _PAGE_COUNT_CACHE[source_id] = page_count
    return page_count


def _strip_dedupe_suffix(stage: str) -> str:
    return _DEDUP_SUFFIX_RE.sub("", stage)


def _min_val(current: Optional[int], candidate: int) -> int:
    if current is None or candidate < current:
        return candidate
    return current


def _max_val(current: Optional[int], candidate: int) -> int:
    if current is None or candidate > current:
        return candidate
    return current
