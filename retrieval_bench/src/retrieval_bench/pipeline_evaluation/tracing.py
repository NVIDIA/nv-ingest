# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helpers for per-query trace caching and logging.

Traces are written as one JSON file per query:
  traces/<trace_run_name>/<dataset_dir>/<query_id>.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    """
    Make a filesystem-friendly string.

    Keeps [A-Za-z0-9._-], replaces anything else with '_'.

    Important: we intentionally preserve double-underscore separators (e.g. `A__B__C`)
    used in run naming for readability.
    """
    value = (value or "").strip()
    if not value:
        return "unnamed"
    value = value.replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    # Preserve `__` separators but avoid pathological underscore runs produced by replacement.
    value = re.sub(r"_{3,}", "__", value).strip("_")
    return value or "unnamed"


def model_id_short(model_id: Optional[str]) -> Optional[str]:
    if not model_id:
        return None
    return str(model_id).split("/")[-1]


def default_trace_run_name(pipeline: Any) -> str:
    cls = pipeline.__class__.__name__
    mid = getattr(pipeline, "model_id", None)
    mid_short = model_id_short(mid)
    llm_mid = getattr(pipeline, "llm_model", None)
    llm_short = model_id_short(llm_mid)

    # Prefer: <PipelineClass>__<retriever_short>__<llm_short>
    parts = [cls]
    if mid_short:
        parts.append(mid_short)
    if llm_short:
        parts.append(llm_short)
    return _slugify("__".join(parts))


def dataset_trace_dir(dataset_name: str, split: str = "test", language: Optional[str] = None) -> str:
    """
    Trace subdirectory name for a dataset.

    Simplified by design: we use a stable filesystem-friendly identifier.

    - ViDoRe: keep existing short-name behavior:
      'vidore/vidore_v3_finance_en' -> 'vidore_v3_finance_en'
    - BRIGHT: include the dataset prefix to avoid collisions:
      'bright/biology' -> 'bright__biology'

    (split/language are intentionally ignored to keep paths stable and simple.)
    """
    ds = str(dataset_name or "unknown_dataset").strip()
    parts = [p for p in ds.split("/") if p]
    if len(parts) >= 2 and parts[0].lower() == "bright":
        return _slugify(f"bright__{parts[1]}")
    return _slugify(parts[-1] if parts else "unknown_dataset")


def trace_path(
    traces_dir: str,
    trace_run_name: str,
    dataset_dir: str,
    query_id: str,
) -> Path:
    return Path(traces_dir) / _slugify(trace_run_name) / _slugify(dataset_dir) / f"{_slugify(query_id)}.json"


def load_trace_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with open(path, "r") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        logger.debug("Failed to load trace file %s", path, exc_info=True)
        return None


def extract_run_and_time_ms(trace_obj: Dict[str, Any]) -> Optional[Tuple[Dict[str, float], float]]:
    """
    Returns (run, retrieval_time_ms) if trace has required fields; otherwise None.
    """
    run = trace_obj.get("run", None)
    t = trace_obj.get("retrieval_time_milliseconds", None)
    if not isinstance(run, dict):
        return None
    if not isinstance(t, (int, float)):
        return None
    return run, float(t)


def write_trace_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def write_query_trace(
    *,
    traces_dir: str,
    trace_run_name: str,
    dataset: str,
    dataset_dir: str,
    query_id: str,
    pipeline_class: str,
    model_id: Optional[str],
    retrieval_time_milliseconds: Optional[float],
    run: Optional[Dict[str, float]],
    split: str = "test",
    language: Optional[str] = None,
    query_ids_selector: Optional[str] = None,
    pipeline_trace: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a per-query trace JSON file in the canonical evaluator format.

    This is intended for pipelines that want to write traces incrementally (per query)
    while preserving evaluator caching semantics (which require `run` + numeric time).
    """
    payload: Dict[str, Any] = {
        "query_id": query_id,
        "dataset": dataset,
        "dataset_dir": dataset_dir,
        "split": split,
        "language": language,
        "query_ids_selector": query_ids_selector,
        "trace_run_name": trace_run_name,
        "pipeline_class": pipeline_class,
        "model_id": model_id,
        "retrieval_time_milliseconds": retrieval_time_milliseconds,
        "run": run,
    }
    if isinstance(pipeline_trace, dict):
        payload["pipeline_trace"] = pipeline_trace

    p = trace_path(traces_dir, trace_run_name, dataset_dir, query_id)
    write_trace_file(p, payload)
    return p
