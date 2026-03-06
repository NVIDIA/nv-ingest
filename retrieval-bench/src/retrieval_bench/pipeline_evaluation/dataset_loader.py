# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset loader for vidore v3 benchmark datasets.

Handles downloading and preparing vidore v3 datasets from HuggingFace,
including queries, corpus images, and ground truth relevance judgments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from vidore_benchmark.pipeline_evaluation.dataset_loader import (
    get_available_datasets as _upstream_get_available_datasets,
    load_vidore_dataset as _upstream_load_vidore_dataset,
)


BRIGHT_TASKS: tuple[str, ...] = (
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
)


def _repo_root() -> Path:
    # <repo>/src/retrieval_bench/pipeline_evaluation/dataset_loader.py
    return Path(__file__).resolve().parents[3]


def _preferred_bright_cache_dir() -> Optional[str]:
    """
    Prefer using the sibling BRIGHT repo cache if available.

    Workflow assumption: BRIGHT repo lives at ../BRIGHT relative to this repo root.
    """
    bright_cache = _repo_root().parent / "BRIGHT" / "cache"
    if bright_cache.exists() and bright_cache.is_dir():
        return str(bright_cache)
    return None


def _load_bright_split(
    *,
    task: str,
    split: str = "test",
    language: Optional[str] = None,
) -> Tuple[
    List[str],
    List[str],
    List[str],
    List[Any],
    List[str],
    Dict[str, Dict[str, int]],
    Dict[str, str],
    Dict[str, List[str]],
]:
    """
    Load one BRIGHT task as a dataset compatible with our pipeline evaluator.

    Default setting only:
    - examples/config: 'examples'
    - corpus/config: 'documents'
    - qrels key: 'gold_ids'
    """
    if split != "test":
        raise ValueError("BRIGHT datasets only support split='test' in this integration.")

    if language and str(language).strip().lower() not in ("english", "en"):
        raise ValueError("BRIGHT datasets are English-only in this integration (use --language english or omit).")

    task = str(task).strip()
    if task not in BRIGHT_TASKS:
        raise ValueError(f"Unknown BRIGHT task '{task}'. Expected one of: {', '.join(BRIGHT_TASKS)}")

    cache_dir = _preferred_bright_cache_dir()

    last_err: Optional[Exception] = None
    examples_ds = None
    docs_ds = None
    for ds_name in ("xlangai/bright", "xlangai/BRIGHT"):
        try:
            examples_ds = load_dataset(ds_name, "examples", cache_dir=cache_dir)[task]
            docs_ds = load_dataset(ds_name, "documents", cache_dir=cache_dir)[task]
            last_err = None
            break
        except Exception as e:
            last_err = e
            examples_ds = None
            docs_ds = None
            continue

    if examples_ds is None or docs_ds is None:
        raise RuntimeError(f"Failed to load BRIGHT task '{task}' from HuggingFace: {last_err}") from last_err

    corpus_ids: List[str] = []
    corpus_images: List[Any] = []
    corpus_texts: List[str] = []
    for dp in docs_ds:
        did = str(dp.get("id", ""))
        corpus_ids.append(did)
        corpus_images.append(None)
        corpus_texts.append(str(dp.get("content", "")))

    query_ids: List[str] = []
    queries: List[str] = []
    qrels: Dict[str, Dict[str, int]] = {}
    query_languages: Dict[str, str] = {}
    excluded_ids_by_query: Dict[str, List[str]] = {}

    for e in examples_ds:
        qid = str(e.get("id", ""))
        q = str(e.get("query", ""))
        excluded = e.get("excluded_ids", None)
        if not isinstance(excluded, list):
            excluded = ["N/A"]
        excluded_list = [str(x) for x in excluded]

        gold = e.get("gold_ids", None)
        if not isinstance(gold, list):
            gold = []
        gold_ids = [str(x) for x in gold]

        overlap = set(excluded_list).intersection(set(gold_ids))
        overlap.discard("N/A")
        if overlap:
            raise ValueError(f"BRIGHT data error: excluded_ids overlaps gold_ids for query_id={qid}: {sorted(overlap)}")

        query_ids.append(qid)
        queries.append(q)
        query_languages[qid] = "english"
        excluded_ids_by_query[qid] = excluded_list

        qrels[qid] = {gid: 1 for gid in gold_ids}

    if not queries:
        raise ValueError(f"No queries found in BRIGHT task '{task}'.")
    if not corpus_texts:
        raise ValueError(f"No corpus documents found in BRIGHT task '{task}'.")
    if not any(v for v in qrels.values()):
        raise ValueError(f"No relevance judgments found in BRIGHT task '{task}'.")

    return query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages, excluded_ids_by_query


def load_vidore_dataset(dataset_name: str, split: str = "test", language: str = None) -> Tuple[
    List[str],
    List[str],
    List[str],
    List[Any],
    List[str],
    Dict[str, Dict[str, int]],
    Dict[str, str],
    Dict[str, List[str]],
]:
    """
    Load a dataset for the pipeline evaluator.

    ViDoRe datasets are delegated to upstream vidore-benchmark loader.
    BRIGHT tasks are handled locally and include excluded-ids semantics.
    """
    if str(dataset_name).startswith("bright/"):
        task = str(dataset_name).split("/", 1)[1]
        return _load_bright_split(task=task, split=split, language=language)

    query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages = _upstream_load_vidore_dataset(
        dataset_name=dataset_name,
        split=split,
        language=language,
    )
    excluded_ids_by_query = {str(qid): ["N/A"] for qid in query_ids}
    return query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages, excluded_ids_by_query


def get_available_datasets() -> List[str]:
    """
    Get list of available vidore v3 datasets.

    Returns:
        List of dataset names that can be used with load_vidore_dataset()
    """
    datasets = list(_upstream_get_available_datasets())
    for name in (f"bright/{t}" for t in BRIGHT_TASKS):
        if name not in datasets:
            datasets.append(name)
    return datasets
