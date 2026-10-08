# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset entry / run-config loaders for skill_eval.

Reads an agent-eval manifest (JSON list). Each manifest entry carries the
original query, the paraphrased prompt (under ``sdg_prompt_candidates``),
the ground-truth pages (with ``doc_id`` + ``page_number_in_doc``), the
ground-truth answer, and a per-domain prompt taxonomy. The manifest format
is upstream of this module; this loader is dataset-agnostic. See
:func:`load_eval_manifest`.
"""

from __future__ import annotations

import json
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import yaml


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object at top-level: {path}")
    return data


class GroundTruthPage(BaseModel):
    doc_id: str
    page_number: int
    score: int = 1


class DatasetEntry(BaseModel):
    entry_id: int
    query_id: str
    taxonomy_slot_id: str
    original_query: str
    paraphrased_prompt: str
    ground_truth_pages: list[GroundTruthPage]
    ground_truth_answer: str = ""
    domain: str = ""
    domain_label: str = ""


def _select_prompt(candidates: list[dict[str, Any]], selected_variant: int | None) -> str:
    """Pick the chosen paraphrased prompt from sdg_prompt_candidates."""
    if not candidates:
        return ""
    if selected_variant is not None:
        for c in candidates:
            if c.get("variant_id") == selected_variant:
                return _normalize_slash_command(str(c.get("prompt") or ""))
    return _normalize_slash_command(str(candidates[0].get("prompt") or ""))


def _normalize_slash_command(prompt: str) -> str:
    """Rewrite SDG-generated slash commands to this project's actual skill name.

    The agent_scenario_manifest contains slash-command scenarios using made-up
    aliases (``/vidore-ingest``, ``/vidore``, ``/vidore_hr``); the real skill
    that ships with this repo is ``nemo-retriever``. Rewriting at load time
    avoids editing the upstream manifest (frozen baseline) while making the
    slash_ingest / slash_retrieval scenarios actually exercisable. The token
    boundary after the alias is preserved so the trailing args/query carry
    over verbatim.
    """
    s = prompt.lstrip()
    if not s.startswith("/"):
        return prompt
    # Order matters: rewrite ``/vidore-ingest`` (which carries the ingest
    # subcommand intent) before the bare ``/vidore`` prefix.
    rewrites = [
        ("/vidore-ingest ", "/nemo-retriever ingest "),
        ("/vidore_hr ", "/nemo-retriever "),
        ("/vidore ", "/nemo-retriever "),
    ]
    for old, new in rewrites:
        if s.startswith(old):
            return new + s[len(old) :]
    return prompt


def load_eval_manifest(path: Path) -> list[DatasetEntry]:
    """Load ``eval_manifest.json`` into ``DatasetEntry`` records.

    ``entry_id`` is the 1-indexed position in the manifest; ``query_id`` is the
    manifest's ``primary_eval_id`` (e.g. ``<domain>:<n>:<variant>``). The
    selected paraphrased prompt comes from ``sdg_prompt_candidates.candidates``
    (matching ``sdg_prompt_validation.selected_variant_id`` when present).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list, got {type(data).__name__}")

    entries: list[DatasetEntry] = []
    for idx, item in enumerate(data, start=1):
        if item.get("prompt_export_status") not in (None, "exported"):
            # Skip entries the SDG pipeline did not finalize.
            continue
        domain = str(item.get("domain") or "")
        task_code = str((item.get("task_family") or {}).get("code") or "?")
        taxonomy = item.get("prompt_taxonomy") or {}
        domain_label = str(taxonomy.get("domain_label") or "")
        # Accept both the legacy SDG keys and the newer scenario-format keys.
        candidates = (item.get("sdg_prompt_candidates") or item.get("scenario_prompt_candidates") or {}).get(
            "candidates"
        ) or []
        selected_variant = (item.get("sdg_prompt_validation") or item.get("scenario_prompt_validation") or {}).get(
            "selected_variant_id"
        )
        prompt = _select_prompt(candidates, selected_variant)
        if not prompt:
            # No usable paraphrased prompt — skip.
            continue

        pages: list[GroundTruthPage] = []
        for p in item.get("relevant_pages") or []:
            doc_id = p.get("doc_id")
            page = p.get("page_number_in_doc")
            if doc_id is None or page is None:
                continue
            pages.append(GroundTruthPage(doc_id=str(doc_id), page_number=int(page), score=int(p.get("score") or 1)))

        entries.append(
            DatasetEntry(
                entry_id=idx,
                query_id=str(item.get("primary_eval_id") or item.get("eval_base_id") or idx),
                taxonomy_slot_id=task_code,
                original_query=str(item.get("original_query") or ""),
                paraphrased_prompt=prompt,
                ground_truth_pages=pages,
                ground_truth_answer=str(item.get("answer") or ""),
                domain=domain,
                domain_label=domain_label,
            )
        )
    return entries


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(str(pkg_files("nemo_retriever.skill_eval").joinpath("configs/skill_eval.yaml")))
    return _read_yaml_mapping(Path(path))
