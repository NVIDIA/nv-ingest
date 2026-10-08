# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ingest (extract → embed → VDB) and Retriever query on ``multimodal_test.pdf``."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _multimodal_pdf() -> Path:
    return REPO_ROOT / "data" / "multimodal_test.pdf"


def _env_strip(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return None
    s = v.strip()
    return s or None


def _remote_nim_config_from_env() -> dict | None:
    """If ``PAGE_ELEMENTS_INVOKE_URL`` is set (e.g. GitHub Actions), use hosted NIMs; else local defaults."""
    page_url = _env_strip("PAGE_ELEMENTS_INVOKE_URL")
    if not page_url:
        return None
    return {
        "page_elements_invoke_url": page_url,
        "ocr_invoke_url": _env_strip("OCR_INVOKE_URL"),
        "table_structure_invoke_url": _env_strip("TABLE_STRUCTURE_INVOKE_URL"),
    }


def _embed_kwargs_from_env() -> dict | None:
    embed_url = _env_strip("EMBED_INVOKE_URL")
    if not embed_url:
        return None
    out: dict = {"embed_invoke_url": embed_url}
    model = _env_strip("EMBED_MODEL_NAME")
    if model:
        out["embed_model_name"] = model
        out["model_name"] = model
    return out


@pytest.fixture
def multimodal_pdf_path() -> Path:
    path = _multimodal_pdf()
    if not path.is_file():
        pytest.skip(f"Fixture PDF not found (clone repo data/): {path}")
    return path


@pytest.mark.integration
def test_ingest_multimodal_pdf_embed_vdb_then_retriever_query(multimodal_pdf_path: Path) -> None:
    """Mirror ``nv-ingest/test.py``: full graph ingest then a semantic query against the default VDB."""
    from nemo_retriever import create_ingestor
    from nemo_retriever.graph.retriever import Retriever

    pdf = str(multimodal_pdf_path.resolve())
    remote = _remote_nim_config_from_env()
    embed_kw = _embed_kwargs_from_env()

    if remote:
        if not embed_kw:
            pytest.skip("PAGE_ELEMENTS_INVOKE_URL is set but EMBED_INVOKE_URL is missing; need both for CPU/CI.")
        extract_kw = {k: v for k, v in remote.items() if v}
        ingestor = create_ingestor().files([pdf]).extract(**extract_kw).embed(**embed_kw).vdb_upload()
    else:
        ingestor = create_ingestor().files([pdf]).extract().embed().vdb_upload()

    results = ingestor.ingest()

    assert isinstance(results, pd.DataFrame)
    assert len(results) >= 1

    query = "Given their activities, which animal is responsible for the typos in my documents?"
    if embed_kw:
        hits = Retriever(run_mode="service", embed_kwargs=embed_kw, top_k=10).query(query)
    else:
        hits = Retriever().query(query)

    assert isinstance(hits, list)
    assert len(hits) >= 1
    for h in hits:
        assert "text" in h
        assert len(str(h["text"]).strip()) > 0

    joined = " ".join(str(h.get("text", "")) for h in hits).lower()
    assert "cat" in joined or "laptop" in joined or "animal" in joined
