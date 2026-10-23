# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.common.vdb.records import _client_record_from_graph_row, _derive_fidelity


def _fidelity_of(row: dict) -> object:
    rec = _client_record_from_graph_row(row)
    assert rec is not None
    return rec["metadata"]["content_metadata"].get("fidelity")


def _row(content_type, *, needs_ocr=None, subtype=None) -> dict:
    meta: dict = {}
    if needs_ocr is not None:
        meta["needs_ocr_for_text"] = needs_ocr
    cm: dict = {"page_number": 1}
    if subtype is not None:
        cm["subtype"] = subtype
    meta["content_metadata"] = cm
    return {"text": "x", "embedding": [0.1, 0.2], "metadata": meta, "_content_type": content_type}


def test_derive_fidelity_pure_mapping() -> None:
    assert _derive_fidelity("text", {}, {}) == "verbatim"
    assert _derive_fidelity("text", {"needs_ocr_for_text": True}, {}) == "ocr"
    assert _derive_fidelity("image", {}, {}) == "vlm_caption"
    assert _derive_fidelity("image", {}, {"subtype": "page_image"}) == "ocr"
    assert _derive_fidelity("table", {}, {}) == "ocr"
    assert _derive_fidelity("chart_caption", {}, {}) == "ocr"
    assert _derive_fidelity("audio", {}, {}) == "transcribed"
    assert _derive_fidelity("video", {}, {}) == "transcribed"
    assert _derive_fidelity("", {}, {}) is None
    assert _derive_fidelity("mystery", {}, {}) is None


def test_fidelity_stamped_into_stored_record() -> None:
    assert _fidelity_of(_row("text")) == "verbatim"
    assert _fidelity_of(_row("text", needs_ocr=True)) == "ocr"
    assert _fidelity_of(_row("image")) == "vlm_caption"
    assert _fidelity_of(_row("image", subtype="page_image")) == "ocr"
    assert _fidelity_of(_row("table")) == "ocr"
    assert _fidelity_of(_row("audio")) == "transcribed"
