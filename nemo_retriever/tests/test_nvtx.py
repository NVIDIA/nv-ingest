# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for GPU inference capture markers at model call sites."""

from contextlib import contextmanager
from types import SimpleNamespace

import nvtx
import pytest


@pytest.fixture
def capture_ranges(monkeypatch: pytest.MonkeyPatch):
    events = []

    @contextmanager
    def annotate(label: str, *, color: str):
        assert color == "blue"
        events.append(("push", label))
        try:
            yield
        finally:
            events.append(("pop",))

    monkeypatch.setattr(nvtx, "annotate", annotate)
    return events


@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("fails", [False, True])
def test_embedding_capture_ranges(capture_ranges, batch_size: int, fails: bool) -> None:
    torch = pytest.importorskip("torch")
    from nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder import LlamaNemotronEmbedVL1BV2Embedder

    events = capture_ranges
    error = ValueError("inference failed")
    inputs = ["document"] * batch_size

    def encode_documents(*, texts):
        assert texts == inputs
        events.append(("inference",))
        if fails:
            raise error
        return torch.tensor([[3.0, 4.0]] * batch_size)

    embedder = LlamaNemotronEmbedVL1BV2Embedder()
    embedder._model = SimpleNamespace(encode_documents=encode_documents)
    if fails:
        with pytest.raises(ValueError) as exc:
            embedder.embed(inputs)
        assert exc.value is error
    else:
        result = embedder.embed(inputs)
        torch.testing.assert_close(result, torch.tensor([[0.6, 0.8]] * batch_size))
    assert events == [
        ("push", "gpu_inference"),
        ("push", f"LlamaNemotronEmbedVL1B | bs={batch_size} | mode=doc_text"),
        ("inference",),
        ("pop",),
        ("pop",),
    ]


@pytest.mark.parametrize("batch_size", [-1, 8])
@pytest.mark.parametrize("fails", [False, True])
def test_legacy_inference_range_remains_compatible(capture_ranges, batch_size: int, fails: bool) -> None:
    from nemo_retriever.common.nvtx import gpu_inference_range

    error = ValueError("original workload error")

    def inference():
        with gpu_inference_range("legacy", batch_size, mode="decode"):
            capture_ranges.append(("inference",))
            if fails:
                raise error
            return "result"

    with pytest.warns(DeprecationWarning, match="deprecated; use nvtx.annotate") as warning:
        if fails:
            with pytest.raises(ValueError) as exc:
                inference()
            assert exc.value is error
        else:
            assert inference() == "result"
    assert warning[0].filename == __file__
    label = "legacy | mode=decode" if batch_size < 0 else "legacy | bs=8 | mode=decode"
    assert capture_ranges == [
        ("push", "gpu_inference"),
        ("push", label),
        ("inference",),
        ("pop",),
        ("pop",),
    ]
