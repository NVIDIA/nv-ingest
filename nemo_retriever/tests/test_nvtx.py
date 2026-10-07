# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for GPU inference capture markers at model call sites."""

from contextlib import contextmanager
from types import SimpleNamespace

import nvtx
import pytest


@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("fails", [False, True])
def test_embedding_capture_ranges(monkeypatch: pytest.MonkeyPatch, batch_size: int, fails: bool) -> None:
    torch = pytest.importorskip("torch")
    from nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder import LlamaNemotronEmbedVL1BV2Embedder

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
