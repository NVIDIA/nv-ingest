# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for batch and GPU inference NVTX ranges."""

from contextlib import contextmanager
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nvtx
import pytest


_BATCH_ACTOR_SCRIPT = r"""
import builtins
import functools
import nvtx
import pandas as pd

events = []
_real_import = builtins.__import__


def guard_torch_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch import blocked (remote CPU actor contract)", name=name)
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guard_torch_import


class RecordingAnnotate:
    def __init__(self, label, *, color):
        assert color == "blue"
        self.label = label

    def __call__(self, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            events.append(("push", self.label))
            try:
                return function(*args, **kwargs)
            finally:
                events.append(("pop",))

        return wrapped

    def __enter__(self):
        events.append(("push", self.label))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        events.append(("pop",))
        return False


nvtx.annotate = RecordingAnnotate

from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.ray_resource_hueristics import Resources
from nemo_retriever.operators.extract.pdf.split import PDFSplitActor, PDFSplitCPUActor
from nemo_retriever.operators.extract.pdf.extract import PDFExtractionActor, PDFExtractionCPUActor
from nemo_retriever.operators.extract.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.operators.extract.ocr.ocr import OCRActor
from nemo_retriever.operators.embed.operators import _BatchEmbedActor
from nemo_retriever.operators.extract.table.table_detection import TableStructureActor as TableStructureArchetype
from nemo_retriever.operators.extract.page_elements import cpu_actor
from nemo_retriever.operators.extract.ocr import cpu_ocr
from nemo_retriever.operators.embed import cpu_operator
from nemo_retriever.operators.extract.pdf import split as pdf_split
from nemo_retriever.operators.extract.pdf import extract as pdf_extract
from nemo_retriever.operators.extract.table import cpu_actor as table_cpu_actor
from nemo_retriever.operators.extract.table import gpu_actor as table_gpu_actor


def record_backend(stage):
    def backend(data, *args, **kwargs):
        events.append(("call", stage))
        return data

    return backend


pdf_split.split_pdf_batch = record_backend("pdf_split")
pdf_extract.pdf_extraction = record_backend("pdf_extract")
cpu_actor.detect_page_elements_v3 = record_backend("page_elements")
cpu_ocr.ocr_page_elements = record_backend("ocr")
cpu_operator.ensure_embedding_input_policy_for_batch = lambda kwargs, data: None
cpu_operator.embed_text_main_text_embed = record_backend("embedding")
table_cpu_actor.probe_endpoint = lambda *args, **kwargs: None
table_cpu_actor.table_structure_ocr_page_elements = record_backend("table_structure")
table_gpu_actor.table_structure_ocr_page_elements = record_backend("table_structure")


def check_ranges(actor_class, constructor_kwargs, stage, expect_startup=True):
    events.clear()
    actor = actor_class(**constructor_kwargs)
    actor.process(pd.DataFrame({"x": [1]}))
    expected = []
    if expect_startup:
        expected.extend(
            [
                ("push", f"nrl.batch::{stage}.startup"),
                ("pop",),
            ]
        )
    expected.extend(
        [
            ("push", f"nrl.batch::{stage}.batch"),
            ("call", stage),
            ("pop",),
        ]
    )
    assert events == expected, (stage, events)


def check_actor(archetype, expected_class, operator_kwargs, stage, expect_startup=True):
    resources = Resources(cpu_count=8, gpu_count=1)
    resolved_class = archetype.resolve_operator_class(resources, operator_kwargs=operator_kwargs)
    assert resolved_class is expected_class
    resolved_kwargs = archetype.variant_operator_kwargs(resolved_class, operator_kwargs)
    check_ranges(resolved_class, resolved_kwargs, stage, expect_startup)


from nemo_retriever.operators.extract.page_elements.cpu_actor import PageElementDetectionCPUActor
from nemo_retriever.operators.extract.ocr.cpu_ocr import OCRCPUActor
from nemo_retriever.operators.embed.cpu_operator import _BatchEmbedCPUActor
from nemo_retriever.operators.extract.table.cpu_actor import TableStructureCPUActor

check_actor(PDFSplitActor, PDFSplitCPUActor, {}, "pdf_split", expect_startup=False)
check_actor(PDFExtractionActor, PDFExtractionCPUActor, {}, "pdf_extract", expect_startup=False)

check_actor(
    PageElementDetectionActor,
    PageElementDetectionCPUActor,
    {"page_elements_invoke_url": "http://page-elements.invalid"},
    "page_elements",
)
check_actor(
    OCRActor,
    OCRCPUActor,
    {"ocr_invoke_url": "http://ocr.invalid"},
    "ocr",
)
check_actor(
    _BatchEmbedActor,
    _BatchEmbedCPUActor,
    {"params": EmbedParams(embed_invoke_url="http://embed.invalid")},
    "embedding",
)
check_actor(
    TableStructureArchetype,
    TableStructureCPUActor,
    {"table_structure_invoke_url": "http://table-structure.invalid"},
    "table_structure",
)
check_ranges(
    table_gpu_actor.TableStructureActor,
    {
        "table_structure_invoke_url": "http://table-structure.invalid",
        "ocr_invoke_url": "http://ocr.invalid",
    },
    "table_structure",
)
print("batch_actor_ranges_ok")
"""


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


def test_batch_actors_emit_stage_ranges() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    env = os.environ.copy()
    env.pop("NVIDIA_API_KEY", None)
    env.pop("NGC_API_KEY", None)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) + (os.pathsep + prev if prev else "")

    proc = subprocess.run(
        [sys.executable, "-c", _BATCH_ACTOR_SCRIPT],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n" f"exit={proc.returncode}"
    assert "batch_actor_ranges_ok" in proc.stdout


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
