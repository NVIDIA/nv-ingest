# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests aligned with documented Python snippets under ``nemo_retriever`` sources.

Covers:

- Every `` ```python`` fenced block in ``src/nemo_retriever/**/*.md`` (syntax / special cases)
- Runnable examples from module docstrings and README-style docs (graph, text embed, API helpers)
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import io
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import nemo_retriever


def _package_dir() -> Path:
    return Path(nemo_retriever.__file__).resolve().parent


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_markdown_python_blocks() -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    root = _package_dir()
    for md in sorted(root.rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="replace")
        for i, code in enumerate(re.findall(r"```python\n(.*?)```", text, re.DOTALL)):
            rel = md.relative_to(root).as_posix()
            blocks.append((f"{rel}#{i}", code))
    return blocks


_MD_BLOCKS = _iter_markdown_python_blocks()
_PUBLIC_RETRIEVER_DOCS = (
    "README.md",
    "docs/docs/extraction/vdbs.md",
    "examples/building_vdb_operator.ipynb",
    "examples/nemo_retriever_retriever_query_metadata_filter.ipynb",
    "nemo_retriever/README.md",
    "nemo_retriever/docs/cli/README.md",
    "nemo_retriever/retriever.md",
    "nemo_retriever/src/nemo_retriever/tools/evaluation/README.md",
    "nemo_retriever/src/nemo_retriever/common/vdb/README.md",
)
_UNSUPPORTED_DIRECT_RETRIEVER_KWARGS = frozenset(
    {
        "vdb",
        "lancedb_uri",
        "lancedb_table",
        "embedder",
        "embedding_endpoint",
        "local_query_embed_backend",
        "reranker",
    }
)


def _public_doc_path(root: Path, rel_path: str) -> Path | None:
    path = root / rel_path
    if path.exists():
        return path
    repo_only_doc = rel_path == "README.md" or rel_path.startswith(("docs/", "examples/"))
    package_only_image = not (root / "README.md").exists() and not (root / "docs").exists()
    if repo_only_doc and package_only_image:
        return None
    assert False, f"Expected public documentation file is missing: {rel_path}"
    return path


@pytest.mark.parametrize("block_id,code", _MD_BLOCKS, ids=[b[0] for b in _MD_BLOCKS])
def test_markdown_python_snippet_is_valid_syntax(block_id: str, code: str) -> None:
    """All in-tree Markdown ``python`` fences parse as Python except documented pseudocode."""
    ast.parse(code)


def _iter_public_retriever_doc_code() -> list[tuple[str, str]]:
    root = _repo_root()
    blocks: list[tuple[str, str]] = []
    for rel_path in _PUBLIC_RETRIEVER_DOCS:
        path = _public_doc_path(root, rel_path)
        if path is None:
            continue
        if path.suffix == ".ipynb":
            nb = json.loads(path.read_text(encoding="utf-8"))
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source") or []
                code = source if isinstance(source, str) else "".join(source)
                blocks.append((f"{rel_path}#cell-{i}", code))
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        for i, code in enumerate(re.findall(r"```python\n(.*?)```", text, re.DOTALL)):
            blocks.append((f"{rel_path}#python-{i}", code))
    return blocks


def _retriever_call_unsupported_kwargs(code: str) -> list[str]:
    tree = ast.parse(code)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_retriever = isinstance(func, ast.Name) and func.id == "Retriever"
        is_retriever = is_retriever or isinstance(func, ast.Attribute) and func.attr == "Retriever"
        if is_retriever:
            for kw in node.keywords:
                if kw.arg in _UNSUPPORTED_DIRECT_RETRIEVER_KWARGS:
                    found.append(str(kw.arg))
                if kw.arg is None and isinstance(kw.value, ast.Dict):
                    found.extend(
                        key.value
                        for key in kw.value.keys
                        if isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and key.value in _UNSUPPORTED_DIRECT_RETRIEVER_KWARGS
                    )
    return found


def test_public_retriever_examples_do_not_use_unsupported_constructor_kwargs() -> None:
    """Public direct ``Retriever(...)`` examples should not use kwargs that the constructor rejects."""
    violations = []
    for block_id, code in _iter_public_retriever_doc_code():
        try:
            unsupported_kwargs = _retriever_call_unsupported_kwargs(code)
        except SyntaxError:
            continue
        if unsupported_kwargs:
            violations.append(f"{block_id}: {', '.join(sorted(set(unsupported_kwargs)))}")

    assert not violations, "Unsupported kwargs in public direct Retriever(...) examples:\n" + "\n".join(violations)


def test_building_vdb_operator_notebook_creates_a_concrete_vdb() -> None:
    """The notebook's DuckDB example implements the current :class:`VDB` contract."""
    from nemo_retriever.common.vdb.adt_vdb import VDB

    pytest.importorskip("duckdb")

    repo_root = _repo_root()
    notebook = next(
        (
            candidate
            for candidate in (
                repo_root / "examples/building_vdb_operator.ipynb",
                repo_root / "nemo_retriever/examples/building_vdb_operator.ipynb",
            )
            if candidate.is_file()
        ),
        None,
    )
    if notebook is None:
        pytest.skip("The package-only image does not include notebook examples.")
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    class_source = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "class DuckDBVDB(VDB):" in "".join(cell["source"])
    )
    smoke_test_source = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "vdb = DuckDBVDB(vector_dim=3)" in "".join(cell["source"])
    )
    namespace: dict[str, Any] = {}
    exec(class_source, namespace)
    exec(smoke_test_source, namespace)
    vdb = namespace["vdb"]
    hits = namespace["hits"]

    assert isinstance(vdb, VDB)
    assert not type(vdb).__abstractmethods__
    assert [[hit["chunk_id"] for hit in query_hits] for query_hits in hits] == [["intro"]]
    assert "vector" not in hits[0][0]


def test_graph_readme_smallest_example() -> None:
    """``graph/README.md`` — single :class:`UDFOperator` on a :class:`Graph`."""
    from nemo_retriever.graph import Graph, UDFOperator

    def double(x: Any) -> Any:
        return x * 2

    graph = Graph()
    graph.add_root(UDFOperator(double, name="Double"))
    assert graph.execute(3) == [6]


def test_graph_readme_chain_example() -> None:
    """``graph/README.md`` — ``>>`` chaining returns a graph executable in one call."""
    from nemo_retriever.graph import UDFOperator

    graph = UDFOperator(lambda x: x + 1, name="AddOne") >> UDFOperator(lambda x: x * 10, name="TimesTen")
    assert graph.execute(5) == [60]


def test_graph_readme_multistage_normalization() -> None:
    """``graph/README.md`` — trim, lower, prefix chain."""
    from nemo_retriever.graph import UDFOperator

    graph = (
        UDFOperator(lambda text: text.strip(), name="Trim")
        >> UDFOperator(lambda text: text.lower(), name="Lower")
        >> UDFOperator(lambda text: f"normalized::{text}", name="Prefix")
    )
    assert graph.execute("  Hello World  ") == ["normalized::hello world"]


def test_graph_readme_custom_add_suffix_operator() -> None:
    """``graph/README.md`` — custom :class:`AbstractOperator` plus ``UDFOperator`` chain."""
    from typing import Any

    from nemo_retriever.graph import AbstractOperator, UDFOperator

    class AddSuffixOperator(AbstractOperator):
        def __init__(self, suffix: str = "_done") -> None:
            super().__init__()
            self.suffix = suffix

        def preprocess(self, data: Any, **kwargs: Any) -> Any:
            return str(data).strip()

        def process(self, data: Any, **kwargs: Any) -> Any:
            return f"{data}{self.suffix}"

        def postprocess(self, data: Any, **kwargs: Any) -> Any:
            return data

    graph = UDFOperator(lambda text: text.upper(), name="Upper") >> AddSuffixOperator("_READY")
    assert graph.execute("hello") == ["HELLO_READY"]


def test_graph_readme_uppercase_prefix_udf() -> None:
    """``graph/README.md`` — single ``UDFOperator`` with named callable."""
    from nemo_retriever.graph import Graph, UDFOperator

    def uppercase_and_prefix(text: str) -> str:
        return f"PROCESSED: {text.upper()}"

    graph = Graph()
    graph.add_root(UDFOperator(uppercase_and_prefix, name="UppercasePrefix"))
    assert graph.execute("hello world") == ["PROCESSED: HELLO WORLD"]


def test_graph_readme_numeric_chain() -> None:
    """``graph/README.md`` — multiply, add, finalize string."""
    from nemo_retriever.graph import UDFOperator

    graph = (
        UDFOperator(lambda x: x * 4, name="MultiplyByFour")
        >> UDFOperator(lambda x: x + 3, name="AddThree")
        >> UDFOperator(lambda x: f"{x}_done", name="Finalize")
    )
    assert graph.execute(2) == ["11_done"]


def test_graph_readme_recommended_imports() -> None:
    """``graph/README.md`` — public re-exports resolve."""
    from nemo_retriever.graph import (
        AbstractOperator,
        Graph,
        InprocessExecutor,
        Node,
        RayDataExecutor,
        UDFOperator,
    )

    assert AbstractOperator is not None
    assert Graph is not None
    assert Node is not None
    assert UDFOperator is not None
    assert InprocessExecutor is not None
    assert RayDataExecutor is not None


def test_main_text_embed_module_doc_example() -> None:
    """``text_embed/main_text_embed.py`` module docstring — local embedder path."""
    from nemo_retriever.models.inference.main_text_embed import create_text_embeddings_for_df

    df = pd.DataFrame([{"text": "hello", "metadata": {"source_path": "/tmp/a.pdf"}}])

    def local_embedder(texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    out_df, _info = create_text_embeddings_for_df(
        df,
        task_config={"embedder": local_embedder, "endpoint_url": None, "local_batch_size": 64},
    )
    assert out_df.loc[0, "embedding"] == [0.0, 1.0]


def test_clustering_doc_boxes_overlap_example() -> None:
    """``clustering.py`` docstring — close boxes overlap with threshold."""
    from nemo_retriever.common.api.util.image_processing.clustering import boxes_are_close_or_overlap

    box1 = (100, 100, 150, 150)
    box2 = (160, 110, 200, 140)
    assert boxes_are_close_or_overlap(box1, box2, threshold=10) is True


def test_clustering_doc_remove_superset_bboxes() -> None:
    """``clustering.py`` docstring — strict superset removed."""
    from nemo_retriever.common.api.util.image_processing.clustering import remove_superset_bboxes

    bboxes = [
        [0, 0, 5, 5],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
    ]
    assert remove_superset_bboxes(bboxes) == [[1, 1, 2, 2], [3, 3, 4, 4]]


def test_transforms_rgba_to_rgb_and_pad_image_doc() -> None:
    """``transforms.py`` docstring examples for RGBA conversion and padding."""
    from nemo_retriever.common.api.util.image_processing.transforms import pad_image, rgba_to_rgb_white_bg

    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, (100, 100, 4), dtype=np.uint8)
    rgb = rgba_to_rgb_white_bg(rgba)
    assert rgb.shape == (100, 100, 3)
    assert rgb.dtype == np.uint8

    rgba_float = rng.random((50, 50, 4), dtype=np.float32)
    rgb_float = rgba_to_rgb_white_bg(rgba_float)
    assert rgb_float.dtype == np.uint8

    image = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    padded_image, offsets = pad_image(image, target_width=1000, target_height=1000)
    assert padded_image.shape == (1000, 1000, 3)
    assert offsets == (100, 200)


def test_transforms_numpy_base64_roundtrip_doc() -> None:
    """``transforms.py`` — ``numpy_to_base64`` / ``base64_to_numpy`` as in docstrings."""
    from nemo_retriever.common.api.util.image_processing.transforms import base64_to_numpy, numpy_to_base64

    rng = np.random.default_rng(1)
    array = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    encoded_str = numpy_to_base64(array, format="PNG")
    assert isinstance(encoded_str, str)
    encoded_jpeg = numpy_to_base64(array, format="JPEG", quality=90)
    assert isinstance(encoded_jpeg, str)

    back = base64_to_numpy(encoded_str)
    assert back.shape == (100, 100, 3)


def test_transforms_base64_to_disk_and_save_image_doc(tmp_path: Path) -> None:
    """``transforms.py`` — write base64 bytes and optional format conversion."""
    from nemo_retriever.common.api.util.image_processing.transforms import base64_to_disk, save_image_to_disk

    rng = np.random.default_rng(2)
    arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    out_raw = tmp_path / "raw.png"
    assert base64_to_disk(image_b64, str(out_raw)) is True
    assert out_raw.stat().st_size > 0

    out_auto = tmp_path / "auto.png"
    assert save_image_to_disk(image_b64, str(out_auto), "auto") is True

    out_jpeg = tmp_path / "q85.jpg"
    assert save_image_to_disk(image_b64, str(out_jpeg), "JPEG", quality=85) is True


def test_schemas_mixins_lowercase_protocol_doc() -> None:
    """``mixins.py`` docstring — protocol fields lowercased."""
    from nemo_retriever.common.api.internal.schemas.mixins import LowercaseProtocolMixin

    class MyConfigSchema(LowercaseProtocolMixin):
        yolox_infer_protocol: str = ""
        ocr_infer_protocol: str = ""

    config = MyConfigSchema(yolox_infer_protocol="GRPC", ocr_infer_protocol="HTTP")
    assert config.yolox_infer_protocol == "grpc"
    assert config.ocr_infer_protocol == "http"


def test_pdf_exception_handler_doc_behaviour() -> None:
    """``pdf.py`` — decorator swallows errors; ``create_exception_tag`` returns validated metadata."""
    from nemo_retriever.common.api.util.exception_handlers.pdf import create_exception_tag, pdfium_exception_handler

    @pdfium_exception_handler("PDF Processing")
    def boom(path: str) -> list:
        raise RuntimeError("x")

    assert boom("example.pdf") == []

    tag = create_exception_tag("File not found", source_id="12345")
    assert isinstance(tag, list) and len(tag) == 1
    row = tag[0]
    assert row[0] is None
    meta = row[1]
    assert meta["error_metadata"]["error_msg"] == "File not found"
    assert meta["error_metadata"]["source_id"] == "12345"


def test_datetools_exception_handler_doc_behaviour() -> None:
    """``converters.py`` (exception_handlers) — invalid dates yield ISO fallback."""
    from datetime import datetime

    from nemo_retriever.common.api.util.exception_handlers.converters import datetools_exception_handler

    @datetools_exception_handler
    def parse_date(date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y-%m-%d")

    assert parse_date("2024-08-22") == datetime(2024, 8, 22, 0, 0)
    fallback = parse_date("not-a-date")
    assert isinstance(fallback, str)
    assert len(fallback) >= 10


@pytest.mark.skipif(
    importlib.util.find_spec("langdetect") is None,
    reason="langdetect not installed",
)
def test_langdetect_exception_handler_with_real_detect() -> None:
    """``detectors.py`` — when langdetect works, non-empty text returns a language enum."""
    import langdetect

    from nemo_retriever.common.api.internal.enums.common import LanguageEnum
    from nemo_retriever.common.api.util.exception_handlers.detectors import langdetect_exception_handler

    @langdetect_exception_handler
    def detect_language(text: str):
        return LanguageEnum(langdetect.detect(text))

    assert detect_language("This is a test sentence.") == LanguageEnum.EN
    assert detect_language("") == LanguageEnum.UNKNOWN


def test_traceable_func_doc_example() -> None:
    """``tagging.py`` — ``traceable_func`` fills ``trace_info`` entry/exit keys."""
    from nemo_retriever.common.api.internal.primitives.tracing.tagging import traceable_func

    @traceable_func(trace_name="pdf_extractor::{model_name}")
    def extract_pdf(model_name: str) -> None:
        return None

    trace_info: dict[str, Any] = {}
    extract_pdf("my_model", trace_info=trace_info)
    assert any(k.startswith("trace::entry::pdf_extractor::my_model") for k in trace_info)
    assert any(k.startswith("trace::exit::pdf_extractor::my_model") for k in trace_info)


def test_set_trace_timestamps_with_parent_context_doc() -> None:
    """``tagging.py`` docstring — keys gain parent namespace."""
    from datetime import datetime

    from nemo_retriever.common.api.internal.primitives.tracing.tagging import set_trace_timestamps_with_parent_context

    ts1 = datetime(2024, 1, 1, 12, 0, 0)
    ts2 = datetime(2024, 1, 1, 12, 0, 1)
    execution_trace_log = {"trace::entry::yolox_inference": ts1, "trace::exit::yolox_inference": ts2}
    control_message = MagicMock()
    set_trace_timestamps_with_parent_context(control_message, execution_trace_log, "pdf_extractor", logger=None)
    calls = {c.args[0]: c.args[1] for c in control_message.set_timestamp.call_args_list}
    assert calls["trace::entry::pdf_extractor::yolox_inference"] == ts1
    assert calls["trace::exit::pdf_extractor::yolox_inference"] == ts2


def test_graph_pipeline_registry_register_docstring_pattern() -> None:
    """``graph_pipeline_registry.py`` ``register`` docstring — decorator registers a factory."""
    from nemo_retriever.graph import Graph, GraphPipelineRegistry, UDFOperator

    reg = GraphPipelineRegistry()

    @reg.register("doc-test-pipeline", description="from docstring Example", overwrite=True)
    def _build() -> Any:
        return Graph() >> UDFOperator(lambda x: x, name="Id")

    assert "doc-test-pipeline" in reg.list_names()


def test_evaluation_readme_protocol_stubs_parse() -> None:
    """``evaluation/README.md`` — Protocol example bodies are syntactically valid."""
    from nemo_retriever.models.llm.types import AnswerJudge, GenerationResult, JudgeResult, LLMClient, RetrievalResult
    from nemo_retriever.models.llm.types import RetrieverStrategy

    class MyRetriever:
        def retrieve(self, query: str, top_k: int) -> RetrievalResult:
            chunks = ["stub"]
            return RetrievalResult(chunks=chunks, metadata=[])

    class MyClient:
        def generate(self, query: str, chunks: list[str]) -> GenerationResult:
            return GenerationResult(answer="stub", latency_s=0.0, model="my-model")

    class MyJudge:
        def judge(self, query: str, reference: str, candidate: str) -> JudgeResult:
            return JudgeResult(score=1, reasoning="ok")

    assert isinstance(MyRetriever(), RetrieverStrategy)
    assert isinstance(MyClient(), LLMClient)
    assert isinstance(MyJudge(), AnswerJudge)
