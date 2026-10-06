# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _front_page_typical_use_python_blocks() -> list[tuple[int, str]]:
    readme = _repo_root() / "README.md"
    if not readme.is_file():
        pytest.skip("front-page README.md is unavailable in this test layout")

    text = readme.read_text(encoding="utf-8")
    match = re.search(
        r"## Typical Use\n(?P<section>.*?)\n## Documentation Resources",
        text,
        re.DOTALL,
    )
    assert match is not None, "README.md is missing the Typical Use section"

    blocks: list[tuple[int, str]] = []
    for index, code in enumerate(re.findall(r"```python\n(.*?)```", match.group("section"), re.DOTALL)):
        # The middle README block is doctest-style inspection output, not an
        # executable setup/query snippet.
        if ">>>" not in code:
            blocks.append((index, code))
    return blocks


class _FakeIngestor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def files(self, documents: list[str]) -> "_FakeIngestor":
        self.calls.append(("files", documents))
        return self

    def urls(self, urls: list[str]) -> "_FakeIngestor":
        self.calls.append(("urls", urls))
        return self

    def extract(self, **kwargs: Any) -> "_FakeIngestor":
        self.calls.append(("extract", kwargs))
        return self

    def embed(self) -> "_FakeIngestor":
        self.calls.append(("embed", None))
        return self

    def vdb_upload(self) -> "_FakeIngestor":
        self.calls.append(("vdb_upload", None))
        return self

    def ingest(self, *, return_failures: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list]:
        self.calls.append(("ingest", {"return_failures": return_failures}))
        result = pd.DataFrame(
            [
                {
                    "page_number": 1,
                    "text": "Cat Jumping onto a laptop In a home office",
                    "path": "data/multimodal_test.pdf",
                },
                {
                    "page_number": 1,
                    "text": "| Animal | Activity | Place |\n| Cat | Jumping onto a laptop | In a home office |",
                    "path": "data/multimodal_test.pdf",
                },
            ]
        )
        return (result, []) if return_failures else result


def test_front_page_typical_use_snippets_execute_through_query(monkeypatch, capsys) -> None:
    """Run the executable front-page README snippets through query/answer.

    The harness keeps CI lightweight by faking the expensive ingestion,
    retrieval, and hosted LLM calls while executing the public imports and
    snippet wiring exactly as documented.
    """
    import nemo_retriever

    retriever_module = importlib.import_module("nemo_retriever.graph.retriever")
    blocks = _front_page_typical_use_python_blocks()
    assert len(blocks) == 3
    assert "create_ingestor" in blocks[0][1]
    assert ".urls(urls)" in blocks[1][1]
    assert "OpenAI" in blocks[2][1]

    ingestors: list[_FakeIngestor] = []
    retriever_init_calls: list[dict[str, Any]] = []
    retriever_queries: list[str] = []
    llm_requests: list[dict[str, Any]] = []

    def fake_create_ingestor(**_kwargs: Any) -> _FakeIngestor:
        ingestor = _FakeIngestor()
        ingestors.append(ingestor)
        return ingestor

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_init_calls.append(kwargs)

        def query(self, query: str) -> list[dict[str, Any]]:
            retriever_queries.append(query)
            return [
                {
                    "text": "Cat Jumping onto a laptop In a home office",
                    "source": "data/multimodal_test.pdf",
                    "page_number": 1,
                }
            ]

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create_completion))

        def _create_completion(self, **kwargs: Any) -> Any:
            llm_requests.append(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=("Cat is responsible for the typos because it is jumping " "onto the laptop.")
                        )
                    )
                ]
            )

    fake_openai_module = types.ModuleType("openai")
    fake_openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setattr(nemo_retriever, "create_ingestor", fake_create_ingestor)
    monkeypatch.setattr(retriever_module, "Retriever", FakeRetriever)

    namespace: dict[str, Any] = {"__name__": "__readme_typical_use__"}
    for block_index, code in blocks:
        exec(compile(code, f"README.md#python-{block_index}", "exec"), namespace)

    assert [call[0] for call in ingestors[0].calls] == [
        "files",
        "extract",
        "embed",
        "vdb_upload",
        "ingest",
    ]
    assert ingestors[0].calls[0] == ("files", ["data/multimodal_test.pdf"])
    assert [call[0] for call in ingestors[1].calls] == [
        "urls",
        "extract",
        "embed",
        "vdb_upload",
        "ingest",
    ]
    assert ingestors[1].calls[0] == (
        "urls",
        ["https://ontheline.trincoll.edu/images/bookdown/sample-local-pdf.pdf"],
    )
    assert ingestors[1].calls[-1] == ("ingest", {"return_failures": True})
    assert retriever_init_calls == [{}]
    assert retriever_queries == ["Given their activities, which animal is responsible for the typos in my documents?"]
    assert "Cat Jumping onto a laptop" in llm_requests[0]["messages"][0]["content"]
    assert namespace["answer"].startswith("Cat is responsible")
    assert "Cat is responsible" in capsys.readouterr().out
