# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dependency-light, revision-pinned chunk tokenizers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import ByteLevel, Whitespace

import nemo_retriever.common.modality.txt.tokenizer_provider as provider
import nemo_retriever.operators.extract.html.ray_data as html_actor_module
import nemo_retriever.operators.extract.txt.ray_data as txt_actor_module
from nemo_retriever.common.modality.html.convert import html_bytes_to_chunks_df
from nemo_retriever.common.modality.txt.split import (
    DEFAULT_TOKENIZER_MODEL_ID,
    txt_bytes_to_chunks_df,
)
from nemo_retriever.models import NEMOTRON_3_EMBED_BF16_MODEL


def _write_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "hello": 1, "world": 2},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def test_default_chunk_tokenizer_matches_default_embedding_model() -> None:
    assert DEFAULT_TOKENIZER_MODEL_ID == NEMOTRON_3_EMBED_BF16_MODEL


def test_load_chunk_tokenizer_uses_pinned_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    provider.load_chunk_tokenizer.cache_clear()
    monkeypatch.setattr(
        provider,
        "hf_hub_download_with_pinned_revision",
        lambda **_kwargs: str(tokenizer_path),
    )

    tokenizer = provider.load_chunk_tokenizer("nvidia/llama-nemotron-embed-vl-1b-v2")

    token_ids = tokenizer.encode("hello world")
    assert tokenizer.decode(token_ids) == "hello world"


def test_txt_and_html_chunk_without_transformers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    provider.load_chunk_tokenizer.cache_clear()
    monkeypatch.setattr(
        provider,
        "hf_hub_download_with_pinned_revision",
        lambda **_kwargs: str(tokenizer_path),
    )

    txt = txt_bytes_to_chunks_df(b"hello world", "document.txt")
    html = html_bytes_to_chunks_df(
        b"<html><body><p>hello world</p></body></html>",
        "document.html",
    )

    assert txt["text"].tolist() == ["hello world"]
    assert html["text"].tolist() == ["hello world"]


@pytest.mark.parametrize("local_checkpoint", [False, True])
@pytest.mark.parametrize("truncation,padding", [(True, False), (False, True), (True, True)])
def test_chunk_tokenizer_ignores_serialized_truncation_and_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_checkpoint: bool,
    truncation: bool,
    padding: bool,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    serialized_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if truncation:
        serialized_tokenizer.enable_truncation(max_length=2048)
    if padding:
        serialized_tokenizer.enable_padding(pad_id=0, pad_token="[UNK]", pad_to_multiple_of=8)
    serialized_tokenizer.save(str(tokenizer_path))
    original_artifact = tokenizer_path.read_bytes()
    monkeypatch.setattr(provider, "hf_hub_download_with_pinned_revision", lambda **_kwargs: str(tokenizer_path))
    provider.load_chunk_tokenizer.cache_clear()
    model_id = str(tmp_path) if local_checkpoint else "nvidia/llama-nemotron-embed-vl-1b-v2"

    tokenizer = provider.load_chunk_tokenizer(model_id)
    long_text = "hello " * 2051

    assert tokenizer.encode(long_text) == [1] * 2051
    assert tokenizer.encode("hello") == [1]
    assert tokenizer.encode_batch([long_text, "hello world", "hello"]) == [[1] * 2051, [1, 2], [1]]
    assert tokenizer_path.read_bytes() == original_artifact


def test_serialized_truncation_does_not_bypass_lossless_embedding_admission(tmp_path: Path) -> None:
    from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy

    alphabet = sorted(ByteLevel.alphabet())
    serialized_tokenizer = Tokenizer(BPE({character: index for index, character in enumerate(alphabet)}, []))
    serialized_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    serialized_tokenizer.decoder = ByteLevelDecoder()
    serialized_tokenizer.enable_truncation(max_length=4)
    serialized_tokenizer.enable_padding(pad_id=0, pad_token=alphabet[0], pad_to_multiple_of=8)
    serialized_tokenizer.save(str(tmp_path / "tokenizer.json"))
    provider.load_chunk_tokenizer.cache_clear()
    tokenizer = provider.load_chunk_tokenizer(str(tmp_path))
    policy = EmbeddingInputPolicy(tokenizer=tokenizer, max_tokens=8, prefix="", add_special_tokens=False)
    text = "hello world " * 5

    plan = policy.plan([text])[0]

    assert plan.formatted_tokens == len(text)
    assert plan.requires_split
    assert "".join(child.content for child in plan.children) == text
    assert all(len(tokenizer.encode(child.content)) <= policy.max_tokens for child in plan.children)


def test_load_chunk_tokenizer_surfaces_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider.load_chunk_tokenizer.cache_clear()

    def fail_download(**_kwargs: object) -> str:
        raise RuntimeError("offline cache miss")

    monkeypatch.setattr(
        provider,
        "hf_hub_download_with_pinned_revision",
        fail_download,
    )

    with pytest.raises(
        provider.TokenizerUnavailableError,
        match="Pre-cache tokenizer.json",
    ):
        provider.load_chunk_tokenizer("nvidia/llama-nemotron-embed-vl-1b-v2")


def test_load_chunk_tokenizer_wraps_unregistered_model_error() -> None:
    provider.load_chunk_tokenizer.cache_clear()

    with pytest.raises(
        provider.TokenizerUnavailableError,
        match="unregistered/model",
    ) as exc_info:
        provider.load_chunk_tokenizer("unregistered/model")

    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("module", "actor_type", "splitter_name"),
    [
        (txt_actor_module, txt_actor_module.TxtSplitCPUActor, "txt_bytes_to_chunks_df"),
        (
            html_actor_module,
            html_actor_module.HtmlSplitCPUActor,
            "html_bytes_to_chunks_df",
        ),
    ],
)
def test_text_actors_propagate_tokenizer_failures(
    module: object,
    actor_type: type,
    splitter_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_split(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise provider.TokenizerUnavailableError("tokenizer unavailable")

    monkeypatch.setattr(module, splitter_name, fail_split)
    actor = actor_type()
    batch = pd.DataFrame([{"bytes": b"content", "path": "document.txt"}])

    with pytest.raises(provider.TokenizerUnavailableError, match="unavailable"):
        actor.process(batch)


@pytest.mark.parametrize(
    ("module", "actor_type", "splitter_name"),
    [
        (txt_actor_module, txt_actor_module.TxtSplitCPUActor, "txt_bytes_to_chunks_df"),
        (
            html_actor_module,
            html_actor_module.HtmlSplitCPUActor,
            "html_bytes_to_chunks_df",
        ),
    ],
)
def test_text_actors_isolate_non_tokenizer_document_failures(
    module: object,
    actor_type: type,
    splitter_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def split_document(_payload: object, path: str, **_kwargs: object) -> pd.DataFrame:
        if path == "bad.txt":
            raise ValueError("malformed document")
        return pd.DataFrame([{"text": "good", "path": path, "page_number": 1, "metadata": {}}])

    monkeypatch.setattr(module, splitter_name, split_document)
    actor = actor_type()
    batch = pd.DataFrame(
        [
            {"bytes": b"bad", "path": "bad.txt"},
            {"bytes": b"good", "path": "good.txt"},
        ]
    )

    result = actor.process(batch)

    assert result["path"].tolist() == ["good.txt"]
