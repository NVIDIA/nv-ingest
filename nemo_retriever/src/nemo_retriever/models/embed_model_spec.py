# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve dense embedding checkpoints (Nemotron and Gemma 3) from immutable HF config."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nemo_retriever.models.hf_model_registry import HF_MODEL_REVISIONS

EmbedModelFamily = Literal["text", "vl"]

_MODEL_PROFILES: dict[str, tuple[EmbedModelFamily, str]] = {
    "gemma3_text": ("text", "Gemma3TextModel"),
    "llama_bidirec": ("text", "LlamaBidirectionalModel"),
    "llama_nemotron_vl": ("vl", "LlamaNemotronVLModel"),
    "ministral3": ("text", "Ministral3Model"),
}
_DEFAULT_QUERY_PREFIX = "query: "
_DEFAULT_DOCUMENT_PREFIX = "passage: "
_MODEL_CONFIG_FILENAME = "config.json"
_PROMPT_CONFIG_FILENAME = "config_sentence_transformers.json"
_SENTENCE_CONFIG_FILENAME = "sentence_bert_config.json"
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class EmbedModelSpec:
    """Immutable loading and input-format information for a dense embedder."""

    model_id: str
    revision: str | None
    family: EmbedModelFamily
    output_dimension: int
    query_prefix: str
    document_prefix: str
    quantization: str | None = None
    requires_vllm: bool = False
    max_input_tokens: int | None = None
    query_prefix_declared: bool = False
    document_prefix_declared: bool = False


def _read_config(path: Path, *, model_id: str) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Embedding model {model_id!r} has an unreadable {path.name}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Embedding model {model_id!r} {path.name} must contain a JSON object.")
    return config


def _local_config_path(model_id: str) -> Path | None:
    path = Path(model_id).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            raise ValueError(f"Local embedding model directory {model_id!r} does not contain config.json.")
        return config_path
    if path.is_absolute() or model_id.startswith((".", "~")) or os.path.exists(path):
        raise ValueError(f"Local embedding model path {model_id!r} is not a model directory containing config.json.")
    return None


def _hub_revision(model_id: str, revision: str | None) -> str:
    candidate = revision or HF_MODEL_REVISIONS.get(model_id)
    if candidate and _COMMIT_SHA_RE.fullmatch(candidate):
        return candidate

    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(model_id, revision=candidate)
    except Exception as exc:
        requested = candidate or "main"
        raise ValueError(
            f"Could not resolve Hugging Face embedding model {model_id!r} at revision {requested!r}: {exc}"
        ) from exc
    sha = str(info.sha or "").strip()
    if not _COMMIT_SHA_RE.fullmatch(sha):
        raise ValueError(f"Hugging Face did not return an immutable commit SHA for embedding model {model_id!r}.")
    return sha


def _hub_json(
    model_id: str,
    filename: str,
    revision: str | None,
    *,
    hf_cache_dir: str | None = None,
    optional: bool = False,
) -> dict[str, Any] | None:
    """Read a JSON file from a Hub repo; ``optional`` maps a missing file to None."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=hf_cache_dir,
        )
    except Exception as exc:
        if optional and isinstance(exc, EntryNotFoundError):
            return None
        raise ValueError(
            f"Could not load {filename} for Hugging Face embedding model {model_id!r} at {revision!r}: {exc}"
        ) from exc
    return _read_config(Path(path), model_id=model_id)


def _local_prompt_config(model_id: str, config_path: Path) -> dict[str, Any] | None:
    """Read prompt metadata sitting beside a local checkpoint's config.json."""
    path = config_path.with_name(_PROMPT_CONFIG_FILENAME)
    return _read_config(path, model_id=model_id) if path.is_file() else None


def _local_sentence_config(model_id: str, config_path: Path) -> dict[str, Any] | None:
    """Read Sentence Transformers input metadata beside a local checkpoint."""
    path = config_path.with_name(_SENTENCE_CONFIG_FILENAME)
    return _read_config(path, model_id=model_id) if path.is_file() else None


def _prompt_prefixes(config: dict[str, Any] | None) -> tuple[str, str, bool, bool]:
    prompts = config.get("prompts") if isinstance(config, dict) else None
    if not isinstance(prompts, dict):
        return _DEFAULT_QUERY_PREFIX, _DEFAULT_DOCUMENT_PREFIX, False, False
    query_declared = "query" in prompts
    document_declared = "document" in prompts
    query = prompts.get("query", _DEFAULT_QUERY_PREFIX)
    document = prompts.get("document", _DEFAULT_DOCUMENT_PREFIX)
    if not isinstance(query, str) or not isinstance(document, str):
        raise ValueError("Sentence Transformers query and document prompts must be strings.")
    return query, document, query_declared, document_declared


def _spec_from_config(
    model_id: str,
    revision: str | None,
    config: dict[str, Any],
    prompt_config: dict[str, Any] | None = None,
    sentence_config: dict[str, Any] | None = None,
) -> EmbedModelSpec:
    model_type = str(config.get("model_type") or "").strip()
    profile = _MODEL_PROFILES.get(model_type)
    if profile is None:
        supported = ", ".join(sorted(_MODEL_PROFILES))
        raise ValueError(
            f"Embedding model {model_id!r} uses unsupported model_type {model_type!r}; "
            f"supported dense embed model types are: {supported}."
        )
    family, expected_architecture = profile

    architectures = config.get("architectures")
    if architectures != [expected_architecture]:
        raise ValueError(
            f"Embedding model {model_id!r} uses unsupported architectures {architectures!r}; "
            f"expected [{expected_architecture!r}] for the {family} dense embedding profile."
        )

    if model_type == "ministral3" and config.get("is_causal") is not False:
        raise ValueError(
            f"Embedding model {model_id!r} uses unsupported is_causal={config.get('is_causal')!r}; "
            "dense Ministral3 embedding profiles require is_causal=false."
        )

    dimension_config = config.get("llm_config") if family == "vl" else config
    dimension = dimension_config.get("hidden_size") if isinstance(dimension_config, dict) else None
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError(
            f"Embedding model {model_id!r} has invalid embedding dimension {dimension!r}; "
            "a positive hidden_size is required."
        )

    requires_vllm = False
    if model_type == "gemma3_text":
        if config.get("use_bidirectional_attention") is not True:
            raise ValueError(
                f"Embedding model {model_id!r} uses unsupported "
                f"use_bidirectional_attention={config.get('use_bidirectional_attention')!r}; "
                "Gemma 3 embedding profiles require use_bidirectional_attention=true."
            )
        # Gemma embedders pool and project through Sentence Transformers modules, which only vLLM applies.
        requires_vllm = True
    else:
        pooling = str(config.get("pooling") or "").strip().lower()
        if pooling != "avg":
            raise ValueError(
                f"Embedding model {model_id!r} uses unsupported pooling {pooling!r}; "
                "dense Nemotron embedding profiles require 'avg'."
            )

    quantization_config = config.get("quantization_config")
    quantization = None
    if isinstance(quantization_config, dict):
        quant_method = str(quantization_config.get("quant_method") or "").strip().lower()
        quantization = str(quantization_config.get("quant_algo") or quant_method or "").strip() or None
        if quant_method == "modelopt":
            requires_vllm = True

    query_prefix, document_prefix, query_prefix_declared, document_prefix_declared = _prompt_prefixes(prompt_config)

    max_input_tokens = sentence_config.get("max_seq_length") if isinstance(sentence_config, dict) else None
    if max_input_tokens is None and family == "vl":
        max_input_tokens = config.get("p_max_length")
    if max_input_tokens is not None:
        if isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int) or max_input_tokens <= 0:
            raise ValueError(
                f"Embedding model {model_id!r} has invalid supported input limit {max_input_tokens!r}; "
                "expected a positive integer."
            )

    return EmbedModelSpec(
        model_id=model_id,
        revision=revision,
        family=family,
        output_dimension=dimension,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        max_input_tokens=max_input_tokens,
        query_prefix_declared=query_prefix_declared,
        document_prefix_declared=document_prefix_declared,
        quantization=quantization,
        requires_vllm=requires_vllm,
    )


def resolve_embed_model_spec(
    model_id: str,
    *,
    revision: str | None = None,
    hf_cache_dir: str | None = None,
) -> EmbedModelSpec:
    """Resolve and validate a dense Nemotron embedding checkpoint.

    Registered Hub repositories retain their project-pinned revision. Other
    repositories are resolved to an immutable Hub commit before their config is
    inspected. Compatible checkpoints use a supported dense embedding
    architecture, a positive declared output dimension, and average pooling.
    Query and document prefixes come from Sentence Transformers metadata when
    present. Local directories are identified by config.json and do not carry
    a Hub revision.
    """
    local_config = _local_config_path(model_id)
    if local_config is not None:
        if revision is not None:
            raise ValueError("A Hugging Face revision cannot be used with a local embedding model directory.")
        return _spec_from_config(
            model_id,
            None,
            _read_config(local_config, model_id=model_id),
            _local_prompt_config(model_id, local_config),
            _local_sentence_config(model_id, local_config),
        )

    resolved_revision = _hub_revision(model_id, revision)
    config = _hub_json(model_id, _MODEL_CONFIG_FILENAME, resolved_revision, hf_cache_dir=hf_cache_dir) or {}
    prompt_config = _hub_json(
        model_id, _PROMPT_CONFIG_FILENAME, resolved_revision, hf_cache_dir=hf_cache_dir, optional=True
    )
    sentence_config = _hub_json(
        model_id, _SENTENCE_CONFIG_FILENAME, resolved_revision, hf_cache_dir=hf_cache_dir, optional=True
    )
    return _spec_from_config(model_id, resolved_revision, config, prompt_config, sentence_config)


def resolve_embed_model_revision(model_id: str, revision: str | None) -> str | None:
    """Return an explicit revision or resolve one for a directly constructed embedder."""
    if revision is not None:
        return revision
    if _local_config_path(model_id) is not None:
        return None
    return _hub_revision(model_id, None)


def validate_embed_model_backend(spec: EmbedModelSpec, backend: str) -> None:
    """Reject backends that cannot load the resolved checkpoint format."""
    if spec.requires_vllm and backend != "vllm":
        quantization = f" ({spec.quantization})" if spec.quantization else ""
        raise ValueError(f"Embedding model {spec.model_id!r}{quantization} requires backend='vllm'.")
