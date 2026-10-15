# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light, revision-pinned tokenizers for text chunking."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from nemo_retriever.models.hf_model_registry import (
    get_hf_revision,
    hf_hub_download_with_pinned_revision,
)


class TokenizerUnavailableError(RuntimeError):
    """Raised when an exact configured tokenizer cannot be loaded."""


class ChunkTokenizer(Protocol):
    """Minimal tokenizer interface required by the text splitter."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token IDs."""

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs into text."""


class _FastTokenizer:
    """Adapt ``tokenizers.Tokenizer`` to :class:`ChunkTokenizer`."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text using the pinned tokenizer artifact."""
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return list(encoding.ids)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs using the pinned tokenizer artifact."""
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens))

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        """Encode texts with the fast tokenizer's native batch path."""
        return [
            list(encoding.ids)
            for encoding in self._tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        ]


@lru_cache(maxsize=16)
def load_chunk_tokenizer(
    model_id: str,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> ChunkTokenizer:
    """Load an immutable tokenizer artifact without model weights.

    Args:
        model_id: Hugging Face model identifier registered with a pinned revision.
        cache_dir: Optional Hugging Face cache directory.
        revision: Optional exact model revision. Registered models remain pinned when omitted.

    Returns:
        A dependency-light tokenizer suitable for deterministic chunking.

    Raises:
        TokenizerUnavailableError: If the pinned tokenizer cannot be resolved.
    """
    resolved_revision: str | None = revision
    try:
        local_path = Path(model_id).expanduser()
        if local_path.is_dir():
            tokenizer_path = str(local_path / "tokenizer.json")
            if not Path(tokenizer_path).is_file():
                raise FileNotFoundError(tokenizer_path)
        else:
            resolved_revision = resolved_revision or get_hf_revision(model_id)
            tokenizer_path = hf_hub_download_with_pinned_revision(
                repo_id=model_id,
                filename="tokenizer.json",
                revision=resolved_revision,
                cache_dir=cache_dir,
            )
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        # Count and split the full input, independent of serialized inference settings.
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return _FastTokenizer(tokenizer)
    except Exception as exc:
        raise TokenizerUnavailableError(
            "Unable to load the exact tokenizer required for text chunking: "
            f"model={model_id!r}, revision={resolved_revision!r}. Pre-cache tokenizer.json "
            "for this revision (service image builds: "
            "DOWNLOAD_DEFAULT_TOKENIZER=True) or allow Hugging Face Hub access "
            "in this runtime."
        ) from exc
