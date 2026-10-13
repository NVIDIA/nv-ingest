# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One resolved construction contract for warmed and cold local embedders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nemo_retriever.models.embed_model_spec import EmbedModelSpec


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


@dataclass(frozen=True)
class LocalEmbedderSpec:
    """Immutable checkpoint identity plus every local factory setting."""

    checkpoint: EmbedModelSpec
    backend: str
    device: str | None
    hf_cache_dir: str | None
    gpu_memory_utilization: float
    enforce_eager: bool
    dimensions: int | None
    normalize: bool
    max_length: int
    query_max_length: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LocalEmbedderSpec":
        """Resolve actor or serialized warmup config to one immutable spec."""
        from nemo_retriever.models import resolve_local_embed_model
        from nemo_retriever.models.embed_model_spec import resolve_embed_model_spec, validate_embed_model_backend

        backend = str(config.get("local_ingest_embed_backend") or config.get("backend") or "vllm").strip().lower()
        model_name = resolve_local_embed_model(
            config.get("embed_model_name") or config.get("model_name"), backend=backend
        )
        revision = config.get("embed_model_revision") or config.get("revision")
        cache = config.get("hf_cache_dir")
        hf_cache_dir = str(cache) if cache else None
        checkpoint = resolve_embed_model_spec(model_name, revision=revision, hf_cache_dir=hf_cache_dir)
        validate_embed_model_backend(checkpoint, backend)

        configured_max = int(config.get("max_length", 8192))
        configured_query_max = int(config.get("query_max_length", 128))
        supported_max = checkpoint.max_input_tokens
        max_length = min(configured_max, supported_max) if supported_max is not None else configured_max
        query_max_length = (
            min(configured_query_max, supported_max) if supported_max is not None else configured_query_max
        )
        device = config.get("local_hf_device") or config.get("device")
        dimensions = config.get("dimensions")
        return cls(
            checkpoint=checkpoint,
            backend=backend,
            device=str(device) if device else None,
            hf_cache_dir=hf_cache_dir,
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.45)),
            enforce_eager=_as_bool(config.get("enforce_eager"), default=False),
            dimensions=int(dimensions) if dimensions is not None else None,
            normalize=_as_bool(config.get("normalize"), default=True),
            max_length=max_length,
            query_max_length=query_max_length,
        )

    @property
    def model_name(self) -> str:
        """Return the checkpoint's canonical model ID."""
        return self.checkpoint.model_id

    @property
    def revision(self) -> str | None:
        """Return the immutable Hub revision, or None for a local checkpoint."""
        return self.checkpoint.revision

    @property
    def prefix_if_missing(self) -> bool:
        """Whether the selected backend preserves an already-prefixed source."""
        return self.checkpoint.model_type == "mistral3" or (self.backend == "hf" and self.checkpoint.family == "text")

    def create(self) -> Any:
        """Construct the embedder described by this exact resolved spec."""
        from nemo_retriever.models import _create_local_embedder_from_spec

        return _create_local_embedder_from_spec(
            self.checkpoint,
            backend=self.backend,
            device=self.device,
            hf_cache_dir=self.hf_cache_dir,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            dimensions=self.dimensions,
            normalize=self.normalize,
            max_length=self.max_length,
            query_max_length=self.query_max_length,
        )


__all__ = ["LocalEmbedderSpec"]
