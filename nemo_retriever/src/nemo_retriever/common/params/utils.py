# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared coercion and option-building helpers for parameter models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from nemo_retriever.common.params.models import BatchTuningParams


NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-Parse-v1.2"


def validate_nemotron_parse_endpoint_list(invoke_url: str | None) -> tuple[str, ...]:
    """Normalize Parse endpoints and reject mixed NVIDIA Build/self-hosted lists."""
    invoke_urls = tuple(part.strip() for part in str(invoke_url or "").split(",") if part.strip())
    build_endpoints = tuple(
        (urlsplit(endpoint).hostname or "").lower() == "integrate.api.nvidia.com" for endpoint in invoke_urls
    )
    if any(build_endpoints) and not all(build_endpoints):
        raise ValueError(
            "Nemotron Parse endpoint lists cannot mix NVIDIA Build and self-hosted endpoints. "
            "One `nemotron_parse_model` and request contract apply to the entire endpoint list, but NVIDIA Build "
            "requires `nvidia/nemotron-parse` with the hosted tool-call contract and self-hosted Parse requires a "
            "versioned model with a tagged contract. Configure a homogeneous endpoint list or use separate "
            "ingestors for NVIDIA Build and self-hosted Parse."
        )
    return invoke_urls


def coerce_params[T](params: T | None, model_cls: type[T], kwargs: dict[str, Any]) -> T:
    """Merge *params* and *kwargs* into an instance of *model_cls*.

    - If *params* is ``None``, construct from *kwargs*.
    - If *kwargs* is non-empty, apply them as overrides via ``model_copy``.
    - Otherwise return *params* unchanged.
    """
    if params is None:
        return model_cls(**kwargs)
    if kwargs:
        return params.model_copy(update=kwargs)  # type: ignore[return-value]
    return params


def normalize_embed_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize embedding endpoint aliases without changing model identity."""
    normalized = dict(kwargs)
    embed_invoke_url = (
        str(normalized.get("embed_invoke_url") or "").strip() if "embed_invoke_url" in normalized else None
    )
    embedding_endpoint = (
        str(normalized.get("embedding_endpoint") or "").strip() if "embedding_endpoint" in normalized else None
    )

    if embed_invoke_url is not None:
        if embed_invoke_url:
            normalized["embed_invoke_url"] = embed_invoke_url
        else:
            normalized.pop("embed_invoke_url", None)

    if embedding_endpoint is not None:
        if embedding_endpoint:
            normalized["embedding_endpoint"] = embedding_endpoint
        else:
            normalized.pop("embedding_endpoint", None)

    if "embed_invoke_url" in normalized:
        normalized.setdefault("embedding_endpoint", normalized["embed_invoke_url"])
    return normalized


def build_embed_option_kwargs(
    embed_invoke_url: str | None,
    embed_model_name: str | None,
    local_ingest_embed_backend: str | None = None,
    embed_api_key: str | None = None,
    embed_model_provider_prefix: str | None = None,
    embed_modality: str | None = None,
    text_elements_modality: str | None = None,
    structured_elements_modality: str | None = None,
    embed_granularity: str | None = None,
    embed_workers: int | None = None,
    embed_workers_min: int | None = None,
    embed_workers_initial: int | None = None,
    embed_workers_max: int | None = None,
    embed_batch_size: int | None = None,
    embed_cpus_per_actor: float | None = None,
    embed_gpus_per_actor: float | None = None,
    embed_model_revision: str | None = None,
) -> Dict[str, Any]:
    """Build ``EmbedParams`` kwargs from CLI/request option values."""
    embed_kwargs: Dict[str, Any] = {}
    if embed_invoke_url is not None:
        embed_kwargs["embed_invoke_url"] = embed_invoke_url
    if embed_model_name is not None:
        # Remote HTTP embedding reads model_name; local/GPU paths read embed_model_name.
        embed_kwargs["model_name"] = embed_model_name
        embed_kwargs["embed_model_name"] = embed_model_name
    if embed_model_revision is not None:
        embed_kwargs["embed_model_revision"] = embed_model_revision
    if local_ingest_embed_backend is not None:
        embed_kwargs["local_ingest_embed_backend"] = local_ingest_embed_backend
    if embed_api_key is not None:
        embed_kwargs["api_key"] = embed_api_key
    if embed_model_provider_prefix is not None:
        embed_kwargs["embed_model_provider_prefix"] = embed_model_provider_prefix
    if embed_modality is not None:
        embed_kwargs["embed_modality"] = embed_modality
    if text_elements_modality is not None:
        embed_kwargs["text_elements_modality"] = text_elements_modality
    if structured_elements_modality is not None:
        embed_kwargs["structured_elements_modality"] = structured_elements_modality
    if embed_granularity is not None:
        embed_kwargs["embed_granularity"] = embed_granularity
    embed_tuning = _build_embed_batch_tuning(
        embed_workers=embed_workers,
        embed_workers_min=embed_workers_min,
        embed_workers_initial=embed_workers_initial,
        embed_workers_max=embed_workers_max,
        embed_batch_size=embed_batch_size,
        embed_cpus_per_actor=embed_cpus_per_actor,
        embed_gpus_per_actor=embed_gpus_per_actor,
    )
    if embed_tuning is not None:
        embed_kwargs["batch_tuning"] = embed_tuning
    return normalize_embed_kwargs(embed_kwargs)


def _build_embed_batch_tuning(
    *,
    embed_workers: int | None,
    embed_workers_min: int | None,
    embed_workers_initial: int | None,
    embed_workers_max: int | None,
    embed_batch_size: int | None,
    embed_cpus_per_actor: float | None,
    embed_gpus_per_actor: float | None,
) -> BatchTuningParams | None:
    from nemo_retriever.common.params.models import BatchTuningParams

    tuning_kwargs = {
        key: value
        for key, value in {
            "embed_workers": embed_workers,
            "embed_workers_min": embed_workers_min,
            "embed_workers_initial": embed_workers_initial,
            "embed_workers_max": embed_workers_max,
            "embed_batch_size": embed_batch_size,
            "embed_cpus_per_actor": embed_cpus_per_actor,
            "gpu_embed": embed_gpus_per_actor,
        }.items()
        if value is not None
    }
    return BatchTuningParams(**tuning_kwargs) if tuning_kwargs else None


def build_embed_kwargs(resolved: Any, *, include_batch_tuning: bool = False) -> Dict[str, Any]:
    """Flatten an ``EmbedParams`` instance into a dict ready for actor/task kwargs.

    Merges ``runtime`` (always) and optionally ``batch_tuning`` sub-models.
    Also normalises ``embed_invoke_url`` → ``embedding_endpoint``.
    """
    exclude = {"runtime", "batch_tuning"}
    kwargs: Dict[str, Any] = {
        **resolved.model_dump(mode="python", exclude=exclude, exclude_none=True),
        **resolved.runtime.model_dump(mode="python", exclude_none=True),
    }
    if include_batch_tuning:
        kwargs.update(resolved.batch_tuning.model_dump(mode="python", exclude_none=True))

    return normalize_embed_kwargs(kwargs)


SPLIT_CONFIG_VALID_KEYS = frozenset({"text", "html", "pdf", "audio", "image", "video"})


def resolve_split_params(
    split_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve a user-supplied split_config dict into per-key effective params.

    Returns a dict keyed by every entry in ``SPLIT_CONFIG_VALID_KEYS``. Each
    value is one of: a ``TextChunkParams`` / ``HtmlChunkParams`` instance
    (chunking enabled for that key), ``None`` (key absent — chunking off
    via the default), or ``False`` (explicit opt-out sentinel).

    Per-key values supplied by the caller may be a plain dict of
    chunk-params fields, a pre-built ``TextChunkParams`` /
    ``HtmlChunkParams`` instance (passed through verbatim), ``None``, or
    ``False``.
    """
    from nemo_retriever.common.params.models import HtmlChunkParams, TextChunkParams

    cfg = split_config or {}
    unknown = set(cfg) - SPLIT_CONFIG_VALID_KEYS
    if unknown:
        raise ValueError(
            f"Unknown split_config key(s): {sorted(unknown)}; " f"expected one of {sorted(SPLIT_CONFIG_VALID_KEYS)}"
        )

    out: dict[str, Any] = {}
    for key in SPLIT_CONFIG_VALID_KEYS:
        v = cfg.get(key)
        if v is None:
            out[key] = None
            continue
        if v is False:
            out[key] = False  # explicit opt-out (distinct from None / absent)
            continue
        if isinstance(v, TextChunkParams):  # HtmlChunkParams is a TextChunkParams subclass
            out[key] = v
            continue
        if isinstance(v, dict):
            cls = HtmlChunkParams if key == "html" else TextChunkParams
            out[key] = cls(**v)
            continue
        raise TypeError(
            f"split_config['{key}'] must be a TextChunkParams, dict, None, or False; got {type(v).__name__}"
        )
    return out
