# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parameter coercion and building helpers used by ingest modes."""

from __future__ import annotations

from typing import Any, Dict


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


def build_embed_kwargs(resolved: Any, *, include_batch_tuning: bool = False) -> Dict[str, Any]:
    """Flatten an ``EmbedParams`` instance into a dict ready for actor/task kwargs.

    Merges ``runtime`` (always) and optionally ``batch_tuning`` sub-models.
    Also normalises ``embed_invoke_url`` → ``embedding_endpoint``.
    """
    exclude = {"runtime", "batch_tuning", "fused_tuning"}
    kwargs: Dict[str, Any] = {
        **resolved.model_dump(mode="python", exclude=exclude, exclude_none=True),
        **resolved.runtime.model_dump(mode="python", exclude_none=True),
    }
    if include_batch_tuning:
        kwargs.update(resolved.batch_tuning.model_dump(mode="python", exclude_none=True))

    if "embedding_endpoint" not in kwargs and kwargs.get("embed_invoke_url"):
        kwargs["embedding_endpoint"] = kwargs["embed_invoke_url"]

    return kwargs
