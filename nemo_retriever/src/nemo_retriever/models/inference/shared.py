# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for graph text-embedding operators."""

from __future__ import annotations

from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.params.utils import normalize_embed_kwargs


def _to_bool(v: object, default: bool = False) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "off", "")
    if v is None:
        return default
    return bool(v)


def build_embed_kwargs(params: EmbedParams) -> dict[str, object]:
    kwargs = {
        **params.model_dump(mode="python", exclude={"runtime", "batch_tuning", "embed_input_audit"}, exclude_none=True),
        **params.runtime.model_dump(mode="python", exclude_none=True),
    }
    return normalize_embed_kwargs(kwargs)
