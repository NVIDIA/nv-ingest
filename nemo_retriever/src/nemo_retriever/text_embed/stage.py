# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.text_embed.commands import app, main, run
from nemo_retriever.text_embed.processor import embed_text_from_primitives_df, maybe_inject_local_hf_embedder

__all__ = ["app", "embed_text_from_primitives_df", "main", "maybe_inject_local_hf_embedder", "run"]
