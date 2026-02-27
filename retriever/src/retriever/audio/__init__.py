# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio pipeline: media chunking (MediaChunkActor) and ASR (ASRActor).

Provides the same semantics as nv-ingest-api dataloader + Parakeet for
batch, inprocess, fused, and online run modes.
"""

from __future__ import annotations

from retriever.audio.asr_actor import ASRActor
from retriever.audio.asr_actor import asr_params_from_env
from retriever.audio.chunk_actor import MediaChunkActor
from retriever.audio.media_interface import MediaInterface
from retriever.params import ASRParams
from retriever.params import AudioChunkParams

from .cli import app

__all__ = [
    "ASRActor",
    "ASRParams",
    "app",
    "asr_params_from_env",
    "AudioChunkParams",
    "MediaChunkActor",
    "MediaInterface",
]
