# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

from nemo_retriever.models.local.nemotron_parse_v1_2 import NemotronParseV12


class NemotronParse20(NemotronParseV12):
    """NVIDIA Nemotron Parse 2.0 local wrapper backed by vLLM."""

    def __init__(
        self,
        model_path: str = "nvidia/NVIDIA-Nemotron-Parse-2.0",
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        task_prompt: str = NemotronParseV12._DEFAULT_TASK_PROMPT,
        gpu_memory_utilization: float = 0.8,
        max_num_seqs: int = 64,
        max_tokens: int = 9000,
    ) -> None:
        """Initialize the local Nemotron Parse 2.0 model.

        Parameters
        ----------
        model_path:
            Hugging Face model identifier for the Parse 2.0 checkpoint.
        device:
            Optional device on which to run inference.
        hf_cache_dir:
            Optional directory for cached Hugging Face assets.
        task_prompt:
            Task-control prompt supplied to the model.
        gpu_memory_utilization:
            Fraction of GPU memory reserved by vLLM.
        max_num_seqs:
            Maximum number of sequences processed concurrently.
        max_tokens:
            Maximum number of generated tokens per request.
        """
        super().__init__(
            model_path=model_path,
            device=device,
            hf_cache_dir=hf_cache_dir,
            task_prompt=task_prompt,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_tokens=max_tokens,
        )
