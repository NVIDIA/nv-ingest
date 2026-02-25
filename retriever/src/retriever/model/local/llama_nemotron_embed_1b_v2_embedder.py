# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


@dataclass
class LlamaNemotronEmbed1BV2Embedder:
    """
    Minimal embedder wrapper for local-only HuggingFace execution.

    This intentionally contains **no remote invocation logic**.
    """

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    normalize: bool = True
    # IMPORTANT: Some HF tokenizers set an effectively "infinite" model_max_length.
    # If we rely on that, `truncation=True` may still allow extremely long sequences,
    # which can explode attention-mask memory (O(seq_len^2)) and OOM the GPU.
    # max_length: int = 4096
    max_length: int = 8192
    model_id: Optional[str] = None

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None

        from transformers import AutoModel, AutoTokenizer

        MODEL_ID = self.model_id or "nvidia/llama-3.2-nv-embedqa-1b-v2"
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = self.hf_cache_dir or str(Path.home() / ".cache" / "huggingface")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=hf_cache_dir)
        self._model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=hf_cache_dir)
        self._model = self._model.to(dev)
        self._model.eval()
        self._device = dev

    @property
    def is_remote(self) -> bool:
        return False

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """
        Returns a CPU tensor of shape [N, D].
        """
        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)

        return self._embed_local(texts_list, batch_size=batch_size)

    def _embed_local(self, texts: List[str], *, batch_size: int) -> torch.Tensor:
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Local embedder was not initialized.")
        dev = self._device

        outs: List[torch.Tensor] = []
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                for i in range(0, len(texts), max(1, int(batch_size))):
                    chunk = texts[i : i + max(1, int(batch_size))]
                    batch = self._tokenizer(
                        chunk,
                        padding=True,
                        truncation=True,
                        max_length=max(1, int(self.max_length)),
                        return_tensors="pt",
                    ).to(dev)
                    out = self._model(**batch)
                    lhs = out.last_hidden_state  # [B, S, D]
                    mask = batch["attention_mask"].unsqueeze(-1)  # [B, S, 1]
                    vec = (lhs * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
                    vec = vec.detach().to("cpu")
                    if self.normalize:
                        vec = _l2_normalize(vec)
                    outs.append(vec)

        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)

    # Intentionally no remote embedding method.
