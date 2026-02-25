# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch


@dataclass
class LlamaNemotronEmbedVL1BV2Embedder:
    """
    Text-only embedder wrapper for ``nvidia/llama-nemotron-embed-vl-1b-v2``.

    The VL model exposes ``encode_queries()`` and ``encode_documents()``
    instead of the standard tokenizer + forward pass used by the embedqa
    model.  This class uses only the text modality.
    """

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    model_id: Optional[str] = None

    # Populated in __post_init__
    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from transformers import AutoModel

        model_id = self.model_id or "nvidia/llama-nemotron-embed-vl-1b-v2"
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = self.hf_cache_dir or str(Path.home() / ".cache" / "huggingface")

        # flash_attention_2 requires the model on GPU at init time, so use
        # device_map when requesting it.  Fall back to sdpa/eager on CPU or
        # when flash-attn is not installed.
        use_gpu = dev.type == "cuda"
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": attn_impl,
                    "cache_dir": hf_cache_dir,
                }
                if attn_impl == "flash_attention_2" and use_gpu:
                    kwargs["device_map"] = dev
                self._model = AutoModel.from_pretrained(model_id, **kwargs)
                break
            except (ValueError, ImportError):
                if attn_impl == "eager":
                    raise
                continue

        if not hasattr(self._model, "device_map"):
            self._model = self._model.to(dev)
        self._model.eval()
        self._device = dev

    @property
    def is_remote(self) -> bool:
        return False

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed document texts. Returns CPU tensor ``[N, 2048]``."""
        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 2048), dtype=torch.float32)
        with torch.inference_mode():
            out = self._model.encode_documents(texts=texts_list)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().float()
        return torch.as_tensor(out, dtype=torch.float32).cpu()

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed query strings. Returns CPU tensor ``[N, 2048]``."""
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return torch.empty((0, 2048), dtype=torch.float32)
        with torch.inference_mode():
            out = self._model.encode_queries(texts_list)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().float()
        return torch.as_tensor(out, dtype=torch.float32).cpu()
