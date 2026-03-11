# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


@dataclass
class LlamaNemotronEmbed1BV2Embedder:
    """
    Minimal embedder wrapper for local HuggingFace or vLLM execution.

    When use_vllm=False (default), uses HuggingFace AutoModel. When use_vllm=True,
    uses vLLM's Python API (bfloat16 + FLASH_ATTN). Exposes the same embed() interface
    for both backends. No remote invocation logic.
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
    # vLLM backend: when True, use vLLM Python API instead of HF.
    use_vllm: bool = False
    gpu_memory_utilization: float = 0.45
    enforce_eager: bool = False
    compile_cache_dir: Optional[str] = None
    dimensions: Optional[int] = None

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None
        self._llm: Any = None

        if self.use_vllm:
            try:
                from nemo_retriever.model import _DEFAULT_EMBED_MODEL
                from nemo_retriever.text_embed.vllm import create_vllm_llm
            except ImportError as e:
                raise RuntimeError(
                    "vLLM embedding requires the embed-vllm extra. "
                    "Install with: uv pip install -e '.[embed-vllm]' or pip install -e '.[embed-vllm]'"
                ) from e
            model_id = self.model_id or _DEFAULT_EMBED_MODEL
            self._llm = create_vllm_llm(
                str(model_id),
                dimensions=self.dimensions,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=self.enforce_eager,
                compile_cache_dir=self.compile_cache_dir,
            )
            return

        from nemo_retriever.model import _DEFAULT_EMBED_MODEL
        from transformers import AutoModel, AutoTokenizer

        MODEL_ID = self.model_id or _DEFAULT_EMBED_MODEL
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = configure_global_hf_cache_base(self.hf_cache_dir)
        _revision = get_hf_revision(MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=_revision,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=_revision,
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
        )
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

        if self.use_vllm and self._llm is not None:
            from nemo_retriever.text_embed.vllm import embed_with_vllm_llm

            vectors = embed_with_vllm_llm(
                texts_list,
                self._llm,
                batch_size=max(1, int(batch_size)),
                prefix="passage: ",
            )
            if not vectors:
                return torch.empty((0, 0), dtype=torch.float32)
            return torch.tensor(vectors, dtype=torch.float32)
        return self._embed_local(texts_list, batch_size=batch_size)

    def _embed_local(self, texts: List[str], *, batch_size: int) -> torch.Tensor:
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Local embedder was not initialized.")
        dev = self._device

        outs: List[torch.Tensor] = []
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
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
                    out = self._model(**batch, output_hidden_states=True)
                    # The bidirectional model returns BaseModelOutputWithPast
                    # (last_hidden_state), but some transformers versions or
                    # model revisions return CausalLMOutputWithPast (hidden_states).
                    lhs = getattr(out, "last_hidden_state", None)
                    if lhs is None:
                        # CausalLMOutputWithPast: use the last layer's hidden state.
                        hs = getattr(out, "hidden_states", None)
                        if hs is not None:
                            lhs = hs[-1]
                        else:
                            raise AttributeError(
                                f"Model output ({type(out).__name__}) has neither "
                                "'last_hidden_state' nor 'hidden_states'. "
                                "Ensure the model is loaded with trust_remote_code=True."
                            )
                    # lhs shape: [B, S, D]
                    mask = batch["attention_mask"].unsqueeze(-1)  # [B, S, 1]
                    vec = (lhs * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
                    vec = vec.detach().to("cpu")
                    if self.normalize:
                        vec = _l2_normalize(vec)
                    outs.append(vec)

        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)

    # Intentionally no remote embedding method.
