# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import nvtx
import torch

from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.embed_model_spec import resolve_embed_model_revision


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


def _to_normalized_cpu(out: Any) -> torch.Tensor:
    """L2-normalize a model output on CPU, accepting tensors or array-likes."""
    if isinstance(out, torch.Tensor):
        return _l2_normalize(out.detach().cpu())
    return _l2_normalize(torch.as_tensor(out, dtype=torch.float32).cpu())


def _b64_to_pil(b64_str: str):
    import base64
    import io

    from PIL import Image

    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to decode base64 image: {exc}") from exc


@dataclass
class LlamaNemotronEmbedVL1BV2Embedder:
    """
    HF wrapper for compatible ``LlamaNemotronVLModel`` checkpoints.

    The VL model exposes ``encode_queries()`` and ``encode_documents()``
    instead of the standard tokenizer + forward pass used by the embedqa
    model. This class supports text, image, and text+image modalities; its
    legacy class name is retained for compatibility.
    """

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    model_id: Optional[str] = None
    revision: Optional[str] = None
    output_dimension: int = 2048
    max_length: int = 4096

    # Populated lazily by _ensure_loaded.
    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModel

        from nemo_retriever.models import VL_EMBED_MODEL

        model_id = self.model_id or VL_EMBED_MODEL
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = configure_global_hf_cache_base(self.hf_cache_dir)

        # flash_attention_2 requires the model on GPU at init time, so use
        # device_map when requesting it.  Fall back to sdpa/eager on CPU or
        # when flash-attn is not installed.
        use_gpu = dev.type == "cuda"
        _revision = resolve_embed_model_revision(model_id, self.revision)
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": attn_impl,
                    "cache_dir": hf_cache_dir,
                    "revision": _revision,
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

    def _set_p_max_length(self, modality: str) -> None:
        _RECOMMENDED = {"text": int(self.max_length), "image": 2048, "text_image": 10240}
        p = _RECOMMENDED.get(modality)
        if p is not None and hasattr(self._model, "processor"):
            self._model.processor.p_max_length = p

    @nvtx.annotate("nrl.model::llama_nemotron_embed_vl.embed", color="blue")
    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed document texts. Returns CPU tensor ``[N, D]``."""
        self._ensure_loaded()
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            self._set_p_max_length("text")
            out = self._model.encode_documents(texts=texts_list)
        return _to_normalized_cpu(out)

    @nvtx.annotate("nrl.model::llama_nemotron_embed_vl.embed_queries", color="blue")
    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed query strings. Returns CPU tensor ``[N, D]``."""
        self._ensure_loaded()
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            out = self._model.encode_queries(texts_list)
        return _to_normalized_cpu(out)

    @nvtx.annotate("nrl.model::llama_nemotron_embed_vl.embed_images", color="blue")
    def embed_images(self, images_b64: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed images (base64-encoded). Returns CPU tensor ``[N, D]``."""
        self._ensure_loaded()
        image_dicts = [{"base64": b64} for b64 in images_b64 if b64]
        if not image_dicts:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            self._set_p_max_length("image")
            out = self._model.encode_documents(images=image_dicts)
        return _to_normalized_cpu(out)

    @nvtx.annotate("nrl.model::llama_nemotron_embed_vl.embed_text_image", color="blue")
    def embed_text_image(
        self, texts: Sequence[str], images_b64: Sequence[str], *, batch_size: int = 64
    ) -> torch.Tensor:
        """Embed paired text+image inputs. Returns CPU tensor ``[N, D]``."""
        self._ensure_loaded()
        paired_texts: list[str] = []
        paired_images: list[dict[str, str]] = []
        for t, b64 in zip(texts, images_b64):
            if b64:
                paired_texts.append(str(t))
                paired_images.append({"base64": b64})
        if not paired_images:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            self._set_p_max_length("text_image")
            out = self._model.encode_documents(texts=paired_texts, images=paired_images)
        return _to_normalized_cpu(out)

    def unload(self) -> None:
        """Release GPU memory held by the HF model."""
        del self._model
        self._model = None
        self._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class LlamaNemotronEmbedVL1BV2VLLMEmbedder:
    """
    vLLM embedder for compatible ``LlamaNemotronVLModel`` checkpoints.

    Supports text, image, and text+image modalities via vLLM's Python API
    (bfloat16 + FLASH_ATTN, pooling runner). Requires vLLM >= 0.17.0.
    The legacy class name is retained for compatibility.
    """

    model_id: Optional[str] = None
    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    gpu_memory_utilization: float = 0.45
    enforce_eager: bool = False
    revision: Optional[str] = None
    output_dimension: int = 2048
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    normalize: bool = True

    _llm: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import vllm as _vllm_mod
        except ImportError as e:
            raise RuntimeError("vLLM is not installed. Install with: uv pip install -e '.[local]'") from e

        from packaging.version import Version

        _vllm_version = Version(_vllm_mod.__version__)
        if _vllm_version < Version("0.17.0"):
            raise RuntimeError(
                f"VLM embedding via vLLM requires vLLM >= 0.17.0, "
                f"but {_vllm_version} is installed. "
                "Update with: uv pip install 'vllm>=0.17.0'"
            )

        if self.device is not None:
            import warnings

            warnings.warn(
                "LlamaNemotronEmbedVL1BV2VLLMEmbedder no longer uses 'device'; "
                "vLLM follows process-level GPU placement. Use CUDA_VISIBLE_DEVICES "
                "or vLLM's tensor_parallel_size instead. 'device' will be removed "
                "in a future release.",
                DeprecationWarning,
                stacklevel=4,
            )

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from nemo_retriever.models.inference.vllm import create_vllm_llm

        configure_global_hf_cache_base(self.hf_cache_dir)
        model_id = self.model_id or "nvidia/llama-nemotron-embed-vl-1b-v2"
        self._llm = create_vllm_llm(
            str(model_id),
            revision=resolve_embed_model_revision(model_id, self.revision),
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            limit_mm_per_prompt={"image": 1},
        )

    @property
    def is_remote(self) -> bool:
        return False

    def _finalize_vectors(self, vectors: Sequence[Sequence[float]]) -> torch.Tensor:
        """Require one real embedding per input, then optionally normalize."""
        if not vectors:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        missing = sum(not vector for vector in vectors)
        if missing:
            raise RuntimeError(f"vLLM failed to return an embedding for {missing}/{len(vectors)} inputs")
        tensor = torch.tensor(vectors, dtype=torch.float32)
        return _l2_normalize(tensor) if self.normalize else tensor

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed document texts. Returns CPU tensor ``[N, D]``."""
        self._ensure_loaded()
        from nemo_retriever.models.inference.vllm import embed_with_vllm_llm

        texts_list = [str(t) for t in texts]
        if not texts_list:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        vectors = embed_with_vllm_llm(
            texts_list,
            self._llm,
            batch_size=max(1, int(batch_size)),
            prefix=self.document_prefix,
            normalize=self.normalize,
        )
        return self._finalize_vectors(vectors)

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed query strings. Returns CPU tensor ``[N, D]``."""
        self._ensure_loaded()
        from nemo_retriever.models.inference.vllm import embed_with_vllm_llm

        texts_list = [str(t) for t in texts]
        if not texts_list:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        vectors = embed_with_vllm_llm(
            texts_list,
            self._llm,
            batch_size=max(1, int(batch_size)),
            prefix=self.query_prefix,
            normalize=self.normalize,
        )
        return self._finalize_vectors(vectors)

    def embed_images(self, images_b64: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed images (base64-encoded). Returns CPU tensor ``[N, D]``."""
        self._ensure_loaded()
        from nemo_retriever.models.inference.vllm import embed_multimodal_with_vllm_llm

        valid_b64 = [b64 for b64 in images_b64 if b64 and str(b64).strip()]
        if not valid_b64:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)

        prompt_dicts = [
            {"prompt": f"{self.document_prefix}<image> ", "multi_modal_data": {"image": _b64_to_pil(b64)}}
            for b64 in valid_b64
        ]
        vectors = embed_multimodal_with_vllm_llm(
            prompt_dicts,
            self._llm,
            batch_size=max(1, int(batch_size)),
            normalize=self.normalize,
        )
        return self._finalize_vectors(vectors)

    def embed_text_image(
        self, texts: Sequence[str], images_b64: Sequence[str], *, batch_size: int = 64
    ) -> torch.Tensor:
        """Embed paired text+image inputs. Returns CPU tensor ``[N, D]``."""
        self._ensure_loaded()
        from nemo_retriever.models.inference.vllm import embed_multimodal_with_vllm_llm

        paired_texts: list = []
        paired_b64: list = []
        for t, b64 in zip(texts, images_b64):
            if b64 and str(b64).strip():
                paired_texts.append(str(t))
                paired_b64.append(b64)

        if not paired_b64:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)

        prompt_dicts = [
            {"prompt": f"{self.document_prefix}<image> {text}", "multi_modal_data": {"image": _b64_to_pil(b64)}}
            for text, b64 in zip(paired_texts, paired_b64)
        ]
        vectors = embed_multimodal_with_vllm_llm(
            prompt_dicts,
            self._llm,
            batch_size=max(1, int(batch_size)),
            normalize=self.normalize,
        )
        return self._finalize_vectors(vectors)

    def unload(self) -> None:
        """Release GPU memory held by the vLLM engine."""
        del self._llm
        self._llm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
