# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import warnings
from io import BytesIO
from typing import Any, List, Optional

import nvtx
from PIL import Image

from nemo_retriever.common.modality.caption.model_profiles import (
    CaptionTarget,
    DEFAULT_LOCAL_CAPTION_MODEL_ID,
    caption_model_aliases,
    caption_model_revisions,
    get_caption_model_profile,
    merge_request_extras,
    resolve_caption_model_name as _resolve_caption_model_name,
    supported_caption_models_by_variant,
)
from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.model import BaseModel, ModelRunMode

_DEFAULT_MAX_NUM_SEQS = 256


def _b64_to_pil(b64: str) -> Image.Image:
    """Decode a base64-encoded image string to a PIL Image."""
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def resolve_caption_model_name(name: str, *, target: CaptionTarget | str = "local") -> str:
    """Deprecated shim for caption model name resolution."""

    warnings.warn(
        "nemo_retriever.model.local.nemotron_vlm_captioner.resolve_caption_model_name is deprecated; "
        "use nemo_retriever.caption.model_profiles.resolve_caption_model_name instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _resolve_caption_model_name(name, target=target)


class NemotronVLMCaptioner(BaseModel):
    """
    Local VLM captioner wrapping supported Nemotron VLM caption profiles.

    Supported models:

    * ``nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`` (default,
      BFloat16; approximately 62 GiB of model weights)
    * ``nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`` (BFloat16)
    * ``nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8``  (FP8 quantised)
    * ``nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD`` (NVFP4 quantised,
      requires GPU compute capability >= 8.9, e.g. Ada Lovelace / Hopper)
    * Nemotron 3 Nano Omni FP8 and NVFP4 profiles and aliases

    Uses vLLM for inference with batched scheduling.

    Usage::

        captioner = NemotronVLMCaptioner()
        captions = captioner.caption_batch(
            ["<base64-png>", "<base64-png>"],
            prompt="Caption the content of this image:",
        )
    """

    SUPPORTED_MODELS: dict[str, str] = supported_caption_models_by_variant(target="local")
    MODEL_ALIASES: dict[str, str] = caption_model_aliases(target="local")
    _MODEL_REVISIONS: dict[str, str | None] = caption_model_revisions()

    def __init__(
        self,
        model_path: str = DEFAULT_LOCAL_CAPTION_MODEL_ID,
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        max_new_tokens: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float | None = None,
    ) -> None:
        super().__init__()

        profile = get_caption_model_profile(model_path, target="local")
        model_path = profile.local_model_id
        if gpu_memory_utilization is None:
            gpu_memory_utilization = profile.local_gpu_memory_utilization

        from nemo_retriever.models.inference.vllm import apply_vllm_startup_defaults

        apply_vllm_startup_defaults()
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            raise ImportError(
                'Local VLM captioning requires vLLM. Install with: pip install "nemo-retriever[local]"'
            ) from e

        self._profile = profile
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._request_extras = profile.request_extras_for("local")

        if device is not None:
            # vLLM uses CUDA_VISIBLE_DEVICES rather than a torch device string.
            # Translate e.g. "cuda:1" → "1" so vLLM sees only the requested GPU.
            import os

            dev_id = device.split(":")[-1] if ":" in device else device
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_id

        configure_global_hf_cache_base(hf_cache_dir)

        revision = profile.revision
        engine_kwargs = profile.engine_kwargs_for_local()
        # Caption workloads are small; vLLM's default 1024 sequences can exceed
        # Mamba cache blocks for Nemotron VL at ordinary memory utilization.
        engine_kwargs.setdefault("max_num_seqs", _DEFAULT_MAX_NUM_SEQS)

        self._llm = LLM(
            model=model_path,
            revision=revision,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=float(gpu_memory_utilization),
            **engine_kwargs,
        )

    def _build_messages(
        self,
        base64_image: str,
        *,
        prompt: str,
        system_prompt: Optional[str],
    ) -> list[dict[str, Any]]:
        """Build chat messages in OpenAI format for vLLM."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return messages

    def caption(
        self,
        base64_image: str,
        *,
        prompt: str = "Caption the content of this image:",
        system_prompt: Optional[str] = "/no_think",
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Generate a caption for a single base64-encoded image."""
        return self.caption_batch(
            [base64_image],
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )[0]

    @nvtx.annotate("nrl.model::vlm_captioner.caption_batch", color="blue")
    def caption_batch(
        self,
        base64_images: List[str],
        *,
        prompt: str = "Caption the content of this image:",
        system_prompt: Optional[str] = "/no_think",
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: dict[str, Any] | None = None,
    ) -> List[str]:
        """Generate captions for a list of base64-encoded images.

        vLLM batches internally and handles scheduling across images.
        """
        from vllm import SamplingParams

        conversations = [self._build_messages(b64, prompt=prompt, system_prompt=system_prompt) for b64 in base64_images]
        sp_kwargs: dict = {
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_new_tokens,
        }
        if top_p is not None:
            sp_kwargs["top_p"] = top_p
        sampling_params = SamplingParams(**sp_kwargs)
        chat_kwargs = merge_request_extras(self._request_extras, extra_body or {})
        chat_kwargs.setdefault("use_tqdm", False)
        outputs = self._llm.chat(conversations, sampling_params=sampling_params, **chat_kwargs)
        return [out.outputs[0].text.strip() for out in outputs]

    # ---- BaseModel abstract interface ----

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def model_type(self) -> str:
        return "vlm-captioner"

    @property
    def model_runmode(self) -> ModelRunMode:
        return "local"

    @property
    def input(self) -> Any:
        return {
            "type": "image",
            "format": "base64",
            "description": "Base64-encoded image for captioning.",
        }

    @property
    def output(self) -> Any:
        return {
            "type": "text",
            "format": "string",
            "description": "Generated caption for the input image.",
        }

    @property
    def input_batch_size(self) -> int:
        return 1
