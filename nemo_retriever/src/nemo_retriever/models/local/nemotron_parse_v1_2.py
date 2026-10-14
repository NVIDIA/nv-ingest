# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.models.model import BaseModel, ModelRunMode

# Type alias for all supported single-image input formats.
ImageInput = Union[torch.Tensor, np.ndarray, Image.Image, str, Path]


# ---------------------------------------------------------------------------
# vLLM processor bug workaround
# ---------------------------------------------------------------------------
# vLLM's bundled NemotronParseProcessor.__call__ passes add_special_tokens=False
# explicitly to the tokenizer AND also forwards it via **kwargs from the vLLM
# pipeline, causing a duplicate keyword argument TypeError.  We monkey-patch
# the processor class at import time to pop the conflicting kwarg.

_VLLM_PROCESSOR_PATCHED = False


def _patch_vllm_nemotron_parse_processor() -> None:
    """Fix duplicate-kwarg bug in vLLM's NemotronParseProcessor.__call__."""
    global _VLLM_PROCESSOR_PATCHED
    if _VLLM_PROCESSOR_PATCHED:
        return

    try:
        from vllm.model_executor.models.nemotron_parse import NemotronParseProcessor
    except ImportError:
        return

    _orig_call = NemotronParseProcessor.__call__

    def _fixed_call(self, text=None, images=None, return_tensors=None, **kwargs):
        kwargs.pop("add_special_tokens", None)
        return _orig_call(self, text=text, images=images, return_tensors=return_tensors, **kwargs)

    NemotronParseProcessor.__call__ = _fixed_call
    _VLLM_PROCESSOR_PATCHED = True


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class NemotronParseV12(BaseModel):
    """
    NVIDIA Nemotron Parse v1.2 local wrapper backed by vLLM.

    This wrapper loads ``nvidia/NVIDIA-Nemotron-Parse-v1.2`` via vLLM's offline
    ``LLM`` engine for image-to-structured-text generation (document parsing).
    vLLM handles KV-cache management, continuous batching, and GPU scheduling
    internally, avoiding the transformers cache-API incompatibility that affects
    the HuggingFace ``trust_remote_code`` model code with transformers >= 4.52.
    """

    _DEFAULT_TASK_PROMPT: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

    def __init__(
        self,
        model_path: str = "nvidia/NVIDIA-Nemotron-Parse-v1.2",
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        task_prompt: str = _DEFAULT_TASK_PROMPT,
        gpu_memory_utilization: float = 0.8,
        max_num_seqs: int = 64,
        max_tokens: int = 9000,
    ) -> None:
        super().__init__()

        from nemo_retriever.models.inference.vllm import apply_vllm_startup_defaults

        apply_vllm_startup_defaults()
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Local Nemotron Parse requires vLLM. " 'Install with: pip install "nemo-retriever[vllm]"'
            ) from e

        _patch_vllm_nemotron_parse_processor()

        self._model_path = model_path
        self._task_prompt = task_prompt
        self._max_tokens = max_tokens

        if device is not None:
            import os

            dev_id = device.split(":")[-1] if ":" in device else device
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_id

        configure_global_hf_cache_base(hf_cache_dir)
        revision = get_hf_revision(model_path)

        self._llm = LLM(
            model=model_path,
            revision=revision,
            trust_remote_code=True,
            dtype="bfloat16",
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self._sampling_params = SamplingParams(
            temperature=0,
            top_k=1,
            repetition_penalty=1.1,
            max_tokens=self._max_tokens,
            skip_special_tokens=False,
        )

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------

    def preprocess(self, input_data: ImageInput) -> Image.Image:
        """Normalize supported input formats to an RGB PIL image."""
        if isinstance(input_data, Image.Image):
            return input_data.convert("RGB")

        if isinstance(input_data, (str, Path)):
            return Image.open(Path(input_data)).convert("RGB")

        if isinstance(input_data, torch.Tensor):
            x = input_data.detach().cpu()
            if x.ndim == 4:
                if int(x.shape[0]) != 1:
                    raise ValueError(f"Expected batch size 1 tensor, got shape {tuple(x.shape)}")
                x = x[0]
            if x.ndim != 3:
                raise ValueError(f"Expected CHW/HWC tensor, got shape {tuple(x.shape)}")
            if int(x.shape[0]) in (1, 3):
                x = x.permute(1, 2, 0).contiguous()
            if x.dtype.is_floating_point:
                max_v = float(x.max().item()) if x.numel() else 1.0
                if max_v <= 1.5:
                    x = x * 255.0
            arr = x.clamp(0, 255).to(torch.uint8).numpy()
            return Image.fromarray(arr).convert("RGB")

        if isinstance(input_data, np.ndarray):
            arr = input_data
            if arr.ndim == 4:
                if int(arr.shape[0]) != 1:
                    raise ValueError(f"Expected batch size 1 array, got shape {arr.shape}")
                arr = arr[0]
            if arr.ndim != 3:
                raise ValueError(f"Expected HWC/CHW array, got shape {arr.shape}")
            if int(arr.shape[0]) in (1, 3) and int(arr.shape[-1]) not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if np.issubdtype(arr.dtype, np.floating):
                max_v = float(arr.max()) if arr.size else 1.0
                if max_v <= 1.5:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

        raise TypeError(f"Unsupported input type for Nemotron Parse: {type(input_data)!r}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def invoke(
        self,
        input_data: ImageInput,
        task_prompt: Optional[str] = None,
    ) -> str:
        """Run Nemotron Parse on a single image and return the decoded text."""
        return self.invoke_batch([input_data], task_prompt=task_prompt)[0]

    def invoke_batch(
        self,
        inputs: Sequence[ImageInput],
        task_prompt: Optional[str] = None,
    ) -> List[str]:
        """Run Nemotron Parse on a batch of images via vLLM.

        vLLM handles continuous batching and GPU scheduling internally,
        making this significantly faster than sequential single-image calls
        for large batches.
        """
        prompt = task_prompt or self._task_prompt
        prompts = [
            {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {"image": self.preprocess(img)},
                },
                "decoder_prompt": prompt,
            }
            for img in inputs
        ]
        outputs = self._llm.generate(prompts, self._sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    def __call__(
        self,
        input_data: ImageInput,
        task_prompt: Optional[str] = None,
    ) -> str:
        return self.invoke(input_data, task_prompt=task_prompt)

    # ------------------------------------------------------------------
    # BaseModel abstract interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_path.rsplit("/", 1)[-1]

    @property
    def model_type(self) -> str:
        return "document-parse"

    @property
    def model_runmode(self) -> ModelRunMode:
        return "local"

    @property
    def input(self) -> Any:
        return {
            "type": "image",
            "format": "RGB",
            "supported_inputs": ["PIL.Image", "path", "torch.Tensor", "np.ndarray"],
            "description": "Document image for parsing into markdown text with structural tags.",
        }

    @property
    def output(self) -> Any:
        return {
            "type": "text",
            "format": "string",
            "description": f"Generated structured parse text from {self.model_name}.",
        }

    @property
    def input_batch_size(self) -> int:
        return 64
