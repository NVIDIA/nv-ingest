# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image

from ..model import BaseModel, RunMode


class NemotronParseV12(BaseModel):
    """
    NVIDIA Nemotron Parse v1.2 local wrapper.

    This wrapper loads `nvidia/NVIDIA-Nemotron-Parse-v1.2` from Hugging Face and
    runs image-to-structured-text generation for document parsing.
    """

    def __init__(
        self,
        model_path: str = "nvidia/NVIDIA-Nemotron-Parse-v1.2",
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
    ) -> None:
        super().__init__()

        from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig

        self._model_path = model_path
        self._task_prompt = task_prompt
        self._device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32

        self._model = AutoModel.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            torch_dtype=self._dtype,
            cache_dir=hf_cache_dir,
        ).to(self._device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, cache_dir=hf_cache_dir)
        self._processor = AutoProcessor.from_pretrained(
            self._model_path, trust_remote_code=True, cache_dir=hf_cache_dir
        )
        self._generation_config = GenerationConfig.from_pretrained(
            self._model_path, trust_remote_code=True, cache_dir=hf_cache_dir
        )

    def preprocess(self, input_data: Union[torch.Tensor, np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """
        Normalize supported input formats to a RGB PIL image.
        """
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

    def invoke(
        self,
        input_data: Union[torch.Tensor, np.ndarray, Image.Image, str, Path],
        task_prompt: Optional[str] = None,
    ) -> str:
        """
        Run local Nemotron Parse inference and return decoded model text.
        """
        image = self.preprocess(input_data)
        prompt = task_prompt or self._task_prompt

        inputs = self._processor(
            images=[image],
            text=prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, generation_config=self._generation_config)

        decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    def __call__(
        self,
        input_data: Union[torch.Tensor, np.ndarray, Image.Image, str, Path],
        task_prompt: Optional[str] = None,
    ) -> str:
        return self.invoke(input_data, task_prompt=task_prompt)

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "NVIDIA-Nemotron-Parse-v1.2"

    @property
    def model_type(self) -> str:
        """Model category/type."""
        return "document-parse"

    @property
    def model_runmode(self) -> RunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        return "local"

    @property
    def input(self) -> Any:
        """
        Input schema for the model.
        """
        return {
            "type": "image",
            "format": "RGB",
            "supported_inputs": ["PIL.Image", "path", "torch.Tensor", "np.ndarray"],
            "description": "Document image for parsing into markdown text with structural tags.",
        }

    @property
    def output(self) -> Any:
        """
        Output schema for the model.
        """
        return {
            "type": "text",
            "format": "string",
            "description": "Generated structured parse text from Nemotron Parse v1.2.",
        }

    @property
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        return 1
