# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

import base64
import io
import os
from pathlib import Path  # noqa: F401

import numpy as np
import nvtx
import torch
from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.hf_model_registry import install_pinned_hf_hub_download
from nemo_retriever.models.model import BaseModel, ModelRunMode

from PIL import Image


class NemotronOCRV2(BaseModel):
    """
    Nemotron OCR v2 model for optical character recognition.

    End-to-end OCR model that integrates:
    - Text detector for region localization (RegNetX-8GF backbone)
    - Text recognizer for transcription (pre-norm Transformer)
    - Relational model for layout and reading order analysis

    Supports the ``nemotron_ocr`` package ``lang`` selector:
    - en
    - english
    - multi
    - multilingual
    - v1
    - legacy
    """

    _VALID_LANG_SELECTORS: frozenset[str] = frozenset({"en", "english", "multi", "multilingual", "v1", "legacy"})

    def __init__(
        self,
        model_dir: Optional[str] = None,
        lang: str = "multi",
    ) -> None:
        if lang not in self._VALID_LANG_SELECTORS:
            raise ValueError(
                f"Invalid lang selector {lang!r}. "
                f"Supported values: {sorted(self._VALID_LANG_SELECTORS)}. "
                "Use 'multi' or 'multilingual' for multilingual, 'en' or 'english' for English-only, "
                "or 'v1'/'legacy' to run the v1 model through the v2 package."
            )
        super().__init__()
        configure_global_hf_cache_base()
        try:
            from nemotron_ocr.inference import pipeline as _nemotron_ocr_pipeline  # local-only import
            from nemotron_ocr.inference import pipeline_v2 as _nemotron_ocr_pipeline_v2  # local-only import
        except ImportError as exc:
            raise ImportError(
                "Local Nemotron OCR v2 requires the `nemotron_ocr` package. "
                "Install `nemotron-ocr` 2.0.0 or newer, or install from source via: "
                "git clone https://huggingface.co/nvidia/nemotron-ocr-v2 && "
                "cd nemotron-ocr-v2/nemotron-ocr && pip install --no-build-isolation -v . "
                "Alternatively, run with --ocr-invoke-url pointed at a v2 endpoint. "
                "--ocr-version v1 still uses this package and selects its legacy mode."
            ) from exc

        # NemotronOCRV2 inherits checkpoint loading from NemotronOCR, whose
        # module owns the hf_hub_download global used by _download_checkpoints.
        install_pinned_hf_hub_download(_nemotron_ocr_pipeline)
        _NemotronOCRV2 = _nemotron_ocr_pipeline_v2.NemotronOCRV2

        if model_dir:
            self._model = _NemotronOCRV2(model_dir=model_dir, lang=lang)
        else:
            self._model = _NemotronOCRV2(lang=lang)
        self._enable_trt = os.getenv("RETRIEVER_ENABLE_TORCH_TRT", "").strip().lower() in {"1", "true", "yes", "on"}
        if self._enable_trt and self._model is not None:
            self._maybe_compile_submodules()

    def _maybe_compile_submodules(self) -> None:
        """
        Best-effort TensorRT compilation of internal nn.Modules.
        Any failure falls back to eager PyTorch without breaking initialization.
        """
        try:
            import torch_tensorrt  # type: ignore
        except Exception:
            return

        if self._model is None:
            return

        detector = getattr(self._model, "detector", None)
        if not isinstance(detector, torch.nn.Module):
            return

        try:
            trt_input = torch_tensorrt.Input((1, 3, 1024, 1024), dtype=torch.float16)
        except TypeError:
            trt_input = torch_tensorrt.Input(shape=(1, 3, 1024, 1024), dtype=torch.float16)

        compile_kwargs: Dict[str, Any] = {
            "inputs": [trt_input],
            "enabled_precisions": {torch.float16},
        }
        if hasattr(torch_tensorrt, "compile"):
            compile_kwargs["torch_executed_ops"] = {"torchvision::nms"}
            compile_kwargs["torch_executed_modules"] = set()
            try:
                self._model.detector = torch_tensorrt.compile(detector, **compile_kwargs)
            except Exception:
                return

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor."""
        return tensor

    @staticmethod
    def _tensor_to_png_b64(img: torch.Tensor) -> str:
        """
        Convert a CHW/BCHW tensor into a base64-encoded PNG.

        Accepts:
          - CHW (3,H,W) or (1,H,W)
        Returns:
          - base64 string (no data: prefix)
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(img)}")
        if img.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")

        x = img.detach()
        if x.device.type != "cpu":
            x = x.cpu()

        if x.dtype.is_floating_point:
            maxv = float(x.max().item()) if x.numel() else 1.0
            if maxv <= 1.5:
                x = x * 255.0
            x = x.clamp(0, 255).to(dtype=torch.uint8)
        else:
            x = x.clamp(0, 255).to(dtype=torch.uint8)

        c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])  # noqa: F841
        if c == 1:
            arr = x.squeeze(0).numpy()
            pil = Image.fromarray(arr, mode="L").convert("RGB")
        elif c == 3:
            arr = x.permute(1, 2, 0).contiguous().numpy()
            pil = Image.fromarray(arr, mode="RGB")
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {c}")

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _extract_text(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            for k in ("text", "output_text", "generated_text", "ocr_text"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            if "words" in obj and isinstance(obj["words"], list):
                parts: List[str] = []
                for w in obj["words"]:
                    if isinstance(w, dict) and isinstance(w.get("text"), str):
                        parts.append(w["text"])
                if parts:
                    return " ".join(parts).strip()
        return str(obj).strip()

    @nvtx.annotate("gpu_inference", color="blue")
    @nvtx.annotate("nrl.model::ocr_v2.inference", color="blue")
    def _infer(self, input_data: Any, *, merge_level: str) -> Any:
        return self._model(input_data, merge_level=merge_level)

    def invoke(
        self,
        input_data: Union[torch.Tensor, str, bytes, np.ndarray, io.BytesIO],
        merge_level: str = "paragraph",
    ) -> Any:
        """
        Invoke OCR locally.

        Supports:
          - file path (str) **only if it exists**
          - base64 (str/bytes) (str is treated as base64 unless it is an existing file path)
          - NumPy array (HWC)
          - io.BytesIO
          - torch.Tensor (CHW/BCHW): converted to base64 PNG internally for compatibility
        """
        if self._model is None:
            raise RuntimeError("Local OCR model was not initialized.")

        if isinstance(input_data, torch.Tensor):
            if input_data.ndim == 4:
                out: List[Any] = []
                for i in range(int(input_data.shape[0])):
                    b64 = self._tensor_to_png_b64(input_data[i])
                    out.extend(self._infer(b64.encode("utf-8"), merge_level=merge_level))
                return out
            if input_data.ndim == 3:
                b64 = self._tensor_to_png_b64(input_data)
                return self._infer(b64.encode("utf-8"), merge_level=merge_level)
            raise ValueError(f"Unsupported torch tensor shape for OCR: {tuple(input_data.shape)}")

        if isinstance(input_data, str):
            return self._infer(input_data.encode("utf-8"), merge_level=merge_level)

        return self._infer(input_data, merge_level=merge_level)

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "Nemotron OCR v2"

    @property
    def model_type(self) -> str:
        """Model category/type."""
        return "ocr"

    @property
    def model_runmode(self) -> ModelRunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        return "local"

    @property
    def input(self) -> Any:
        return {
            "type": "image",
            "format": "RGB",
            "supported_formats": ["PNG", "JPEG"],
            "data_types": ["float32", "uint8"],
            "dimensions": "variable (H x W)",
            "batch_support": True,
            "value_range": {"float32": "[0, 1]", "uint8": "[0, 255] (auto-converted)"},
            "aggregation_levels": ["word", "sentence", "paragraph"],
            "description": "Document or scene image in RGB format with automatic multi-scale resizing",
        }

    @property
    def output(self) -> Any:
        return {
            "type": "ocr_results",
            "format": "structured",
            "structure": {
                "boxes": "List[List[List[float]]] - quadrilateral bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]",  # noqa: E501
                "texts": "List[str] - recognized text strings",
                "confidences": "List[float] - confidence scores per detection",
            },
            "properties": {
                "reading_order": True,
                "layout_analysis": True,
                "multi_line_support": True,
                "multi_block_support": True,
            },
            "description": "Structured OCR results with bounding boxes, recognized text, and confidence scores",
        }

    @property
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        return 8
