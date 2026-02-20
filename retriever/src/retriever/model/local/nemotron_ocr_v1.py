from typing import Any, Dict, List, Optional, Tuple, Union

import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from ..model import BaseModel, RunMode

from PIL import Image


class NemotronOCRV1(BaseModel):
    """
    Nemotron OCR v1 model for optical character recognition.

    End-to-end OCR model that integrates:
    - Text detector for region localization
    - Text recognizer for transcription
    - Relational model for layout and reading order analysis
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        from nemotron_ocr.inference.pipeline import NemotronOCR  # local-only import

        if model_dir:
            self._model = NemotronOCR(model_dir=model_dir)
        else:
            self._model = NemotronOCR()
        # NemotronOCR is a high-level pipeline (not an nn.Module). We can optionally
        # TensorRT-compile individual submodules (e.g. the detector backbone) but
        # must keep post-processing (NMS, box decoding, etc.) in eager PyTorch/C++.
        self._enable_trt = os.getenv("SLIMGEST_ENABLE_TORCH_TRT", "").strip().lower() in {"1", "true", "yes", "on"}
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

        # Detector is the safest candidate: input is a BCHW image tensor.
        if self._model is None:
            return

        detector = getattr(self._model, "detector", None)
        if not isinstance(detector, torch.nn.Module):
            return

        # NemotronOCR internally resizes/pads to 1024 and runs B=1 (see upstream FIXME);
        # keep the TRT input shape fixed to avoid accidental batching issues.
        try:
            trt_input = torch_tensorrt.Input((1, 3, 1024, 1024), dtype=torch.float16)
        except TypeError:
            # Older/newer API variants: fall back to named arg.
            trt_input = torch_tensorrt.Input(shape=(1, 3, 1024, 1024), dtype=torch.float16)

        # If any torchvision NMS makes it into a compiled graph elsewhere, forcing
        # that op to run in Torch avoids hard failures.
        compile_kwargs: Dict[str, Any] = {
            "inputs": [trt_input],
            "enabled_precisions": {torch.float16},
        }
        if hasattr(torch_tensorrt, "compile"):
            for k in ("torch_executed_ops", "torch_executed_modules"):
                if k == "torch_executed_ops":
                    compile_kwargs[k] = {"torchvision::nms"}
                elif k == "torch_executed_modules":
                    compile_kwargs[k] = set()
            try:
                self._model.detector = torch_tensorrt.compile(detector, **compile_kwargs)
            except Exception:
                # Leave detector as-is on any failure.
                return

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor."""
        # no-op for now
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

        # Convert to uint8 in [0,255]
        if x.dtype.is_floating_point:
            maxv = float(x.max().item()) if x.numel() else 1.0
            # Heuristic: treat [0,1] images as normalized.
            if maxv <= 1.5:
                x = x * 255.0
            x = x.clamp(0, 255).to(dtype=torch.uint8)
        else:
            x = x.clamp(0, 255).to(dtype=torch.uint8)

        c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
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
            # Some APIs return nested structures; best-effort flatten common shapes.
            if "words" in obj and isinstance(obj["words"], list):
                parts: List[str] = []
                for w in obj["words"]:
                    if isinstance(w, dict) and isinstance(w.get("text"), str):
                        parts.append(w["text"])
                if parts:
                    return " ".join(parts).strip()
        return str(obj).strip()

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

        # Convert torch tensors to base64 bytes (NemotronOCR expects file path/base64/ndarray/BytesIO).
        if isinstance(input_data, torch.Tensor):
            if input_data.ndim == 4:
                out: List[Any] = []
                for i in range(int(input_data.shape[0])):
                    b64 = self._tensor_to_png_b64(input_data[i])
                    out.extend(self._model(b64.encode("utf-8"), merge_level=merge_level))
                return out
            if input_data.ndim == 3:
                b64 = self._tensor_to_png_b64(input_data)
                return self._model(b64.encode("utf-8"), merge_level=merge_level)
            raise ValueError(f"Unsupported torch tensor shape for OCR: {tuple(input_data.shape)}")

        # Disambiguate str: existing file path vs base64 string.
        if isinstance(input_data, str):
            # s = input_data.strip()
            # breakpoint()
            # if s and Path(s).is_file():
            #     return self._model(s, merge_level=merge_level)
            # Treat as base64 string (nemotron_ocr expects bytes for base64).
            return self._model(input_data.encode("utf-8"), merge_level=merge_level)

        # bytes / ndarray / BytesIO are supported directly by nemotron_ocr.
        return self._model(input_data, merge_level=merge_level)

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "Nemotron OCR v1"

    @property
    def model_type(self) -> str:
        """Model category/type."""
        return "ocr"

    @property
    def model_runmode(self) -> RunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        return "local"

    @property
    def input(self) -> Any:
        """
        Input schema for the model.

        Returns:
            dict: Schema describing RGB image input with variable dimensions
        """
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
        """
        Output schema for the model.

        Returns:
            dict: Schema describing OCR output format
        """
        return {
            "type": "ocr_results",
            "format": "structured",
            "structure": {
                "boxes": "List[List[List[float]]] - quadrilateral bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]",
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
