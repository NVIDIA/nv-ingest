from typing import Any, Dict, List, Optional, Tuple, Union

import base64
import io
import os

import httpx
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
        model_dir: str,
        *,
        endpoint: Optional[str] = None,
        timeout_seconds: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        remote_batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._endpoint = self._normalize_endpoint(endpoint) if endpoint else None
        self._timeout_seconds = float(timeout_seconds)
        self._headers = dict(headers or {})
        self._remote_batch_size = int(remote_batch_size)

        # If a remote endpoint is provided, skip loading local weights (saves GPU memory).
        self._model = None
        if self._endpoint is None:
            from nemotron_ocr.inference.pipeline import NemotronOCR  # local-only import

            self._model = NemotronOCR(model_dir=model_dir)
        # NemotronOCR is a high-level pipeline (not an nn.Module). We can optionally
        # TensorRT-compile individual submodules (e.g. the detector backbone) but
        # must keep post-processing (NMS, box decoding, etc.) in eager PyTorch/C++.
        self._enable_trt = os.getenv("SLIMGEST_ENABLE_TORCH_TRT", "").strip().lower() in {"1", "true", "yes", "on"}
        if self._enable_trt and self._model is not None:
            self._maybe_compile_submodules()

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        ep = (endpoint or "").strip()
        if not ep:
            return ep
        if "://" not in ep:
            ep = "http://" + ep
        return ep

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

    def invoke_remote(
        self,
        input_data: Union[torch.Tensor, List[torch.Tensor]],
        *,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Invoke the OCR NIM endpoint over HTTP.

        Contract:
          - If input is CHW: returns a list of length 1.
          - If input is BCHW: returns a list of length B (one dict per image).
        """
        if not self._endpoint:
            raise RuntimeError("invoke_remote() called but no endpoint was configured.")

        bs = int(batch_size or self._remote_batch_size or 32)
        if isinstance(input_data, list):
            imgs = input_data
        elif isinstance(input_data, torch.Tensor) and input_data.ndim == 4:
            imgs = [input_data[i] for i in range(int(input_data.shape[0]))]
        elif isinstance(input_data, torch.Tensor) and input_data.ndim == 3:
            imgs = [input_data]
        else:
            raise ValueError(f"Unsupported input type/shape for remote OCR: {type(input_data)}")

        headers = {"Accept": "application/json", **(self._headers or {})}

        # If the user is hitting Nvidia hosted APIs, allow env-based key automatically.
        # For local docker NIM you typically do not need this.
        if "Authorization" not in headers:
            api_key = os.getenv("NVIDIA_API_KEY", "").strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        out: List[Dict[str, Any]] = []
        with httpx.Client(timeout=self._timeout_seconds) as client:
            for i in range(0, len(imgs), bs):
                chunk = imgs[i : i + bs]
                chunk_b64 = [self._tensor_to_png_b64(t) for t in chunk]
                payload = {"input": [{"type": "image_url", "url": f"data:image/png;base64,{b64}"} for b64 in chunk_b64]}

                resp = client.post(self._endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

                per_image: Optional[List[Any]] = None
                if isinstance(data, dict):
                    for k in ("data", "output", "outputs", "results", "result"):
                        v = data.get(k)
                        if isinstance(v, list):
                            per_image = v
                            break

                # If the API returns a single object, only accept it for single-image calls.
                if per_image is None:
                    if len(chunk) != 1:
                        raise RuntimeError(
                            f"Remote OCR response did not contain a per-image list for batch size {len(chunk)}."
                        )
                    per_image = [data]

                if len(per_image) != len(chunk):
                    raise RuntimeError(
                        f"Remote OCR response size mismatch: got {len(per_image)} outputs for {len(chunk)} inputs."
                    )

                for item in per_image:
                    out.append({"text": self._extract_text(item), "raw": item})

        return out

    def invoke(self, input_data: torch.Tensor, merge_level: str = "paragraph") -> List[Dict[str, torch.Tensor]]:
        if self._endpoint is not None:
            # Remote NIM invocation. Note: returns list[dict] (not torch tensors).
            return self.invoke_remote(input_data, batch_size=self._remote_batch_size)  # type: ignore[return-value]

        if self._model is None:
            raise RuntimeError("Local OCR model was not initialized.")

        # NemotronOCR expects a single image tensor (CHW). If a batch (BCHW) is
        # provided, run per-image to keep behavior correct.
        if isinstance(input_data, torch.Tensor) and input_data.ndim == 4:
            out: List[Dict[str, torch.Tensor]] = []
            for i in range(int(input_data.shape[0])):
                out.extend(self._model(input_data[i]))
            return out

        results = self._model(input_data, merge_level=merge_level)
        return results

    def postprocess(self, preds: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, labels, scores = postprocess_preds_page_element(
            preds, self._model.thresholds_per_class, self._model.labels
        )
        return boxes, labels, scores

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
        return "NIM" if self._endpoint is not None else "local"

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
        return 32 if self._endpoint is not None else 8
