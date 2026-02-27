# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Sequence, Tuple, Union, cast  # noqa: F401

from torch import nn
import torch
import numpy as np
from ..model import HuggingFaceModel, RunMode

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_page_elements_v3.model import resize_pad as resize_pad_page_elements
from nemotron_page_elements_v3.utils import postprocess_preds_page_element as postprocess_preds_page_element


class NemotronPageElementsV3(HuggingFaceModel):
    """
    Nemotron Page Elements v3 model for detecting document page elements.

    Detects and extracts structural elements from document pages including:
    - Tables
    - Charts
    - Infographics
    - Titles
    - Headers/Footers
    - Text regions
    """

    def __init__(self) -> None:
        super().__init__(self.model_name)
        self._model = define_model_page_elements(self.model_name)
        self._page_elements_input_shape = (1024, 1024)

    def preprocess(self, tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess the input tensor."""
        # Upstream `nemotron_page_elements_v3.model.resize_pad` expects a CHW tensor.
        # Our pipeline helpers often pass BCHW (typically B=1), so normalize here.
        if isinstance(tensor, torch.Tensor):
            x = tensor
        elif isinstance(tensor, np.ndarray):
            arr = tensor
            # Normalize to CHW tensor.
            if arr.ndim == 4 and int(arr.shape[0]) == 1:
                arr = arr[0]
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D image array, got shape {getattr(arr, 'shape', None)}")
            # Heuristic: HWC (RGB) -> CHW; otherwise assume CHW-like.
            if int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
                x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
            else:
                x = torch.from_numpy(np.ascontiguousarray(arr))
            # Keep 0-255 range: resize_pad pads with 114.0 (designed for 0-255),
            # and YoloXWrapper.forward() handles the 0-255 → model-input conversion.
            x = x.to(dtype=torch.float32)
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)!r}")

        if x.ndim == 4:
            b = int(x.shape[0])
            if b == 1:
                x_chw = x[0]
                y = resize_pad_page_elements(x_chw, self.input_shape)
                if not isinstance(y, torch.Tensor):
                    raise TypeError(f"resize_pad returned non-tensor: {type(y)!r}")
                if y.ndim != 3:
                    raise ValueError(f"Expected CHW from resize_pad, got {tuple(y.shape)}")
                return y.unsqueeze(0)

            outs: List[torch.Tensor] = []
            for i in range(b):
                y = resize_pad_page_elements(x[i], self.input_shape)
                if not isinstance(y, torch.Tensor) or y.ndim != 3:
                    raise ValueError(f"resize_pad produced unexpected output for batch item {i}: {type(y)!r}")
                outs.append(y)
            return torch.stack(outs, dim=0)

        if x.ndim == 3:
            y = resize_pad_page_elements(x, self.input_shape)
            if not isinstance(y, torch.Tensor):
                raise TypeError(f"resize_pad returned non-tensor: {type(y)!r}")
            if y.ndim != 3:
                raise ValueError(f"Expected CHW from resize_pad, got {tuple(y.shape)}")
            return y.unsqueeze(0)

        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(x.shape)}")

    def __call__(
        self, input_data: torch.Tensor, orig_shape: Union[Tuple[int, int], Sequence[Tuple[int, int]]]
    ) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        return self.invoke(input_data, orig_shape)

    def invoke(
        self, input_data: torch.Tensor, orig_shape: Union[Tuple[int, int], Sequence[Tuple[int, int]]]
    ) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")

        # The upstream model returns a container where index [0] is the predictions.
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                out = self._model(input_data, orig_shape)
                return out
        # preds0: Any
        # if isinstance(out, (list, tuple)) and len(out) > 0:
        #     preds0 = out[0]
        # else:
        #     preds0 = out

        # # If we got per-image preds, return them as-is so pipeline code can
        # # avoid slow per-image fallback.
        # if isinstance(preds0, list):
        #     # If caller passed a batch, keep list; if single-item batch, unwrap to dict for back-compat.
        #     if isinstance(orig_shape, (list, tuple)) and len(orig_shape) == 1:
        #         one = preds0[0] if preds0 else {}
        #         return cast(Dict[str, torch.Tensor], one)
        #     return cast(List[Dict[str, torch.Tensor]], preds0)

        # return cast(Dict[str, torch.Tensor], preds0)

    def postprocess(self, preds: Union[Dict[str, torch.Tensor], Sequence[Dict[str, torch.Tensor]]]) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
    ]:
        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")

        # Upstream `nemotron_page_elements_v3.utils.postprocess_preds_page_element` expects a
        # single prediction dict (with torch tensors) and returns numpy arrays. Our pipeline
        # may pass a *list* of per-image preds for batched inference, so handle both cases
        # and always return torch tensors (or lists of torch tensors).

        def _one(p: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            b_np, l_np, s_np = postprocess_preds_page_element(p, self._model.thresholds_per_class, self._model.labels)
            b = torch.as_tensor(b_np, dtype=torch.float32)
            l = torch.as_tensor(l_np, dtype=torch.int64)  # noqa: E741
            s = torch.as_tensor(s_np, dtype=torch.float32)
            return b, l, s

        if isinstance(preds, dict):
            return _one(preds)

        boxes_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        for p in preds:
            b, l, s = _one(p)
            boxes_list.append(b)
            labels_list.append(l)
            scores_list.append(s)

        return boxes_list, labels_list, scores_list

    @property
    def model_dir(self) -> str:
        """Model directory."""
        return self.model_name

    @property
    def model(self) -> nn.Module:
        """Model instance."""
        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")
        return self._model

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "page_element_v3"

    @property
    def model_type(self) -> str:
        """Model category/type."""
        return "object-detection"

    @property
    def model_runmode(self) -> RunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        return "local"

    @property
    def input(self) -> Any:
        """
        Input schema for the model.

        Returns:
            dict: Schema describing RGB image input with dimensions (1024, 1024)
        """
        return {
            "type": "image",
            "format": "RGB",
            "dimensions": "(1024, 1024)",
            "description": "Document page image in RGB format, resized to 1024x1024",
        }

    @property
    def output(self) -> Any:
        """
        Output schema for the model.

        Returns:
            dict: Schema describing detection output format
        """
        return {
            "type": "detection",
            "format": "dict",
            "structure": {
                "boxes": "np.ndarray[N, 4] - normalized (x_min, y_min, x_max, y_max)",
                "labels": "List[str] - class names",
                "scores": "np.ndarray[N] - confidence scores",
            },
            "classes": ["table", "chart", "infographic", "title", "text", "header_footer"],
            "post_processing": {"conf_thresh": 0.01, "iou_thresh": 0.5},
        }

    @property
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        return 1

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Input shape."""
        return self._page_elements_input_shape
