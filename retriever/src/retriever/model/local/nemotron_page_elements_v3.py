from typing import Any, Dict, List, Optional, Tuple, Union

from torch import nn
import torch
from ..model import HuggingFaceModel, RunMode

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_page_elements_v3.model import resize_pad as resize_pad_page_elements
from nemotron_page_elements_v3.utils import postprocess_preds_page_element as postprocess_preds_page_element

from ..nim.http_utils import (
    coerce_boxes_labels_scores,
    default_headers,
    extract_per_item_list,
    normalize_endpoint,
    post_batched_images,
)


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

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        timeout_seconds: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        remote_batch_size: int = 32,
    ) -> None:
        super().__init__(self.model_name)
        self._endpoint = normalize_endpoint(endpoint) if endpoint else None
        self._timeout_seconds = float(timeout_seconds)
        self._headers = default_headers(headers)
        self._remote_batch_size = int(remote_batch_size)

        self._model = None
        if self._endpoint is None:
            self._model = define_model_page_elements(self.model_name)
        self._page_elements_input_shape = (1024, 1024)

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor."""
        # Delegate activities to the HuggingFace model itself which implements logic for resizing and padding
        # There is a high likelihood that further logic for performance improvements can be done
        # here therefore that is why this shim layer is present to allow for that opportunity
        # without modifications to the HuggingFace model itself
        return resize_pad_page_elements(tensor, self.input_shape)

    def invoke(
        self, input_data: torch.Tensor, orig_shape: Tuple[int, int]
    ) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        # Remote: return already-postprocessed detections dict {boxes, labels, scores}
        if self._endpoint is not None:
            return self.invoke_remote(input_data)

        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")

        # Conditionally check and make sure the input data is on the correct device and shape
        return self._model(input_data, orig_shape)[0]

    def invoke_remote(
        self,
        input_data: Union[torch.Tensor, List[torch.Tensor]],
        *,
        batch_size: Optional[int] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Invoke the page elements NIM endpoint over HTTP.

        - If CHW: returns a single dict {boxes, labels, scores}.
        - If BCHW: returns a list of dicts (len == B).
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
            raise ValueError(f"Unsupported input type/shape for remote page elements: {type(input_data)}")

        per_image_out: List[Dict[str, torch.Tensor]] = []
        for i in range(0, len(imgs), bs):
            chunk = imgs[i : i + bs]
            data = post_batched_images(
                endpoint=self._endpoint,
                images_chw=chunk,
                headers=self._headers,
                timeout_seconds=self._timeout_seconds,
            )
            per_list = extract_per_item_list(data)
            if per_list is None:
                if len(chunk) != 1:
                    raise RuntimeError("Remote page elements response did not contain a per-image list.")
                per_list = [data]
            if len(per_list) != len(chunk):
                raise RuntimeError(
                    f"Remote page elements response size mismatch: got {len(per_list)} outputs for {len(chunk)} inputs."
                )
            for item in per_list:
                boxes, labels, scores = coerce_boxes_labels_scores(item)
                per_image_out.append({"boxes": boxes, "labels": labels, "scores": scores})

        return per_image_out[0] if (isinstance(input_data, torch.Tensor) and input_data.ndim == 3) else per_image_out

    def postprocess(self, preds: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Remote path: invoke() returns already-postprocessed dict, so passthrough.
        if isinstance(preds, dict) and all(k in preds for k in ("boxes", "labels", "scores")):
            return preds["boxes"], preds["labels"], preds["scores"]

        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")

        boxes, labels, scores = postprocess_preds_page_element(
            preds, self._model.thresholds_per_class, self._model.labels
        )
        return boxes, labels, scores

    @property
    def model_dir(self) -> str:
        """Model directory."""
        pass

    @property
    def model(self) -> nn.Module:
        """Model instance."""
        if self._model is None:
            raise RuntimeError("Model is remote-only; no local nn.Module is loaded.")
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
        return "NIM" if self._endpoint is not None else "local"

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
        return 32 if self._endpoint is not None else 1

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Input shape."""
        return self._page_elements_input_shape
