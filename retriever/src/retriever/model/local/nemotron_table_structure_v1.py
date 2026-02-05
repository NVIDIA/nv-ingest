from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from ..model import BaseModel, RunMode

from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_table_structure_v1.model import resize_pad as resize_pad_table_structure

from .http_utils import default_headers, normalize_endpoint


class NemotronTableStructureV1(BaseModel):
    """
    Nemotron Table Structure v1 model for detecting table structure elements.

    Detects and extracts structural components from table images:
    - Individual cells (including merged cells)
    - Rows
    - Columns

    Note: Input should be a cropped table image, typically obtained using
    Nemotron Page Elements v3 model.
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        timeout_seconds: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        remote_batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._endpoint = normalize_endpoint(endpoint) if endpoint else None
        self._timeout_seconds = float(timeout_seconds)
        self._headers = default_headers(headers)
        self._remote_batch_size = int(remote_batch_size)

        self._model = None
        if self._endpoint is None:
            # table_structure_model = define_model_table_structure("table_structure_v1")
            self._model = define_model_table_structure(self.model_name)
        self._table_structure_input_shape = (1024, 1024)

    def preprocess(self, input_tensor: torch.Tensor, orig_shape: Tuple[int, int]) -> torch.Tensor:
        """Preprocess the input tensor."""
        return resize_pad_table_structure(input_tensor, self.input_shape)

    def invoke(
        self, input_tensor: torch.Tensor, orig_shape: Tuple[int, int]
    ) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if self._endpoint is not None:
            return self.invoke_remote(input_tensor)

        if self._model is None:
            raise RuntimeError("Local table_structure_v1 model was not initialized.")

        table_preds = self._model(input_tensor, orig_shape)[0]
        return table_preds

    def invoke_remote(
        self,
        input_data: Union[torch.Tensor, List[torch.Tensor]],
        *,
        batch_size: Optional[int] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Invoke the table structure NIM endpoint over HTTP.

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
            raise ValueError(f"Unsupported input type/shape for remote table structure: {type(input_data)}")

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
                    raise RuntimeError("Remote table structure response did not contain a per-image list.")
                per_list = [data]
            if len(per_list) != len(chunk):
                raise RuntimeError(
                    f"Remote table structure response size mismatch: got {len(per_list)} outputs for {len(chunk)} inputs."
                )
            for item in per_list:
                boxes, labels, scores = coerce_boxes_labels_scores(item)
                per_image_out.append({"boxes": boxes, "labels": labels, "scores": scores})

        return per_image_out[0] if (isinstance(input_data, torch.Tensor) and input_data.ndim == 3) else per_image_out

    def postprocess(self, preds: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, labels, scores = postprocess_preds_page_element(
            preds, self._model.thresholds_per_class, self._model.labels
        )
        return boxes, labels, scores

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "table_structure_v1"

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
            dict: Schema describing RGB table image input
        """
        return {
            "type": "image",
            "format": "RGB",
            "dimensions": "(1024, 1024)",
            "expected_content": "table",
            "description": "Table image in RGB format, resized to 1024x1024. Should be cropped from document using page element detection.",
            "preprocessing_recommendation": "Use Nemotron Page Elements v3 to detect and crop table regions",
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
            "classes": ["cell", "row", "column"],
            "post_processing": {"conf_thresh": 0.01, "iou_thresh": 0.25},
            "use_cases": ["Table-to-markdown conversion", "Structured data extraction", "Table analysis"],
            "known_limitations": [
                "Less confident in detecting cells at bottom of table",
                "Not robust to rotated tables",
                "No support for non full-page tables",
            ],
        }

    @property
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        return 32 if self._endpoint is not None else 1

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Input shape."""
        return self._table_structure_input_shape
