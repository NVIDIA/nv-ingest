from typing import Any, Dict, List, Tuple, Union

from torch import nn
import torch
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
        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")

        # Conditionally check and make sure the input data is on the correct device and shape
        return self._model(input_data, orig_shape)[0]

    def postprocess(self, preds: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._model is None:
            raise RuntimeError("Local page_elements_v3 model was not initialized.")

        boxes, labels, scores = postprocess_preds_page_element(
            preds, self._model.thresholds_per_class, self._model.labels
        )
        return boxes, labels, scores

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
