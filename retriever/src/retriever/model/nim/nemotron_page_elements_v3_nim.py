from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Tuple, Union

from numpy import float32, float64
import requests, base64

from torch import nn
import torch

from ..model import NvidiaNIMModel, RunMode


class NemotronPageElementsV3NIM(NvidiaNIMModel):
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
        self._endpoint_url = "http://localhost:8000/v1/infer"

    def preprocess(self, image: Union[Path, str, BytesIO, torch.Tensor]) -> BytesIO:
        """Preprocess the input image: loads, converts to RGB, encodes as base64, and returns as BytesIO.
        Supports Path, str filename, BytesIO, or torch.Tensor (C,H,W or H,W,C)."""
        # Load image
        if isinstance(image, BytesIO):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, (str, Path)):
            img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Move tensor to cpu and convert to numpy
            tensor = image.detach().cpu()
            # If batch dimension present, remove it
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]
            # Handle (C,H,W) or (H,W,C)
            if tensor.ndim == 3:
                if tensor.shape[0] in (1, 3):  # (C,H,W)
                    npimg = tensor.permute(1, 2, 0).numpy()
                else:  # probably (H,W,C)
                    npimg = tensor.numpy()
            else:
                raise ValueError("Tensor must have 3 or 4 dimensions of an image")
            # Handle normalization/scaling if needed (assume [0,1] or [0,255])
            if npimg.dtype in [float, float32, float64] or npimg.max() <= 1.0:
                npimg = (npimg * 255).clip(0, 255).astype("uint8")
            else:
                npimg = npimg.astype("uint8")
            img = Image.fromarray(npimg).convert("RGB")
        else:
            raise ValueError("Input must be a Path, str filename, BytesIO image object, or torch.Tensor")

        # Save to bytes (RGB, PNG to preserve all data)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_bytes = buffer.read()

        # Encode to base64
        img_base64 = base64.b64encode(img_bytes)

        # Return as BytesIO
        return BytesIO(img_base64)

    def invoke(self, input_data: Union[Path, str, BytesIO]) -> Any:
        # If input_data is not a str, convert it to str (base64 encoded string)
        if not isinstance(input_data, str):
            if isinstance(input_data, BytesIO):
                input_data = input_data.read()
            elif hasattr(input_data, "read"):  # Some file-like object
                input_data = input_data.read()
            # At this point, input_data is bytes; decode to str
            if isinstance(input_data, bytes):
                input_data = input_data.decode("utf-8")
            else:
                input_data = str(input_data)
        payload = {"input": [{"type": "image_url", "url": f"data:image/png;base64,{input_data}"}]}

        response = requests.post(self._endpoint_url, json=payload)
        return response.json()

    def postprocess(
        self, preds: Dict[str, Any]
    ) -> Tuple[Union[List[str], torch.Tensor], Union[List[str], torch.Tensor], Union[List[str], torch.Tensor]]:

        boxes = []
        labels = []
        scores = []

        for pred in preds["data"]:
            breakpoint()
            print(pred)

        return boxes, labels, scores

    @property
    def model_dir(self) -> str:
        """Model directory."""
        pass

    @property
    def model(self) -> nn.Module:
        """Model instance."""
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
