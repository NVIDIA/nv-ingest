from typing import Any, Dict, List, Tuple, Union

import torch
from ..model import BaseModel, RunMode

from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_table_structure_v1.model import resize_pad as resize_pad_table_structure


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

    def __init__(self) -> None:
        super().__init__()
        # table_structure_model = define_model_table_structure("table_structure_v1")
        self._model = define_model_table_structure(self.model_name)
        self._table_structure_input_shape = (1024, 1024)

    def preprocess(self, input_tensor: torch.Tensor, orig_shape: Tuple[int, int]) -> torch.Tensor:
        """Preprocess the input tensor.

        Upstream `nemotron_table_structure_v1.model.resize_pad` expects CHW.
        The pipeline often passes BCHW (typically B=1); normalize here.
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(input_tensor)!r}")

        x = input_tensor
        if x.ndim == 4:
            b = int(x.shape[0])
            if b == 1:
                y = resize_pad_table_structure(x[0], self.input_shape)
                if not isinstance(y, torch.Tensor) or y.ndim != 3:
                    raise ValueError(f"resize_pad produced unexpected output: {type(y)!r}")
                return y.unsqueeze(0)
            outs = []
            for i in range(b):
                y = resize_pad_table_structure(x[i], self.input_shape)
                if not isinstance(y, torch.Tensor) or y.ndim != 3:
                    raise ValueError(f"resize_pad produced unexpected output for batch item {i}: {type(y)!r}")
                outs.append(y)
            return torch.stack(outs, dim=0)

        if x.ndim == 3:
            y = resize_pad_table_structure(x, self.input_shape)
            if not isinstance(y, torch.Tensor) or y.ndim != 3:
                raise ValueError(f"resize_pad produced unexpected output: {type(y)!r}")
            return y.unsqueeze(0)

        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(x.shape)}")

    def invoke(
        self, input_tensor: torch.Tensor, orig_shape: Tuple[int, int]
    ) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if self._model is None:
            raise RuntimeError("Local table_structure_v1 model was not initialized.")

        table_preds = self._model(input_tensor, orig_shape)[0]
        return table_preds

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
        return "local"

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
            "description": "Table image in RGB format, resized to 1024x1024. Should be cropped from document using page element detection.",  # noqa: E501
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
        return 1

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Input shape."""
        return self._table_structure_input_shape
