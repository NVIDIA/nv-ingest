from abc import ABC, abstractmethod
from typing import Any, Literal, Tuple
import torch.nn as nn


RunMode = Literal["local", "NIM", "build-endpoint"]


class BaseModel(ABC):
    """
    Abstract base class for all models.

    This class must never be instantiated directly.
    """

    def __init__(self) -> None:
        if type(self) is BaseModel:
            raise TypeError("BaseModel is abstract and cannot be instantiated directly")

    # ---- Required metadata ----

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model category/type (e.g. llm, vision, embedding)."""
        pass

    @property
    @abstractmethod
    def model_runmode(self) -> RunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        pass

    # ---- I/O contract ----

    @property
    @abstractmethod
    def input(self) -> Any:
        """Input schema or object."""
        pass

    @property
    @abstractmethod
    def output(self) -> Any:
        """Output schema or object."""
        pass

    @property
    @abstractmethod
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        pass


class HuggingFaceModel(BaseModel):
    """
    Abstract base class for all HuggingFace models.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__()

    @property
    @abstractmethod
    def model_dir(self) -> str:
        """Model directory."""
        pass

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """Model instance."""
        pass

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, int]:
        """Input shape."""
        pass


class NvidiaNIMModel(BaseModel):
    """
    Abstract base class for all Nvidia NIM models.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__()

    @property
    @abstractmethod
    def model_dir(self) -> str:
        """Model directory."""
        pass
