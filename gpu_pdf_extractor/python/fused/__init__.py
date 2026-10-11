"""Fused GPU operator prototype (P3'): compose existing operators in one process so device
tensors hand off zero-copy (via DLPack) instead of being serialized between Ray stages."""
from .operators import (
    FusedGPUOperator,
    RasterizeGPUOperator,
    PageElementStubGPUOperator,
    PageElementGPUOperator,
    CropGPUOperator,
    OCRStubGPUOperator,
    TableStructureGPUOperator,
    GraphicElementsGPUOperator,
    OCRGPUOperator,
    HostFinalizeOperator,
)

__all__ = [
    "FusedGPUOperator", "RasterizeGPUOperator", "PageElementStubGPUOperator",
    "PageElementGPUOperator", "CropGPUOperator", "OCRStubGPUOperator",
    "TableStructureGPUOperator", "GraphicElementsGPUOperator", "OCRGPUOperator",
    "HostFinalizeOperator",
]
