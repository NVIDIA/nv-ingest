"""
Local model implementations for slim-gest.

This module contains implementations of locally-runnable models that extend
the BaseModel abstract class.
"""

from .nemotron_page_elements_v3 import NemotronPageElementsV3
from .nemotron_ocr_v1 import NemotronOCRV1
from .nemotron_table_structure_v1 import NemotronTableStructureV1
from .nemotron_graphic_elements_v1 import NemotronGraphicElementsV1

__all__ = [
    "NemotronPageElementsV3",
    "NemotronOCRV1",
    "NemotronTableStructureV1",
    "NemotronGraphicElementsV1",
]
