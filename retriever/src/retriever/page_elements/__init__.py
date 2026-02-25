"""
Page element detection stage (Nemotron Page Elements v3).

This package provides:
- `detect_page_elements_v3`: pure-Python batch function (pandas.DataFrame in/out)
- `PageElementDetectionActor`: Ray-friendly callable that initializes the model once
"""

from .page_elements import PageElementDetectionActor, detect_page_elements_v3

__all__ = [
    "detect_page_elements_v3",
    "PageElementDetectionActor",
]
