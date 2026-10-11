"""Subset of pypdfium2.raw constants the nemo_retriever codebase consumes.

Values are the stable PDFium ABI page-object type enums. Mirrors `import pypdfium2.raw`.
"""
FPDF_PAGEOBJ_UNKNOWN = 0
FPDF_PAGEOBJ_TEXT = 1
FPDF_PAGEOBJ_PATH = 2
FPDF_PAGEOBJ_IMAGE = 3
FPDF_PAGEOBJ_SHADING = 4
FPDF_PAGEOBJ_FORM = 5

# Bitmap formats (parity with pypdfium2.raw.FPDFBitmap_*), exposed for completeness.
FPDFBitmap_Unknown = 0
FPDFBitmap_Gray = 1
FPDFBitmap_BGR = 2
FPDFBitmap_BGRx = 3
FPDFBitmap_BGRA = 4
