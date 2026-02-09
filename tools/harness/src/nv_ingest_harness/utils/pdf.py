"""PDF utility functions using pypdfium2.

This module contains utilities that depend on pypdfium2 and are only needed
for PDF-based benchmarking workflows.
"""

import glob
import os

import pypdfium2 as pdfium


def pdf_page_count_glob(pattern: str) -> int:
    """Count the total number of pages in all PDF files matching a glob pattern.

    This function searches for PDF files using a glob pattern and counts the total
    number of pages across all matching files.

    Args:
        pattern (str): A glob pattern to match PDF files (supports recursive search).

    Returns:
        int: The total number of pages across all matching PDF files.
    """
    total_pages = 0
    for filepath in glob.glob(pattern, recursive=True):
        if filepath.lower().endswith(".pdf"):
            pdf = pdfium.PdfDocument(filepath)
            total_pages += len(pdf)
    return total_pages


def pdf_page_count(path: str) -> int:
    """Count the total number of pages in PDF file(s).

    This function handles both single PDF files and directories containing PDFs.
    For a file path, it counts pages in that specific PDF.
    For a directory path, it recursively scans for all PDF files and counts total pages.
    If a PDF file cannot be processed, an error message is printed and the file is skipped.

    Args:
        path (str): Path to a PDF file or directory containing PDF files.

    Returns:
        int: The total number of pages across all PDF files.
    """
    total_pages = 0

    # Handle single file
    if os.path.isfile(path):
        if path.lower().endswith(".pdf"):
            try:
                pdf = pdfium.PdfDocument(path)
                total_pages = len(pdf)
            except Exception as e:
                print(f"{path} failed: {e}")
        return total_pages

    # Handle directory - recursively search for PDFs
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    filepath = os.path.join(root, filename)
                    try:
                        pdf = pdfium.PdfDocument(filepath)
                        total_pages += len(pdf)
                    except Exception as e:
                        print(f"{filepath} failed: {e}")
                        continue

    return total_pages
