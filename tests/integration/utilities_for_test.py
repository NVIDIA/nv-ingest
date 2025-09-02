# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re


def canonicalize_markdown_table(table_string):
    """Converts a markdown table string to a canonical format.

    This function takes a string potentially containing a Markdown table
    (which might have inconsistent spacing, tabs, asterisks for formatting,
    and a separator line) and transforms it into a standardized format.

    The canonical format ensures:
      - The header/separator line (e.g., |:---|:---|) is removed.
      - Leading/trailing whitespace around the entire table is removed.
      - Empty lines within the input are ignored.
      - Each row starts and ends with '| '.
      - Cells within a row are separated by ' | '.
      - Leading/trailing whitespace within each cell is removed.
      - Internal whitespace sequences (spaces, tabs) within cells are
        collapsed into a single space.
      - Asterisk characters ('*') are removed from cell content.

    Args:
      table_string: The input string containing the Markdown table.

    Returns:
      A string representing the table in a canonical, cleaned format,
      with rows joined by newline characters. Returns an empty string if
      the input only contained whitespace or separator lines.

    Example:
        >>> messy_table = '''
        ... | Animal*   | Activity                  | Place      |
        ... |:----------|:--------------------------|:-----------|
        ... | Giraffe   | Driving \t a car         | At the beach |
        ... '''
        >>> canonicalize_markdown_table(messy_table)
        '| Animal | Activity | Place |\n| Giraffe | Driving a car | At the beach |'
    """
    lines = table_string.strip().splitlines()
    # Filter out separator lines and process data/header lines
    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("|:--"):
            continue  # Skip empty lines or separator lines
        # Clean cells: strip whitespace, collapse internal whitespace/tabs
        cells = [" ".join(cell.replace("*", "").strip().split()) for cell in stripped_line.strip("|").split("|")]
        processed_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(processed_lines)


# ------------------------------
# Similarity utilities (duplicated for integration tests)
# ------------------------------


def _tokenize(s: str):
    r"""
    Tokenize a string into lowercase word tokens (alphanumeric + underscore).

    Uses regex "\w+" to be robust to punctuation/spacing differences.  # noqa: W605
    """
    return re.findall(r"\w+", (s or "").lower())


def jaccard_similarity(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity between strings a and b.

    J(A,B) = |A ∩ B| / |A ∪ B|, where A and B are sets of tokens.
    Returns 1.0 if both are empty.
    """
    A = set(_tokenize(a))
    B = set(_tokenize(b))
    if not A and not B:
        return 1.0
    union = A | B
    if not union:
        return 0.0
    inter = A & B
    return len(inter) / len(union)


def token_f1(a: str, b: str) -> float:
    """
    Token-level F1 between strings a and b using set tokens.

    precision = |A ∩ B| / |A|
    recall    = |A ∩ B| / |B|
    F1 = 2 * P * R / (P + R), with edge cases handled.
    """
    A = set(_tokenize(a))
    B = set(_tokenize(b))
    if not A and not B:
        return 1.0
    inter = len(A & B)
    precision = inter / len(A) if A else 0.0
    recall = inter / len(B) if B else 0.0
    denom = precision + recall
    return (2 * precision * recall / denom) if denom else 0.0


def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings without external deps.

    Operations: insert, delete, substitute (cost 1 each).
    """
    if a is None:
        a = ""
    if b is None:
        b = ""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    if n > m:
        a, b = b, a
        n, m = m, n

    previous = list(range(n + 1))
    for j in range(1, m + 1):
        current = [j] + [0] * n
        bj = b[j - 1]
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == bj else 1
            current[i] = min(
                previous[i] + 1, current[i - 1] + 1, previous[i - 1] + cost  # deletion  # insertion  # substitution
            )
        previous = current
    return previous[n]


def levenshtein_ratio(a: str, b: str) -> float:
    """
    Character-level similarity derived from Levenshtein distance.

    ratio = 1 - dist / max_len, where max_len = max(len(a), len(b)).
    Returns 1.0 if both inputs are empty.
    """
    a = a or ""
    b = b or ""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - (dist / max_len)
