# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import re


def get_git_root(file_path):
    """
    Attempts to get the root of the git repository for the given file.

    Parameters:
        file_path (str): The path of the file to determine the git root for.

    Returns:
        str: The absolute path to the git repository root if found,
             otherwise None.
    """
    try:
        # Get the absolute directory of the file.
        directory = os.path.dirname(os.path.abspath(file_path))
        # Run the git command to get the repository's top-level directory.
        git_root = (
            subprocess.check_output(["git", "-C", directory, "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT)
            .strip()
            .decode("utf-8")
        )
        return git_root
    except subprocess.CalledProcessError:
        # In case the file is not inside a git repository.
        return None
    except Exception:
        return None


def find_root_by_pattern(pattern, start_dir=None):
    """
    Backtracks up the directory tree looking for the first directory
    where the specified pattern exists.

    Parameters:
        pattern (str): The relative path to check for (e.g., "data/test.pdf").
        start_dir (str, optional): The starting directory. Defaults to the current working directory.

    Returns:
        str: The absolute path of the first directory where pattern exists,
             or None if not found.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    current_dir = os.path.abspath(start_dir)

    while True:
        candidate = os.path.join(current_dir, pattern)
        if os.path.exists(candidate):
            return current_dir

        # Determine the parent directory.
        parent_dir = os.path.dirname(current_dir)
        # If we're at the filesystem root, break.
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    return None


def get_project_root(file_path):
    """
    Attempts to get the project root directory, trying git first and falling back to pattern matching.

    This function first tries to use git to find the repository root. If that fails (e.g., not in a git
    repository or git not available), it falls back to searching for common project patterns.

    Parameters:
        file_path (str): The path of the file to determine the project root for.

    Returns:
        str: The absolute path to the project root if found, otherwise None.
    """
    # First try git-based detection
    git_root = get_git_root(file_path)
    if git_root is not None:
        return git_root

    # Fall back to pattern-based detection
    start_dir = os.path.dirname(os.path.abspath(file_path))

    # Try common project patterns in order of preference
    patterns = [
        "data",  # Look for data directory (common in this project)
        "pyproject.toml",  # Python project file
        "setup.py",  # Python setup file
        "requirements.txt",  # Python requirements
        "Dockerfile",  # Docker project
        "README.md",  # Common project file
        ".gitignore",  # Git ignore file
    ]

    for pattern in patterns:
        root = find_root_by_pattern(pattern, start_dir)
        if root is not None:
            return root

    # If all else fails, return the directory containing the file
    return start_dir


# ------------------------------
# Similarity utilities
# ------------------------------


def _tokenize(s: str):
    r"""
    Tokenize a string into lowercase word tokens (alphanumeric + underscore).

    Uses regex "\w+" to be robust to punctuation/spacing differences. # noqa: W605
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

    # Ensure the first dimension is the shorter string to use less memory
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
