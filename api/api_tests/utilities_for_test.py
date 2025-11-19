# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess


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
