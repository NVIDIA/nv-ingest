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
             or "./" if not found.
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
