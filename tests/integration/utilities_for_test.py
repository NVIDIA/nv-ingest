# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
