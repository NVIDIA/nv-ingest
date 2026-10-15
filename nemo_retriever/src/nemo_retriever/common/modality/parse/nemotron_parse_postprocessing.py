# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Postprocessing utilities for Nemotron Parse v1.2 model output.

The functions in this file were copied from the model repository:
  https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2/blob/main/postprocessing.py
  https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2/blob/main/latex2html.py
"""

from __future__ import annotations

import re
from typing import List, Tuple

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# latex2html – LaTeX tabular → HTML → Markdown conversion
# ---------------------------------------------------------------------------


def _skip_whitespace(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _parse_braced_argument(text: str, i: int) -> Tuple[str, int]:
    if i >= len(text) or text[i] != "{":
        raise ValueError(f"Expected '{{' at position {i}")
    i += 1
    start = i
    level = 1
    while i < len(text) and level > 0:
        if text[i] == "{":
            level += 1
        elif text[i] == "}":
            level -= 1
        i += 1
    if level != 0:
        raise ValueError(f"Unbalanced braces starting at position {start - 1}")
    return text[start : i - 1], i


def _parse_command(text: str, i: int) -> Tuple[str, int]:
    if text.startswith(r"\multirow", i):
        command_name = r"\multirow"
        i += len(r"\multirow")
    elif text.startswith(r"\multicolumn", i):
        command_name = r"\multicolumn"
        i += len(r"\multicolumn")
    else:
        raise ValueError(f"Expected \\multirow or \\multicolumn at position {i}")

    i = _skip_whitespace(text, i)
    args: list[str] = []
    for arg_index in range(3):
        if i >= len(text) or text[i] != "{":
            raise ValueError(f"Expected '{{' for argument {arg_index + 1} at position {i}")
        arg_content, i = _parse_braced_argument(text, i)
        processed_arg = _clean_multi_cells(arg_content)
        if arg_index == 2:
            processed_arg = re.sub(r"(?<!\\)&", r"\\&", processed_arg)
        args.append(processed_arg)
        if arg_index < 2:
            i = _skip_whitespace(text, i)
    return f"{command_name}{{{args[0]}}}{{{args[1]}}}{{{args[2]}}}", i


def _clean_multi_cells(text: str) -> str:
    result: list[str] = []
    i = 0
    while i < len(text):
        idx_multi = text.find(r"\multirow", i)
        idx_multiC = text.find(r"\multicolumn", i)
        if idx_multi == -1 and idx_multiC == -1:
            result.append(text[i:])
            break
        if idx_multi == -1:
            next_idx = idx_multiC
        elif idx_multiC == -1:
            next_idx = idx_multi
        else:
            next_idx = min(idx_multi, idx_multiC)
        result.append(text[i:next_idx])
        command_text, new_index = _parse_command(text, next_idx)
        result.append(command_text)
        i = new_index
    return "".join(result)


def _parse_brace(s: str, pos: int) -> Tuple[str, int]:
    if pos >= len(s) or s[pos] != "{":
        raise ValueError(f"Expected '{{' at position {pos}")
    pos += 1
    content = ""
    depth = 1
    while pos < len(s) and depth:
        char = s[pos]
        if char == "{":
            depth += 1
            content += char
        elif char == "}":
            depth -= 1
            if depth:
                content += char
        else:
            content += char
        pos += 1
    if depth != 0:
        raise ValueError("Unmatched '{' in string.")
    return content, pos


def _parse_command_merge(s: str, pos: int):
    if s.startswith(r"\multirow", pos):
        newpos = pos + len(r"\multirow")
        rowspan, newpos = _parse_brace(s, newpos)
        width, newpos = _parse_brace(s, newpos)
        content, newpos = _parse_brace(s, newpos)
        index_mr = content.find(r"\multirow")
        index_mc = content.find(r"\multicolumn")
        if index_mr == -1 and index_mc == -1:
            return {"rowspan": rowspan.strip(), "width": width.strip(), "content": content.strip()}, newpos
        indices = [i for i in (index_mr, index_mc) if i != -1]
        first_index = min(indices)
        inner, _ = _parse_command_merge(content, first_index)
        merged = {"rowspan": rowspan.strip(), "width": width.strip()}
        merged.update(inner)
        return merged, newpos

    elif s.startswith(r"\multicolumn", pos):
        newpos = pos + len(r"\multicolumn")
        colspan, newpos = _parse_brace(s, newpos)
        alignment, newpos = _parse_brace(s, newpos)
        content, newpos = _parse_brace(s, newpos)
        index_mr = content.find(r"\multirow")
        index_mc = content.find(r"\multicolumn")
        if index_mr == -1 and index_mc == -1:
            return {"colspan": colspan.strip(), "alignment": alignment.strip(), "content": content.strip()}, newpos
        indices = [i for i in (index_mr, index_mc) if i != -1]
        first_index = min(indices)
        inner, _ = _parse_command_merge(content, first_index)
        merged = {"colspan": colspan.strip(), "alignment": alignment.strip()}
        merged.update(inner)
        return merged, newpos

    return None, pos


def _extract_merged_commands(s: str):
    pos = 0
    results = []
    while pos < len(s):
        if s[pos] == "\\":
            res, newpos = _parse_command_merge(s, pos)
            if res is not None:
                results.append(res)
                pos = newpos
                continue
        pos += 1
    return results


def _replace_italic(text: str) -> str:
    # "__" is two markers, not an empty italic span.
    pattern = re.compile(r"(?<![\\_])_(?!_)(.+?)(?<!\\)_")

    def italic_replacer(match):
        content = match.group(1).replace(r"\_", "_")
        return f"<i>{content}</i>"

    return pattern.sub(italic_replacer, text)


def _replace_bold(text: str) -> str:
    # "****" is not an empty bold span.
    pattern = re.compile(r"(?<![\\*])\*\*(?!\*)(.+?)(?<!\\)\*\*")

    def bold_replacer(match):
        content = match.group(1).replace(r"\*", "*")
        return f"<b>{content}</b>"

    return pattern.sub(bold_replacer, text)


def _latex_table_to_html(latex_str: str, add_head_body: bool = False) -> str:
    table_pattern = r"\\begin{tabular}{([^}]*)}\s*(.*?)\\end{tabular}"

    def process_cell(cell):
        cell = cell.strip()
        out = _extract_merged_commands(cell)
        if len(out) > 0:
            cell = process_cell(out[0]["content"])["content"]
            rowspan = int(out[0].get("rowspan", "1"))
            colspan = int(out[0].get("colspan", "1"))
            return {"content": cell, "colspan": colspan, "rowspan": rowspan}

        cell = re.sub(r"\$([^$]*)\$", r"\1", cell)
        cell = re.sub(r"\\textbf{([^}]*)}", r"<b>\1</b>", cell)
        cell = re.sub(r"\\textit{([^}]*)}", r"<i>\1</i>", cell)
        cell = _replace_italic(cell)
        cell = _replace_bold(cell)
        cell = (
            cell.replace("\\$", "$")
            .replace("\\%", "%")
            .replace("\\newline", "\n")
            .replace("\\textless", "<")
            .replace("\\textgreater", ">")
            .replace("\\*", "*")
            .replace("\\_", "_")
            .replace("\\backslash", "\\")
        )
        cell = cell.replace(r"\&", "&")
        cell = cell.replace("<tbc>", "")
        cell = cell.replace("\\unknown", "").replace("\\<|unk|\\>", "")
        cell = cell.replace("<u>", "<underline>").replace("</u>", "</underline>")
        return {"content": cell, "colspan": 1, "rowspan": 1}

    def split_row(input_string):
        return re.split(r"(?<!\\)&", input_string)

    def convert_table(match):
        _format_spec, content = match.groups()
        html = ["<table>"]
        multirow_tracker: set = set()
        current_row = 0
        rows = re.split(r"\\\\", content)
        for row in rows:
            if not row.strip():
                continue
            row = row.strip()
            if "\\hline" in row:
                row = row.replace("\\hline", "")
                if not row.strip():
                    continue
            row = _clean_multi_cells(row)
            cells = split_row(row)
            processed_cells = [process_cell(cell) for cell in cells]

            def split_lines(text):
                parts = re.split(r"(?:\n|<br\s*/?>)+", text)
                return parts if parts is not None else [""]

            line_lists = [split_lines(cell["content"]) for cell in processed_cells]
            max_lines = max(len(lst) for lst in line_lists) if line_lists else 1

            for line_idx in range(max_lines):
                if add_head_body:
                    if current_row == 0:
                        html.append(" <thead>")
                    if current_row == 1:
                        html.append(" <tbody>")
                html.append("  <tr>")
                current_col = 0
                for col_idx, cell in enumerate(processed_cells):
                    content_segment = line_lists[col_idx][line_idx] if line_idx < len(line_lists[col_idx]) else ""
                    attrs = []
                    if cell["colspan"] > 1:
                        attrs.append(f'colspan="{cell["colspan"]}"')
                    if cell["rowspan"] > 1 and line_idx == 0:
                        attrs.append(f'rowspan="{cell["rowspan"]}"')
                        for r in range(current_row + 1, current_row + cell["rowspan"]):
                            for c in range(current_col, current_col + cell["colspan"]):
                                multirow_tracker.add((r, c))
                    if cell["rowspan"] > 1 and line_idx > 0:
                        current_col += cell["colspan"]
                        continue
                    if (
                        (current_row, current_col) in multirow_tracker
                        and content_segment == ""
                        and cell["colspan"] == 1
                        and cell["rowspan"] == 1
                    ):
                        current_col += cell["colspan"]
                        continue
                    attr_str = " " + " ".join(attrs) if attrs else ""
                    html.append(f"    <td{attr_str}>{content_segment}</td>")
                    current_col += cell["colspan"]
                if add_head_body and current_row == 0:
                    html.append(" </thead>")
                html.append("  </tr>")
                current_row += 1
        if add_head_body:
            html.append(" </tbody>")
        html.append("</table>")
        return "\n".join(html)

    return re.sub(table_pattern, convert_table, latex_str, flags=re.DOTALL)


def _convert_single_table_to_markdown(table) -> str:
    markdown_lines = []
    rows = table.find_all("tr")
    for i, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        row_data = []
        for cell in cells:
            cell_text = cell.get_text(separator=" ", strip=True)
            cell_text = cell_text.replace("|", "\\|")
            row_data.append(cell_text)
        markdown_lines.append("| " + " | ".join(row_data) + " |")
        if i == 0:
            separator = "| " + " | ".join(["---"] * len(cells)) + " |"
            markdown_lines.append(separator)
    return "\n".join(markdown_lines)


def _convert_html_tables_to_markdown(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return html_content
    for table in tables:
        markdown_table = _convert_single_table_to_markdown(table)
        replacement = soup.new_string("\n" + markdown_table + "\n")
        table.replace_with(replacement)
    return str(soup)


# ---------------------------------------------------------------------------
# postprocessing.py – high-level output parsing
# ---------------------------------------------------------------------------

_RE_EXTRACT_CLASS_BBOX = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)" r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.DOTALL,
)


def extract_classes_bboxes(
    text: str,
) -> Tuple[List[str], List[Tuple[float, float, float, float]], List[str]]:
    """Parse raw Nemotron Parse output into (classes, bboxes, texts)."""
    classes: List[str] = []
    bboxes: List[Tuple[float, float, float, float]] = []
    texts: List[str] = []
    for m in _RE_EXTRACT_CLASS_BBOX.finditer(text):
        x1, y1, inner_text, x2, y2, cls = m.groups()
        cls = "Formula" if cls == "Inline-formula" else cls
        classes.append(cls)
        bboxes.append((float(x1), float(y1), float(x2), float(y2)))
        texts.append(inner_text)
    return classes, bboxes, texts


def transform_bbox_to_original(
    bbox: Tuple[float, float, float, float],
    original_width: int,
    original_height: int,
    target_w: int = 1664,
    target_h: int = 2048,
) -> Tuple[float, float, float, float]:
    """Convert normalised model bbox coords back to original image pixels."""
    aspect_ratio = original_width / original_height
    new_height = original_height
    new_width = original_width

    if original_height > target_h:
        new_height = target_h
        new_width = int(new_height * aspect_ratio)
    if new_width > target_w:
        new_width = target_w
        new_height = int(new_width / aspect_ratio)

    resized_width = new_width
    resized_height = new_height

    pad_left = (target_w - resized_width) // 2
    pad_top = (target_h - resized_height) // 2

    left = ((bbox[0] * target_w) - pad_left) * original_width / resized_width
    right = ((bbox[2] * target_w) - pad_left) * original_width / resized_width
    top = ((bbox[1] * target_h) - pad_top) * original_height / resized_height
    bottom = ((bbox[3] * target_h) - pad_top) * original_height / resized_height

    return left, top, right, bottom


def _convert_mmd_to_plain_text(mmd_text: str) -> str:
    mmd_text = re.sub(r"<sup>(.*?)</sup>", r"^{\\1}", mmd_text, flags=re.DOTALL)
    mmd_text = re.sub(r"<sub>(.*?)</sub>", r"_{\\1}", mmd_text, flags=re.DOTALL)
    mmd_text = mmd_text.replace("<br>", "\n")
    mmd_text = re.sub(r"#+\s", "", mmd_text)
    mmd_text = re.sub(r"\*\*(.*?)\*\*", r"\1", mmd_text)
    mmd_text = re.sub(r"\*(.*?)\*", r"\1", mmd_text)
    mmd_text = re.sub(r"(?<!\w)_([^_]+)_", r"\1", mmd_text)
    return mmd_text.strip()


def postprocess_text(
    text: str,
    cls: str = "Text",
    text_format: str = "markdown",
    table_format: str = "markdown",
    blank_text_in_figures: bool = False,
) -> str:
    """Format a single parsed element's text based on its class.

    Parameters
    ----------
    text : str
        Raw text extracted from the model output.
    cls : str
        Element class (e.g. ``"Text"``, ``"Table"``, ``"Picture"``).
    text_format : str
        ``"markdown"`` or ``"plain"``.
    table_format : str
        ``"latex"``, ``"HTML"``, or ``"markdown"``.
    blank_text_in_figures : bool
        If ``True``, clear text for ``Picture`` elements.
    """
    if cls != "Table":
        if text_format == "plain":
            text = _convert_mmd_to_plain_text(text)
    elif table_format == "HTML":
        text = _latex_table_to_html(text)
    elif table_format == "markdown":
        text = _convert_html_tables_to_markdown(_latex_table_to_html(text))

    if blank_text_in_figures and cls == "Picture":
        text = ""

    return text


def remove_nemotron_formatting(text: str) -> str:
    """Strip Nemotron-specific formatting tokens."""
    text = text.replace("<tbc>", "")
    text = text.replace("\\<|unk|\\>", "")
    text = text.replace("\\unknown", "")
    return text


# ---------------------------------------------------------------------------
# Convenience: full postprocessing pipeline
# ---------------------------------------------------------------------------


def postprocess_output(
    raw_text: str,
    *,
    text_format: str = "markdown",
    table_format: str = "markdown",
    blank_text_in_figures: bool = False,
) -> str:
    """Run full postprocessing on raw Nemotron Parse output.

    Parses the structured output, applies per-class formatting, and returns
    the concatenated text of all detected elements separated by double
    newlines.
    """
    classes, _bboxes, texts = extract_classes_bboxes(raw_text)
    if not texts:
        return raw_text.strip()
    processed = [
        postprocess_text(
            t,
            cls=c,
            text_format=text_format,
            table_format=table_format,
            blank_text_in_figures=blank_text_in_figures,
        )
        for t, c in zip(texts, classes)
    ]
    return "\n\n".join(t for t in processed if t)
