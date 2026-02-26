# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


app = typer.Typer(
    no_args_is_help=True,
    help="Compare two JSON files and render a side-by-side diff.",
)


@dataclass(frozen=True)
class _DiffRow:
    marker: str  # " " | "~" | "-" | "+"
    a_idx: int | None
    b_idx: int | None
    a_line: str
    b_line: str


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        # Surface as a CLI error rather than a stack trace.
        raise typer.BadParameter(f"Invalid JSON in '{path}': {e.msg} (line {e.lineno}, column {e.colno})") from e


def _to_pretty_lines(obj: Any, *, sort_keys: bool) -> list[str]:
    # Stable formatting is important for meaningful diffs.
    rendered = json.dumps(obj, indent=2, sort_keys=sort_keys, ensure_ascii=False)
    return rendered.splitlines()


def _strip_ignored_fields(obj: Any, *, ignore_fields: set[str], ignore_case: bool) -> Any:
    """
    Remove dict keys matching `ignore_fields` at any depth.

    This is a simple (field-name based) filter intended to hide expected-to-differ
    metadata like timestamps, runtimes, UUIDs, etc.
    """
    if not ignore_fields:
        return obj

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            k_norm = k.lower() if ignore_case and isinstance(k, str) else k
            if isinstance(k_norm, str) and k_norm in ignore_fields:
                continue
            out[k] = _strip_ignored_fields(v, ignore_fields=ignore_fields, ignore_case=ignore_case)
        return out

    if isinstance(obj, list):
        return [_strip_ignored_fields(v, ignore_fields=ignore_fields, ignore_case=ignore_case) for v in obj]

    return obj


def _iter_rows(a_lines: list[str], b_lines: list[str]) -> tuple[list[_DiffRow], dict[str, int]]:
    sm = SequenceMatcher(a=a_lines, b=b_lines, autojunk=False)
    rows: list[_DiffRow] = []
    stats = {"equal": 0, "replace": 0, "delete": 0, "insert": 0}

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        stats[tag] = stats.get(tag, 0) + (
            max(i2 - i1, j2 - j1) if tag == "replace" else (i2 - i1) if tag in {"equal", "delete"} else (j2 - j1)
        )

        if tag == "equal":
            for k in range(i2 - i1):
                ai = i1 + k
                bj = j1 + k
                rows.append(_DiffRow(" ", ai, bj, a_lines[ai], b_lines[bj]))
        elif tag == "replace":
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                ai = i1 + k
                bj = j1 + k
                a_idx = ai if ai < i2 else None
                b_idx = bj if bj < j2 else None
                rows.append(
                    _DiffRow(
                        "~",
                        a_idx,
                        b_idx,
                        a_lines[a_idx] if a_idx is not None else "",
                        b_lines[b_idx] if b_idx is not None else "",
                    )
                )
        elif tag == "delete":
            for ai in range(i1, i2):
                rows.append(_DiffRow("-", ai, None, a_lines[ai], ""))
        elif tag == "insert":
            for bj in range(j1, j2):
                rows.append(_DiffRow("+", None, bj, "", b_lines[bj]))
        else:  # pragma: no cover
            raise RuntimeError(f"Unexpected diff opcode tag: {tag}")

    return rows, stats


def _collapse_equal_rows(rows: list[_DiffRow], *, context: int) -> list[_DiffRow | None]:
    """
    Return a list of rows where long equal runs are replaced with `None` sentinels.
    The rendering step will convert `None` gaps into a single "unchanged" row.
    """
    if context < 0:
        context = 0

    diff_idxs = [i for i, r in enumerate(rows) if r.marker != " "]
    if not diff_idxs:
        return [*rows]

    keep = [False] * len(rows)
    for i in diff_idxs:
        start = max(0, i - context)
        end = min(len(rows), i + context + 1)
        for j in range(start, end):
            keep[j] = True

    collapsed: list[_DiffRow | None] = []
    in_gap = False
    for i, r in enumerate(rows):
        if keep[i]:
            collapsed.append(r)
            in_gap = False
        else:
            if not in_gap:
                collapsed.append(None)
                in_gap = True
    return collapsed


def _styled_line(idx: int | None, line: str, *, style: str) -> Text:
    if idx is None:
        prefix = "     │ "
    else:
        prefix = f"{idx + 1:>5}│ "
    t = Text(prefix + line)
    if style:
        t.stylize(style)
    return t


def _marker_style(marker: str) -> tuple[str, str]:
    if marker == " ":
        return " ", "dim"
    if marker == "~":
        return "~", "yellow"
    if marker == "-":
        return "-", "red"
    if marker == "+":
        return "+", "green"
    return marker, ""


def _render_side_by_side(
    *,
    a_path: Path,
    b_path: Path,
    rows: list[_DiffRow],
    context: int,
    show_all: bool,
    max_rows: int,
    no_color: bool,
) -> None:
    console = Console(color_system=None if no_color else "auto")

    if show_all:
        display_rows: list[_DiffRow | None] = [*rows]
    else:
        display_rows = _collapse_equal_rows(rows, context=context)

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
        pad_edge=False,
        collapse_padding=True,
    )
    table.add_column("", no_wrap=True, width=1)
    table.add_column(f"A: {a_path}", overflow="fold", ratio=1)
    table.add_column(f"B: {b_path}", overflow="fold", ratio=1)

    rendered = 0
    gap_size = 0

    def flush_gap() -> None:
        nonlocal gap_size, rendered
        if gap_size <= 0:
            return
        msg = Text(f"... {gap_size} unchanged line(s) ...", style="dim")
        table.add_row(Text("…", style="dim"), msg, msg)
        rendered += 1
        gap_size = 0

    for r in display_rows:
        if max_rows > 0 and rendered >= max_rows:
            flush_gap()
            msg = Text(f"... truncated after {max_rows} row(s) ...", style="bold red")
            table.add_row(Text("!", style="bold red"), msg, msg)
            break

        if r is None:
            gap_size += 1
            continue

        flush_gap()
        m, m_style = _marker_style(r.marker)
        a_style = "dim" if r.marker == " " else ("yellow" if r.marker == "~" else ("red" if r.marker == "-" else ""))
        b_style = (
            "dim"
            if r.marker == " "
            else ("yellow" if r.marker == "~" else ("" if r.marker == "-" else "green" if r.marker == "+" else ""))
        )

        table.add_row(
            Text(m, style=m_style),
            _styled_line(r.a_idx, r.a_line, style=a_style),
            _styled_line(r.b_idx, r.b_line, style=b_style),
        )
        rendered += 1

    flush_gap()
    console.print(table)


@app.callback(invoke_without_command=True)
def compare_json(
    ctx: typer.Context,
    a_json: Path = typer.Argument(..., exists=True, dir_okay=False, file_okay=True, readable=True, help="JSON file A"),
    b_json: Path = typer.Argument(..., exists=True, dir_okay=False, file_okay=True, readable=True, help="JSON file B"),
    context: int = typer.Option(
        3,
        "--context",
        "-C",
        help="Number of unchanged lines of context to show around differences (ignored with --show-all).",
    ),
    show_all: bool = typer.Option(False, "--show-all", help="Show all lines (not just differences with context)."),
    sort_keys: bool = typer.Option(
        True,
        "--sort-keys/--no-sort-keys",
        help="Sort object keys before diffing for stable output.",
    ),
    ignore_field: list[str] = typer.Option(
        [],
        "--ignore-field",
        "-i",
        help=(
            "Field name to ignore during comparison. Can be passed multiple times. "
            "Example: -i runtime_seconds -i uuid -i date"
        ),
    ),
    ignore_field_case_insensitive: bool = typer.Option(
        True,
        "--ignore-field-case-insensitive/--ignore-field-case-sensitive",
        help="Whether --ignore-field matching is case-insensitive.",
    ),
    max_rows: int = typer.Option(
        2000,
        "--max-rows",
        help="Maximum number of table rows to render (0 = unlimited).",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
) -> None:
    """
    Render a side-by-side diff of two JSON files.

    Exit codes:
      - 0: no differences
      - 1: differences found
      - 2: invalid inputs (e.g., JSON parse error)
    """
    _ = ctx  # typer passes context; we don't use it yet.

    a_obj = _load_json(a_json)
    b_obj = _load_json(b_json)

    ignore_fields_set = {s.lower() if ignore_field_case_insensitive else s for s in ignore_field if s}
    if ignore_fields_set:
        a_obj = _strip_ignored_fields(a_obj, ignore_fields=ignore_fields_set, ignore_case=ignore_field_case_insensitive)
        b_obj = _strip_ignored_fields(b_obj, ignore_fields=ignore_fields_set, ignore_case=ignore_field_case_insensitive)

    a_lines = _to_pretty_lines(a_obj, sort_keys=sort_keys)
    b_lines = _to_pretty_lines(b_obj, sort_keys=sort_keys)

    rows, _stats = _iter_rows(a_lines, b_lines)
    has_diff = any(r.marker != " " for r in rows)

    if not has_diff:
        Console(color_system=None if no_color else "auto").print(
            Text("No differences.", style="bold green" if not no_color else "")
        )
        raise typer.Exit(code=0)

    _render_side_by_side(
        a_path=a_json,
        b_path=b_json,
        rows=rows,
        context=context,
        show_all=show_all,
        max_rows=max_rows,
        no_color=no_color,
    )
    raise typer.Exit(code=1)
