# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI for .txt extraction: tokenizer-based split, write *.txt_extraction.json.

Use with: retriever local stage5 run --pattern "*.txt_extraction.json" then stage6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional  # noqa: F401

import typer
from rich.console import Console
from retriever.params import TextChunkParams

from . import txt_file_to_chunks_df

console = Console()
app = typer.Typer(help="Txt extraction: tokenizer-based split, write primitives-like JSON.")


def _to_jsonable(val: Any) -> Any:
    """Convert to JSON-serializable (e.g. numpy int64 -> int)."""
    if hasattr(val, "item"):
        return val.item()
    if hasattr(val, "tolist"):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _to_jsonable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_jsonable(x) for x in val]
    return val


@app.command("run")
def run(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan for *.txt files.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory to write *.txt_extraction.json. Default: alongside each input file.",
    ),
    max_tokens: int = typer.Option(512, "--max-tokens", min=1, help="Max tokens per chunk."),
    overlap: int = typer.Option(0, "--overlap", min=0, help="Overlap tokens between consecutive chunks."),
    encoding: str = typer.Option("utf-8", "--encoding", help="File encoding for reading .txt."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Limit number of .txt files processed."),
) -> None:
    """
    Scan input_dir for *.txt, tokenizer-split each into chunks, write <stem>.txt_extraction.json.

    Output JSON has the same primitives-like shape as stage5 input (text, path, page_number, metadata).
    Then run: retriever local stage5 run --input-dir <dir> --pattern "*.txt_extraction.json"
    and retriever local stage6 run --input-dir <dir>.
    """
    input_dir = Path(input_dir)
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        console.print(f"[red]No *.txt files found in[/red] {input_dir}")
        raise typer.Exit(code=2)
    if limit is not None:
        txt_files = txt_files[:limit]

    for path in txt_files:
        try:
            df = txt_file_to_chunks_df(
                str(path),
                params=TextChunkParams(
                    max_tokens=max_tokens,
                    overlap_tokens=overlap,
                    encoding=encoding,
                ),
            )
        except Exception as e:
            console.print(f"[red]Failed[/red] {path}: {e}")
            continue

        out_path: Path
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = Path(output_dir) / f"{path.stem}.txt_extraction.json"
        else:
            out_path = path.with_suffix(".txt_extraction.json")

        records = df.to_dict(orient="records")
        jsonable = _to_jsonable(records)
        # Write as JSON array so stage5's pd.read_json(in_path) returns a DataFrame
        out_path.write_text(json.dumps(jsonable, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {out_path} ({len(df)} chunks)")

    console.print(f"[green]Done[/green] processed={len(txt_files)}")


def main() -> None:
    app()
