#!/usr/bin/env python3
"""
Filter bo767_query_gt.csv down to rows whose `pdf` id exists
as a prefix in any `*text_embeddings.json` file under an input directory.

Example embedding filename:
  1179117.pdf.pdf_extraction.infographic.table.text_embeddings.json
Extracted id:
  1179117
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable, Set


def iter_text_embedding_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith("text_embeddings.json"):
                yield Path(dirpath) / name


def extract_pdf_id_from_embedding_filename(path: Path) -> str | None:
    """
    Returns the prefix before the first '.' in the file name if it looks like a pdf id.
    """
    first = path.name.split(".", 1)[0]
    if not first:
        return None
    if not first.isdigit():
        return None
    return first


def collect_pdf_ids(embeddings_root: Path) -> Set[str]:
    ids: Set[str] = set()
    for p in iter_text_embedding_files(embeddings_root):
        pdf_id = extract_pdf_id_from_embedding_filename(p)
        if pdf_id is not None:
            ids.add(pdf_id)
    return ids


def filter_csv_by_pdf_ids(input_csv: Path, output_csv: Path, allowed_pdf_ids: Set[str]) -> tuple[int, int]:
    """
    Returns (kept_rows, total_rows) excluding the header.
    """
    kept = 0
    total = 0

    with input_csv.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {input_csv}")
        if "pdf" not in reader.fieldnames:
            raise ValueError(f"Expected a 'pdf' column in {input_csv}, got columns: {reader.fieldnames}")

        with output_csv.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                total += 1
                pdf_val = (row.get("pdf") or "").strip()
                # normalize '001234' vs '1234' cases
                pdf_norm = pdf_val.lstrip("0") or "0" if pdf_val.isdigit() else pdf_val

                if pdf_norm in allowed_pdf_ids or pdf_val in allowed_pdf_ids:
                    writer.writerow(row)
                    kept += 1

    return kept, total


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Filter bo767_query_gt.csv to PDFs present in *text_embeddings.json filenames."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory to search (recursively) for *text_embeddings.json",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("bo767_query_gt.csv"),
        help="Input CSV path (default: bo767_query_gt.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("jp20_query_gt.csv"),
        help="Output CSV path (default: jp20_query_gt.csv)",
    )
    args = parser.parse_args(argv)

    if not args.input_dir.exists():
        print(f"ERROR: input_dir does not exist: {args.input_dir}", file=sys.stderr)
        return 2
    if not args.input_dir.is_dir():
        print(f"ERROR: input_dir is not a directory: {args.input_dir}", file=sys.stderr)
        return 2
    if not args.input_csv.exists():
        print(f"ERROR: input_csv does not exist: {args.input_csv}", file=sys.stderr)
        return 2

    allowed = collect_pdf_ids(args.input_dir)
    if not allowed:
        print(
            f"WARNING: Found 0 pdf ids from *text_embeddings.json under {args.input_dir}. "
            f"Output will contain only the header.",
            file=sys.stderr,
        )

    kept, total = filter_csv_by_pdf_ids(args.input_csv, args.output_csv, allowed)
    print(f"Found {len(allowed)} pdf ids under {args.input_dir}")
    print(f"Filtered {args.input_csv}: kept {kept}/{total} rows")
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

