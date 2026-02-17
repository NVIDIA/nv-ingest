from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text


app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Compare nv-ingest-main outputs vs inprocess outputs for the same PDFs.\n\n"
        "This command is designed for schema drift: it normalizes both JSON formats into a small"
        " common shape, then prints a per-PDF breakdown of differences so you can quickly find"
        " where rewritten code diverges."
    ),
)


# --------------------------------------------------------------------------------------
# Mapping / normalization configuration
#
# This is intentionally simple and meant to be edited as schemas evolve.
# --------------------------------------------------------------------------------------

# How to align "structured subtype" (nv-ingest-main) to label names (inprocess).
# If your inprocess output uses different names, update this dict.
SUBTYPE_TO_INPROCESS_LABEL: dict[str, str] = {
    "table": "table",
    "chart": "chart",
    "infographic": "infographic",
}

# Some inprocess outputs might use alternate label strings; normalize them here.
INPROCESS_LABEL_NORMALIZATION: dict[str, str] = {
    "figure": "chart",
    "graphic": "chart",
}


# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _MainStructuredItem:
    page_number: int | None
    subtype: str
    bbox_xyxy: list[float] | None
    table_markdown: str | None


@dataclass(frozen=True)
class _NormalizedMain:
    pdf_key: str
    page_count: int | None
    document_text: str
    structured_items: list[_MainStructuredItem]
    structured_counts_by_page: dict[int, Counter[str]]
    structured_counts_total: Counter[str]


@dataclass(frozen=True)
class _NormalizedInprocess:
    pdf_key: str
    page_count: int
    document_text: str
    page_element_counts_by_page: dict[int, Counter[str]]
    page_element_counts_total: Counter[str]
    table_region_counts_by_page: dict[int, int]
    table_region_counts_total: int


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(
            f"Invalid JSON in '{path}': {e.msg} (line {e.lineno}, column {e.colno})"
        ) from e


def _main_pdf_key(path: Path) -> str:
    """
    nv-ingest-main file naming looks like: 1015168.pdf.metadata.json
    We normalize the key to the PDF filename: 1015168.pdf
    """
    name = path.name
    if name.endswith(".metadata.json"):
        name = name[: -len(".metadata.json")]
    return name


def _simple_pdf_key_variants(pdf_key: str) -> set[str]:
    """
    Convenience for matching user-provided --only values.
    """
    v = {pdf_key}
    if pdf_key.endswith(".pdf"):
        v.add(pdf_key[: -len(".pdf")])
    return v


def _normalize_inprocess_label(label: str) -> str:
    if not label:
        return label
    label_norm = label.strip().lower()
    return INPROCESS_LABEL_NORMALIZATION.get(label_norm, label_norm)


def _counter_from_counts_by_label(value: Any) -> Counter[str]:
    if not isinstance(value, dict):
        return Counter()
    out: Counter[str] = Counter()
    for k, v in value.items():
        if not isinstance(k, str):
            continue
        try:
            n = int(v)
        except Exception:
            continue
        out[_normalize_inprocess_label(k)] += n
    return out


def _counter_from_page_elements_v3(value: Any) -> Counter[str]:
    """
    Fallback when *_counts_by_label is missing/empty.
    Expected shape:
      { "detections": [ {"label_name": "text", ...}, ... ] }
    """
    if not isinstance(value, dict):
        return Counter()
    dets = value.get("detections")
    if not isinstance(dets, list):
        return Counter()
    c: Counter[str] = Counter()
    for d in dets:
        if not isinstance(d, dict):
            continue
        ln = d.get("label_name")
        if isinstance(ln, str) and ln:
            c[_normalize_inprocess_label(ln)] += 1
    return c


def _count_table_regions(table_data: Any) -> int:
    """
    Count table regions from either the ``table`` column format (list of
    OCR entries) or the legacy ``table_structure_v1`` format (dict with
    ``regions``).

    ``table`` format::

        [ {"bbox_xyxy_norm": [...], "text": "..."}, ... ]

    Legacy format (``table_structure_v1``)::

        { "regions": [ {"label_name": "table", ...}, ... ] }
    """
    # New format: list of OCR entries — each entry is one table region.
    if isinstance(table_data, list):
        return len(table_data)

    # Legacy format: dict with "regions" list.
    if not isinstance(table_data, dict):
        return 0
    regions = table_data.get("regions")
    if not isinstance(regions, list):
        return 0
    n = 0
    for r in regions:
        if not isinstance(r, dict):
            continue
        if _normalize_inprocess_label(str(r.get("label_name") or "")) == "table":
            n += 1
    return n


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _normalize_main(
    *,
    pdf_key: str,
    structured_path: Path | None,
    text_path: Path | None,
) -> _NormalizedMain:
    structured_items: list[_MainStructuredItem] = []
    structured_counts_by_page: dict[int, Counter[str]] = defaultdict(Counter)
    structured_counts_total: Counter[str] = Counter()

    page_count: int | None = None

    if structured_path is not None and structured_path.exists():
        structured_obj = _load_json(structured_path)
        if not isinstance(structured_obj, list):
            raise typer.BadParameter(f"Expected list JSON in '{structured_path}', got {type(structured_obj).__name__}")

        for item in structured_obj:
            if not isinstance(item, dict):
                continue
            md = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            cm = md.get("content_metadata") if isinstance(md.get("content_metadata"), dict) else {}
            hierarchy = cm.get("hierarchy") if isinstance(cm.get("hierarchy"), dict) else {}

            pc = _safe_int(hierarchy.get("page_count"))
            if pc is not None:
                page_count = pc

            subtype = cm.get("subtype") if isinstance(cm.get("subtype"), str) else ""
            subtype_norm = subtype.strip().lower() if subtype else ""

            pn = _safe_int(cm.get("page_number"))
            if pn is None:
                pn = _safe_int(hierarchy.get("page"))

            tm = md.get("table_metadata") if isinstance(md.get("table_metadata"), dict) else {}
            bbox = tm.get("table_location")
            bbox_xyxy: list[float] | None = None
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    bbox_xyxy = [float(x) for x in bbox]
                except Exception:
                    bbox_xyxy = None

            table_md = tm.get("table_content")
            if not isinstance(table_md, str) or not table_md.strip():
                table_md = None

            structured_items.append(
                _MainStructuredItem(page_number=pn, subtype=subtype_norm, bbox_xyxy=bbox_xyxy, table_markdown=table_md)
            )

            if subtype_norm:
                structured_counts_total[subtype_norm] += 1
                if pn is not None and pn >= 1:
                    structured_counts_by_page[pn][subtype_norm] += 1

    # Document-level text: nv-ingest-main text output is typically a list of document chunks.
    document_text_parts: list[str] = []
    if text_path is not None and text_path.exists():
        text_obj = _load_json(text_path)
        if not isinstance(text_obj, list):
            raise typer.BadParameter(f"Expected list JSON in '{text_path}', got {type(text_obj).__name__}")

        for item in text_obj:
            if not isinstance(item, dict):
                continue
            md = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            cm = md.get("content_metadata") if isinstance(md.get("content_metadata"), dict) else {}
            hierarchy = cm.get("hierarchy") if isinstance(cm.get("hierarchy"), dict) else {}

            pc = _safe_int(hierarchy.get("page_count"))
            if pc is not None:
                page_count = pc

            content = md.get("content")
            if isinstance(content, str) and content.strip():
                document_text_parts.append(content)

    document_text = "\n\n".join(document_text_parts).strip()
    return _NormalizedMain(
        pdf_key=pdf_key,
        page_count=page_count,
        document_text=document_text,
        structured_items=structured_items,
        structured_counts_by_page=dict(structured_counts_by_page),
        structured_counts_total=structured_counts_total,
    )


def _normalize_inprocess(*, path: Path) -> _NormalizedInprocess:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        raise typer.BadParameter(f"Expected dict JSON in '{path}', got {type(obj).__name__}")

    source_path = obj.get("source_path")
    if isinstance(source_path, str) and source_path:
        pdf_key = Path(source_path).name
    else:
        # Fallback: 1015168.20260210T....json -> 1015168.pdf
        prefix = path.name.split(".", 1)[0]
        pdf_key = prefix if prefix.endswith(".pdf") else f"{prefix}.pdf"

    records = obj.get("records")
    if not isinstance(records, list):
        raise typer.BadParameter(f"Expected 'records' list in '{path}'")

    page_element_counts_by_page: dict[int, Counter[str]] = {}
    table_region_counts_by_page: dict[int, int] = {}
    text_by_page: dict[int, str] = {}

    for rec in records:
        if not isinstance(rec, dict):
            continue
        pn = _safe_int(rec.get("page_number"))
        if pn is None:
            continue

        # Text
        t = rec.get("text")
        if isinstance(t, str) and t.strip():
            text_by_page[pn] = t

        # Page elements counts
        counts = _counter_from_counts_by_label(rec.get("page_elements_v3_counts_by_label"))
        if not counts:
            counts = _counter_from_page_elements_v3(rec.get("page_elements_v3"))
        if counts:
            page_element_counts_by_page[pn] = counts

        # Table regions (independent of "page elements" model)
        table_region_counts_by_page[pn] = _count_table_regions(rec.get("table") or rec.get("table_structure_v1"))

    # Document text: join in ascending page order (keeps comparison stable).
    document_text = "\n\n".join(text_by_page[p] for p in sorted(text_by_page.keys())).strip()

    page_element_counts_total: Counter[str] = Counter()
    for c in page_element_counts_by_page.values():
        page_element_counts_total.update(c)

    table_region_counts_total = sum(table_region_counts_by_page.values())

    return _NormalizedInprocess(
        pdf_key=pdf_key,
        page_count=len([r for r in records if isinstance(r, dict) and _safe_int(r.get("page_number")) is not None]),
        document_text=document_text,
        page_element_counts_by_page=page_element_counts_by_page,
        page_element_counts_total=page_element_counts_total,
        table_region_counts_by_page=table_region_counts_by_page,
        table_region_counts_total=table_region_counts_total,
    )


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def _fmt_ratio(r: float) -> str:
    return f"{r*100:.1f}%"


def _fmt_counter(c: Counter[str], keys: Iterable[str]) -> str:
    parts = []
    for k in keys:
        parts.append(f"{k}={c.get(k, 0)}")
    return ", ".join(parts)


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _counter_to_dict(c: Counter[str]) -> dict[str, int]:
    return {k: int(v) for k, v in sorted(c.items())}


def _slug_filename(s: str) -> str:
    """
    Keep filenames stable and shell-friendly.
    """
    s = s.strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in {".", "-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    # avoid empty
    return "".join(out) or "output"


def _compute_diff_json(
    *,
    main: _NormalizedMain | None,
    inp: _NormalizedInprocess | None,
    missing: list[str] | None,
    min_text_similarity: float,
) -> dict[str, Any]:
    """
    Returns a structured diff payload that can be written to JSON and later rendered.
    """
    if missing:
        return {
            "status": "missing_inputs",
            "pdf_key": (main.pdf_key if main else inp.pdf_key if inp else None),
            "missing": missing,
        }

    assert main is not None and inp is not None

    txt_sim = _similarity(main.document_text, inp.document_text)
    page_count_ok = (main.page_count is None) or (main.page_count == inp.page_count)
    text_ok = txt_sim >= min_text_similarity

    subtype_keys = sorted({*SUBTYPE_TO_INPROCESS_LABEL.keys()})

    # Compare structured subtype totals vs mapped inprocess label totals
    structured_mismatches: list[dict[str, Any]] = []
    for st in subtype_keys:
        label = _normalize_inprocess_label(SUBTYPE_TO_INPROCESS_LABEL.get(st, st))
        a_n = int(main.structured_counts_total.get(st, 0))
        b_n = int(inp.page_element_counts_total.get(label, 0))
        ok = a_n == b_n
        if not ok:
            structured_mismatches.append(
                {"subtype": st, "inprocess_label": label, "main": a_n, "inprocess": b_n}
            )

    # Per-page diffs (store all diffs; UI printing can truncate)
    pages = sorted(
        {
            *main.structured_counts_by_page.keys(),
            *inp.page_element_counts_by_page.keys(),
            *inp.table_region_counts_by_page.keys(),
        }
    )
    page_diffs: list[dict[str, Any]] = []
    for p in pages:
        main_c = main.structured_counts_by_page.get(p, Counter())
        inp_c = inp.page_element_counts_by_page.get(p, Counter())
        inp_tables = int(inp.table_region_counts_by_page.get(p, 0))

        a_tbl = int(main_c.get("table", 0))
        a_cht = int(main_c.get("chart", 0))
        a_inf = int(main_c.get("infographic", 0))

        b_tbl = int(inp_c.get(_normalize_inprocess_label(SUBTYPE_TO_INPROCESS_LABEL.get("table", "table")), 0))
        b_cht = int(inp_c.get(_normalize_inprocess_label(SUBTYPE_TO_INPROCESS_LABEL.get("chart", "chart")), 0))
        b_inf = int(
            inp_c.get(_normalize_inprocess_label(SUBTYPE_TO_INPROCESS_LABEL.get("infographic", "infographic")), 0)
        )

        ok = (a_tbl == b_tbl) and (a_cht == b_cht) and (a_inf == b_inf) and (a_tbl == inp_tables)
        if not ok:
            page_diffs.append(
                {
                    "page": int(p),
                    "main": {"table": a_tbl, "chart": a_cht, "infographic": a_inf},
                    "inprocess_page_elements_v3": {"table": b_tbl, "chart": b_cht, "infographic": b_inf},
                    "inprocess_table_regions": inp_tables,
                }
            )

    main_tables_total = int(main.structured_counts_total.get("table", 0))
    table_regions_ok = main_tables_total == int(inp.table_region_counts_total)

    return {
        "status": "ok",
        "pdf_key": main.pdf_key,
        "schema": {
            "subtype_to_inprocess_label": dict(SUBTYPE_TO_INPROCESS_LABEL),
            "inprocess_label_normalization": dict(INPROCESS_LABEL_NORMALIZATION),
        },
        "metrics": {
            "page_count_ok": page_count_ok,
            "main_page_count": main.page_count,
            "inprocess_page_count": inp.page_count,
            "text_similarity_ratio": float(txt_sim),
            "text_ok": text_ok,
            "min_text_similarity": float(min_text_similarity),
            "main_text_len": int(len(main.document_text)),
            "inprocess_text_len": int(len(inp.document_text)),
            "main_text_sha256": _sha256_text(main.document_text) if main.document_text else None,
            "inprocess_text_sha256": _sha256_text(inp.document_text) if inp.document_text else None,
            "table_regions_ok": table_regions_ok,
            "main_tables_total": main_tables_total,
            "inprocess_table_regions_total": int(inp.table_region_counts_total),
        },
        "totals": {
            "main_structured_counts_total": _counter_to_dict(main.structured_counts_total),
            "inprocess_page_elements_counts_total": _counter_to_dict(inp.page_element_counts_total),
        },
        "differences": {
            "structured_total_mismatches": structured_mismatches,
            "page_diffs": page_diffs,
        },
    }


def _render_pdf_summary(
    *,
    console: Console,
    main: _NormalizedMain,
    inp: _NormalizedInprocess,
    show_page_diffs: bool,
    max_page_diffs: int,
    min_text_similarity: float,
) -> bool:
    """
    Returns True if any differences were found.
    """
    has_diff = False

    title = Text(main.pdf_key, style="bold")
    console.print(title)

    # Summary table
    summary = Table(show_header=True, header_style="bold", pad_edge=False)
    summary.add_column("Field", style="bold")
    summary.add_column("nv-ingest-main")
    summary.add_column("inprocess")
    summary.add_column("Status", no_wrap=True)

    def add_row(field: str, a: str, b: str, ok: bool, note: str | None = None) -> None:
        nonlocal has_diff
        if not ok:
            has_diff = True
        status = Text("OK", style="green") if ok else Text("DIFF", style="red")
        if note:
            status.append(f" ({note})", style="dim")
        summary.add_row(field, a, b, status)

    # Page count
    main_pc = str(main.page_count) if main.page_count is not None else "?"
    inp_pc = str(inp.page_count)
    add_row("page_count", main_pc, inp_pc, (main.page_count is None) or (main.page_count == inp.page_count))

    # Text
    txt_sim = _similarity(main.document_text, inp.document_text)
    add_row(
        "document_text",
        f"len={len(main.document_text)}",
        f"len={len(inp.document_text)}",
        txt_sim >= min_text_similarity,
        note=f"sim={_fmt_ratio(txt_sim)}",
    )

    # Structured subtype totals vs inprocess page-elements label totals
    subtype_keys = sorted({*SUBTYPE_TO_INPROCESS_LABEL.keys()})
    main_counts = main.structured_counts_total
    inp_counts = inp.page_element_counts_total

    # Compare using mapping (subtype -> label)
    mismatch_notes: list[str] = []
    for st in subtype_keys:
        label = _normalize_inprocess_label(SUBTYPE_TO_INPROCESS_LABEL.get(st, st))
        a_n = main_counts.get(st, 0)
        b_n = inp_counts.get(label, 0)
        if a_n != b_n:
            mismatch_notes.append(f"{st}:{a_n}!={b_n}")
    add_row(
        "structured_counts",
        _fmt_counter(main_counts, subtype_keys),
        _fmt_counter(inp_counts, [_normalize_inprocess_label(SUBTYPE_TO_INPROCESS_LABEL[k]) for k in subtype_keys]),
        ok=(len(mismatch_notes) == 0),
        note=None
        if not mismatch_notes
        else "; ".join(mismatch_notes[:4]) + ("; ..." if len(mismatch_notes) > 4 else ""),
    )

    # Table-region totals (different model, but useful sanity check)
    main_tables = main.structured_counts_total.get("table", 0)
    add_row(
        "table_regions",
        str(main_tables),
        str(inp.table_region_counts_total),
        ok=(main_tables == inp.table_region_counts_total),
    )

    console.print(summary)

    # Per-page diffs
    if show_page_diffs:
        pages = sorted({*main.structured_counts_by_page.keys(), *inp.page_element_counts_by_page.keys(), *inp.table_region_counts_by_page.keys()})
        if pages:
            per_page = Table(show_header=True, header_style="bold", pad_edge=False)
            per_page.add_column("page", justify="right", no_wrap=True)
            per_page.add_column("main (tables/charts/infographics)")
            per_page.add_column("inprocess page_elements_v3 (table/chart/infographic)")
            per_page.add_column("inprocess table_regions(table)")
            per_page.add_column("Status", no_wrap=True)

            shown = 0
            for p in pages:
                main_c = main.structured_counts_by_page.get(p, Counter())
                inp_c = inp.page_element_counts_by_page.get(p, Counter())
                inp_tables = inp.table_region_counts_by_page.get(p, 0)

                # Align counts for display
                a_tbl = main_c.get("table", 0)
                a_cht = main_c.get("chart", 0)
                a_inf = main_c.get("infographic", 0)

                b_tbl = inp_c.get(_normalize_inprocess_label(SUBTYPE_TO_INPROCESS_LABEL.get("table", "table")), 0)
                b_cht = inp_c.get(_normalize_inprocess_label(SUBTYPE_TO_INPROCESS_LABEL.get("chart", "chart")), 0)
                b_inf = inp_c.get(_normalize_inprocess_label(SUBTYPE_TO_INPROCESS_LABEL.get("infographic", "infographic")), 0)

                ok = (a_tbl == b_tbl) and (a_cht == b_cht) and (a_inf == b_inf) and (a_tbl == inp_tables)
                if ok:
                    continue

                shown += 1
                if max_page_diffs > 0 and shown > max_page_diffs:
                    per_page.add_row("…", "…", "…", "…", Text("truncated", style="dim"))
                    break

                status = Text("DIFF", style="red")
                per_page.add_row(
                    str(p),
                    f"table={a_tbl}, chart={a_cht}, infographic={a_inf}",
                    f"table={b_tbl}, chart={b_cht}, infographic={b_inf}",
                    str(inp_tables),
                    status,
                )

            if shown > 0:
                console.print(per_page)

    console.print()  # spacer
    return has_diff


def _discover_main_files(nv_ingest_main_results: Path) -> dict[str, dict[str, Path]]:
    """
    Returns:
      { pdf_key: { 'structured': Path, 'text': Path } }  (either key may be missing)
    """
    out: dict[str, dict[str, Path]] = defaultdict(dict)
    structured_dir = nv_ingest_main_results / "structured"
    text_dir = nv_ingest_main_results / "text"

    if structured_dir.exists():
        for p in structured_dir.glob("*.json"):
            out[_main_pdf_key(p)]["structured"] = p
    if text_dir.exists():
        for p in text_dir.glob("*.json"):
            out[_main_pdf_key(p)]["text"] = p
    return dict(out)


def _discover_inprocess_files(inprocess_results: Path) -> dict[str, Path]:
    """
    Discovers per-PDF JSON files in the inprocess results directory.

    File naming typically looks like:
      1015168.20260210T030444Z.e7cfd55b.json

    We derive the PDF key from the filename without loading the (potentially huge) JSON:
      1015168.pdf
    Returns:
      { pdf_key: inprocess_json_path }
    """
    out: dict[str, Path] = {}
    for p in sorted(inprocess_results.glob("*.json")):
        stem = p.name[:-len(".json")] if p.name.endswith(".json") else p.stem
        parts = stem.split(".")
        # If the name matches "<pdfstem>.<timestamp>.<hash>", keep everything except the last 2 parts.
        # This is resilient to dots in the original PDF stem.
        if len(parts) >= 3:
            pdf_stem = ".".join(parts[:-2])
        else:
            pdf_stem = parts[0]
        pdf_key = pdf_stem if pdf_stem.endswith(".pdf") else f"{pdf_stem}.pdf"
        out[pdf_key] = p
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def compare_results(
    nv_ingest_main_results: Path = typer.Option(
        ...,
        "--nv-ingest-main-results",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory like /home/jdyer/datasets/nv-ingest-main-results-jp20 (contains structured/ and text/).",
    ),
    inprocess_results: Path = typer.Option(
        ...,
        "--inprocess-results",
        "--jp20-results-inprocess",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory like /home/jdyer/datasets/jp20_results_inprocess (contains per-PDF .json).",
    ),
    only: list[str] = typer.Option(
        [],
        "--only",
        help="Only compare PDFs whose key matches this value (repeatable). Accepts '1015168' or '1015168.pdf'.",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Limit how many PDFs to compare (0 = all).",
    ),
    show_page_diffs: bool = typer.Option(
        True,
        "--show-page-diffs/--no-show-page-diffs",
        help="Show per-page count diffs (tables/charts/infographics).",
    ),
    max_page_diffs: int = typer.Option(
        25,
        "--max-page-diffs",
        help="Max per-page diff rows to show per PDF (0 = unlimited).",
    ),
    min_text_similarity: float = typer.Option(
        0.985,
        "--min-text-similarity",
        help="Mark document text as DIFF when similarity ratio drops below this threshold.",
    ),
    out_dir: Path = typer.Option(
        Path("compare_results_out"),
        "--out-dir",
        "-o",
        help="Directory where per-PDF .txt and .diff.json files are written.",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    fail_on_diff: bool = typer.Option(
        False,
        "--fail-on-diff",
        help="Exit with code 1 if any differences are found.",
    ),
) -> None:
    console = Console(color_system=None if no_color else "auto")
    out_dir.mkdir(parents=True, exist_ok=True)

    main_files = _discover_main_files(nv_ingest_main_results)
    inproc_files = _discover_inprocess_files(inprocess_results)

    all_pdf_keys = sorted({*main_files.keys(), *inproc_files.keys()})

    if only:
        only_set = {o.strip() for o in only if o.strip()}
        filtered: list[str] = []
        for k in all_pdf_keys:
            variants = _simple_pdf_key_variants(k)
            if variants & only_set:
                filtered.append(k)
        all_pdf_keys = filtered

    if limit > 0:
        all_pdf_keys = all_pdf_keys[:limit]

    if not all_pdf_keys:
        raise typer.BadParameter("No matching PDFs found to compare.")

    any_diff = False

    for pdf_key in all_pdf_keys:
        mf = main_files.get(pdf_key, {})
        structured_path = mf.get("structured")
        text_path = mf.get("text")
        inproc_path = inproc_files.get(pdf_key)

        if inproc_path is None or (structured_path is None and text_path is None):
            # Missing pairing
            missing = []
            if structured_path is None:
                missing.append("nv-ingest-main structured")
            if text_path is None:
                missing.append("nv-ingest-main text")
            if inproc_path is None:
                missing.append("inprocess")

            # Render to screen
            console.print(Text(pdf_key, style="bold"))
            console.print(Text("Missing: " + ", ".join(missing), style="yellow"))
            console.print()

            # Also write per-PDF outputs
            safe = _slug_filename(pdf_key)
            txt_path = out_dir / f"{safe}.txt"
            json_path = out_dir / f"{safe}.diff.json"

            file_console = Console(record=True, color_system=None, width=140)
            file_console.print(Text(pdf_key, style="bold"))
            file_console.print(Text("Missing: " + ", ".join(missing), style="yellow"))
            file_console.print()

            txt_path.write_text(file_console.export_text(), encoding="utf-8")
            json_payload = _compute_diff_json(main=None, inp=None, missing=missing, min_text_similarity=min_text_similarity)
            json_payload["pdf_key"] = pdf_key
            json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

            any_diff = True
            continue

        main_norm = _normalize_main(pdf_key=pdf_key, structured_path=structured_path, text_path=text_path)
        inproc_norm = _normalize_inprocess(path=inproc_path)

        # If source_path-derived key disagrees, still compare but show a hint.
        if inproc_norm.pdf_key != pdf_key:
            console.print(Text(f"Note: inprocess source_path key '{inproc_norm.pdf_key}' != '{pdf_key}'", style="dim"))

        # Render to screen
        this_has_diff = _render_pdf_summary(
            console=console,
            main=main_norm,
            inp=inproc_norm,
            show_page_diffs=show_page_diffs,
            max_page_diffs=max_page_diffs,
            min_text_similarity=min_text_similarity,
        )
        any_diff |= this_has_diff

        # Render to file (no ANSI, wide)
        safe = _slug_filename(pdf_key)
        txt_path = out_dir / f"{safe}.txt"
        json_path = out_dir / f"{safe}.diff.json"

        file_console = Console(record=True, color_system=None, width=140)
        _ = _render_pdf_summary(
            console=file_console,
            main=main_norm,
            inp=inproc_norm,
            show_page_diffs=show_page_diffs,
            max_page_diffs=0,  # don't truncate in file (table itself prints only diffs anyway)
            min_text_similarity=min_text_similarity,
        )
        txt_path.write_text(file_console.export_text(), encoding="utf-8")

        json_payload = _compute_diff_json(main=main_norm, inp=inproc_norm, missing=None, min_text_similarity=min_text_similarity)
        json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    if any_diff and fail_on_diff:
        raise typer.Exit(code=1)


def main() -> None:
    app()
