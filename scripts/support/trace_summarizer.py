#!/usr/bin/env python3
"""
Trace Summarizer

Parses NV-Ingest trace maps (trace::entry::<name>, trace::exit::<name>) and computes
total time spent in each area. Aggregates across numbered suffixes like _0, _1.

Usage:
  - From file(s):
      python scripts/support/trace_summarizer.py traces.json [more.json]
  - From stdin:
      cat traces.json | python scripts/support/trace_summarizer.py -

Options:
  --units [ns|ms|s]     Output units (default: ms)
  --json                Output JSON summary
  --top N               Show top N entries only
"""

import sys
import json
import re
import argparse
import os
import glob
import math
from collections import defaultdict
from typing import Dict, Any, List, Tuple


ENTRY_PREFIX = "trace::entry::"
EXIT_PREFIX = "trace::exit::"
SUFFIX_NUM_RE = re.compile(r"(.*)_(\d+)$")


def _normalize_name(name: str) -> str:
    """
    Normalize a trace name by stripping trailing _<digits> to aggregate repeated items.
    Example: "pdf_extractor::pdf_extraction::pdfium_pages_to_numpy_0" -> "...::pdfium_pages_to_numpy"
    """
    m = SUFFIX_NUM_RE.match(name)
    return m.group(1) if m else name


def _load_trace_map(path: str) -> Dict[str, Any]:
    if path == "-":
        data = sys.stdin.read()
        return json.loads(data)
    with open(path, "r") as f:
        return json.load(f)


def _iter_entries(trace_map: Dict[str, Any]) -> List[Tuple[str, int]]:
    """
    Return list of (name, entry_ts) for all entry keys.
    """
    out = []
    for k, v in trace_map.items():
        if isinstance(k, str) and k.startswith(ENTRY_PREFIX):
            name = k[len(ENTRY_PREFIX) :]
            try:
                ts = int(v)
            except Exception:
                # Attempt float -> int if provided as float
                ts = int(float(v))
            out.append((name, ts))
    return out


def _get_exit_ts(trace_map: Dict[str, Any], name: str) -> int | None:
    k = EXIT_PREFIX + name
    if k not in trace_map:
        return None
    v = trace_map[k]
    try:
        return int(v)
    except Exception:
        return int(float(v))


def _convert_units(ns: int, units: str) -> float:
    if units == "ns":
        return float(ns)
    if units == "ms":
        return ns / 1e6
    if units == "s":
        return ns / 1e9
    raise ValueError("Unsupported units: " + units)


def summarize_durations(trace_maps: List[Dict[str, Any]], normalize_suffixes: bool = True) -> Dict[str, List[int]]:
    """
    Compute list of durations (in ns) per normalized area name across all maps.
    """
    durations: Dict[str, List[int]] = defaultdict(list)
    for trace_map in trace_maps:
        entries = _iter_entries(trace_map)
        for name, ts_entry in entries:
            ts_exit = _get_exit_ts(trace_map, name)
            if ts_exit is None:
                print(f"[warn] missing exit for '{name}'", file=sys.stderr)
                continue
            if ts_exit < ts_entry:
                print(f"[warn] exit < entry for '{name}'", file=sys.stderr)
                continue
            norm = _normalize_name(name) if normalize_suffixes else name
            durations[norm].append(ts_exit - ts_entry)
    return dict(durations)


def summarize(trace_maps: List[Dict[str, Any]], normalize_suffixes: bool = True) -> Dict[str, int]:
    """
    Compute total durations in nanoseconds per normalized area name across all maps.
    """
    totals_ns: Dict[str, int] = defaultdict(int)
    durations = summarize_durations(trace_maps, normalize_suffixes=normalize_suffixes)
    for k, vals in durations.items():
        totals_ns[k] = sum(vals)
    return dict(totals_ns)


def main():
    ap = argparse.ArgumentParser(description="Summarize NV-Ingest trace timings.")
    ap.add_argument("inputs", nargs="+", help="JSON files or '-' for stdin")
    ap.add_argument("--units", choices=["ns", "ms", "s"], default="ms", help="Output units (default: ms)")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of table")
    ap.add_argument("--tree", action="store_true", help="Output hierarchical, indented tree view")
    ap.add_argument(
        "--no-aggregate-suffixes",
        action="store_true",
        help="Do not strip trailing numeric suffixes; keep entries like *_0, *_1 separate",
    )
    ap.add_argument(
        "--cumulative",
        action="store_true",
        help="Tree mode: show parent totals including the sum of all descendants",
    )
    ap.add_argument("--top", type=int, default=0, help="Show top N entries")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum fraction (0..1) of total aggregate time a stage must account for to be shown",
    )
    ap.add_argument(
        "--exclude-channel-in",
        action="store_true",
        help="Exclude entries whose names contain 'channel_in' or 'network_in' from listings",
    )
    args = ap.parse_args()

    trace_maps = []
    # Aggregate primitive counts across inputs if present
    primitive_total = 0
    primitive_by_type = defaultdict(int)
    structured_by_subtype = defaultdict(int)
    # Expand inputs: allow directories and glob patterns
    expanded_inputs: List[str] = []
    for spec in args.inputs:
        if spec == "-":
            expanded_inputs.append(spec)
            continue
        if os.path.isdir(spec):
            # Aggregate all *.traces.json files in the directory (non-recursive)
            expanded_inputs.extend(sorted(glob.glob(os.path.join(spec, "*.traces.json"))))
            continue
        # Glob pattern (supports recursive with **)
        matches = glob.glob(spec, recursive=True)
        if matches:
            expanded_inputs.extend(sorted(matches))
        else:
            # Fallback: treat as a single file path
            expanded_inputs.append(spec)

    for p in expanded_inputs:
        try:
            m = _load_trace_map(p)
            trace_maps.append(m)
            pc = m.get("primitive_counts")
            if isinstance(pc, dict):
                try:
                    primitive_total += int(pc.get("total", 0))
                except Exception:
                    pass
                by_type = pc.get("by_type") or {}
                if isinstance(by_type, dict):
                    for k, v in by_type.items():
                        try:
                            primitive_by_type[k] += int(v)
                        except Exception:
                            continue
                by_sub = pc.get("structured_by_subtype") or {}
                if isinstance(by_sub, dict):
                    for k, v in by_sub.items():
                        try:
                            structured_by_subtype[k] += int(v)
                        except Exception:
                            continue
        except Exception as e:
            print(f"[error] failed to load {p}: {e}", file=sys.stderr)
            return 2

    normalize_suffixes = not args.no_aggregate_suffixes
    durations_by_name = summarize_durations(trace_maps, normalize_suffixes=normalize_suffixes)

    # Optionally exclude channel_in/network_in entries
    if args.exclude_channel_in:
        durations_by_name = {
            k: v for k, v in durations_by_name.items() if ("channel_in" not in k and "network_in" not in k)
        }

    # Compute stats per name
    def _percentile(values: List[int], p: float) -> float:
        if not values:
            return 0.0
        vs = sorted(values)
        n = len(vs)
        # Nearest-rank method
        rank = max(1, int(math.ceil(p * n)))
        return float(vs[rank - 1])

    stats_map: Dict[str, Dict[str, float]] = {}
    for name, vals in durations_by_name.items():
        total = float(sum(vals))
        count = len(vals)
        mean = (total / count) if count else 0.0
        p95 = _percentile(vals, 0.95)
        p99 = _percentile(vals, 0.99)
        stats_map[name] = {
            "total_ns": total,
            "count": count,
            "mean_ns": mean,
            "p95_ns": p95,
            "p99_ns": p99,
        }

    # Apply threshold filtering as a fraction of total aggregate time
    total_ns_all = sum(meta["total_ns"] for meta in stats_map.values())
    if args.threshold and total_ns_all > 0:
        stats_map = {
            name: meta for name, meta in stats_map.items() if (meta["total_ns"] / total_ns_all) >= args.threshold
        }

    # Sort by total time desc
    items = sorted(stats_map.items(), key=lambda kv: kv[1]["total_ns"], reverse=True)

    if args.json:
        out = {
            name: {
                "total": _convert_units(int(meta["total_ns"]), args.units),
                "count": int(meta["count"]),
                "mean": _convert_units(int(meta["mean_ns"]), args.units),
                "p95": _convert_units(int(meta["p95_ns"]), args.units),
                "p99": _convert_units(int(meta["p99_ns"]), args.units),
            }
            for name, meta in items
        }
        print(json.dumps(out, indent=2))
        return 0

    if args.tree:
        # Print primitive distribution before timing tree (non-JSON)
        print("primitive distribution")
        print("-" * 98)
        print(f"{'total':<80} {primitive_total:>16}")
        if primitive_by_type:
            for k, v in sorted(primitive_by_type.items(), key=lambda kv: kv[1], reverse=True):
                print(f"{k:<80} {v:>16}")
        if structured_by_subtype:
            print()
            print("structured by subtype")
            print("-" * 98)
            for k, v in sorted(structured_by_subtype.items(), key=lambda kv: kv[1], reverse=True):
                print(f"{k:<80} {v:>16}")
        print()

        # Human-readable hierarchical tree
        def build_tree(pairs: List[Tuple[str, int]]):
            tree = {}
            for full_name, total_ns in pairs:
                parts = full_name.split("::") if full_name else [full_name]
                cur = tree
                for i, part in enumerate(parts):
                    if part not in cur:
                        cur[part] = {"__total_ns__": 0, "__children__": {}}
                    if i == len(parts) - 1:
                        cur[part]["__total_ns__"] += total_ns
                    cur = cur[part]["__children__"]
            return tree

        def flatten_tree(node: dict, level: int = 0):
            rows = []
            children = [(k, v) for k, v in node.items() if not k.startswith("__")]
            children.sort(key=lambda kv: kv[1]["__total_ns__"], reverse=True)
            for name, meta in children:
                rows.append((level, name, meta["__total_ns__"]))
                rows.extend(flatten_tree(meta["__children__"], level + 1))
            return rows

        # Apply --top to root-level only (tree mode)
        root_totals = defaultdict(int)
        for name, meta in items:
            total_ns = int(meta["total_ns"])
            root = name.split("::")[0]
            root_totals[root] += total_ns
        sorted_roots = sorted(root_totals.items(), key=lambda kv: kv[1], reverse=True)
        if args.top and args.top > 0:
            keep_roots = set([r for r, _ in sorted_roots[: args.top]])
            filtered_items = [
                (name, int(meta["total_ns"])) for name, meta in items if name.split("::")[0] in keep_roots
            ]
        else:
            filtered_items = [(name, int(meta["total_ns"])) for name, meta in items]

        tree = build_tree(filtered_items)
        rows = flatten_tree(tree)
        # Print as table with indentation for the area column
        col1 = "area"
        col2 = f"total ({args.units})"
        col3 = "count"
        col4 = f"mean ({args.units})"
        col5 = f"p95 ({args.units})"
        col6 = f"p99 ({args.units})"
        print(f"{col1:<60} {col2:>12} {col3:>8} {col4:>12} {col5:>12} {col6:>12}")
        print("-" * 120)
        for level, name, total_ns in rows:
            total_val = _convert_units(total_ns, args.units)
            display = f"{'  ' * level}{name}"
            meta = stats_map.get(name)
            if meta:
                count = int(meta["count"])
                mean_val = _convert_units(int(meta["mean_ns"]), args.units)
                p95_val = _convert_units(int(meta["p95_ns"]), args.units)
                p99_val = _convert_units(int(meta["p99_ns"]), args.units)
                print(
                    f"{display:<60} {total_val:>12.3f} {count:>8} {mean_val:>12.3f} {p95_val:>12.3f} {p99_val:>12.3f}"
                )
            else:
                print(f"{display:<60} {total_val:>12.3f} {'-':>8} {'-':>12} {'-':>12} {'-':>12}")
    else:
        # Print primitive distribution before flat table (non-JSON)
        print("primitive distribution")
        print("-" * 98)
        print(f"{'total':<80} {primitive_total:>16}")
        if primitive_by_type:
            for k, v in sorted(primitive_by_type.items(), key=lambda kv: kv[1], reverse=True):
                print(f"{k:<80} {v:>16}")
        if structured_by_subtype:
            print()
            print("structured by subtype")
            print("-" * 98)
            for k, v in sorted(structured_by_subtype.items(), key=lambda kv: kv[1], reverse=True):
                print(f"{k:<80} {v:>16}")
        print()
        # Flat table output with stats
        flat_items = items[: args.top] if (args.top and args.top > 0) else items
        col1 = "area"
        col2 = f"total ({args.units})"
        col3 = "count"
        col4 = f"mean ({args.units})"
        col5 = f"p95 ({args.units})"
        col6 = f"p99 ({args.units})"
        print(f"{col1:<60} {col2:>12} {col3:>8} {col4:>12} {col5:>12} {col6:>12}")
        print("-" * 120)
        for name, meta in flat_items:
            total_val = _convert_units(int(meta["total_ns"]), args.units)
            count = int(meta["count"])
            mean_val = _convert_units(int(meta["mean_ns"]), args.units)
            p95_val = _convert_units(int(meta["p95_ns"]), args.units)
            p99_val = _convert_units(int(meta["p99_ns"]), args.units)
            print(f"{name:<60} {total_val:>12.3f} {count:>8} {mean_val:>12.3f} {p95_val:>12.3f} {p99_val:>12.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
