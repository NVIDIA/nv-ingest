# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Comparison utilities for benchmark runs."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nv_ingest_harness.sinks.slack import SlackSink

# Metrics to compare, grouped by category
THROUGHPUT_METRICS = ["pages_per_second", "ingestion_time_s"]
VOLUME_METRICS = ["total_pages", "result_count", "failure_count"]
EXTRACTION_METRICS = ["text_chunks", "table_chunks", "chart_chunks"]
RECALL_METRICS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "recall@1_reranker",
    "recall@3_reranker",
    "recall@5_reranker",
    "recall@10_reranker",
]

ALL_METRICS = THROUGHPUT_METRICS + VOLUME_METRICS + EXTRACTION_METRICS + RECALL_METRICS

# Metrics where lower is better (for delta coloring)
LOWER_IS_BETTER = {"ingestion_time_s", "failure_count"}


@dataclass
class RunData:
    """Parsed benchmark run data."""

    path: str
    name: str
    timestamp: str
    git_commit: str | None
    dataset: str | None
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""

    name: str
    value_a: float | int | None
    value_b: float | int | None
    delta: float | None  # Percentage change
    improved: bool | None  # True if change is an improvement


@dataclass
class ComparisonResult:
    """Full comparison between two runs."""

    run_a: RunData
    run_b: RunData
    metrics: list[MetricComparison] = field(default_factory=list)


@dataclass
class SessionData:
    """Parsed nightly session data."""

    path: str
    name: str
    timestamp: str
    runs: dict[str, RunData] = field(default_factory=dict)  # dataset -> RunData


@dataclass
class SessionComparisonResult:
    """Full comparison between two sessions."""

    session_a: SessionData
    session_b: SessionData
    comparisons: dict[str, ComparisonResult] = field(default_factory=dict)  # dataset -> ComparisonResult
    datasets_only_in_a: list[str] = field(default_factory=list)
    datasets_only_in_b: list[str] = field(default_factory=list)


def load_results(path: str | Path) -> RunData:
    """Load results.json from an artifact directory."""
    path = Path(path)

    # Handle both directory and direct file paths
    if path.is_dir():
        results_file = path / "results.json"
    else:
        results_file = path

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file) as f:
        raw = json.load(f)

    # Extract name from path or test_config
    name = raw.get("test_config", {}).get("test_name") or path.parent.name if path.is_file() else path.name

    # Extract metrics from results (or ingestion_results for older format)
    ingestion = raw.get("results", {}) or raw.get("ingestion_results", {})
    metrics: dict[str, float | int | None] = {}

    for metric in THROUGHPUT_METRICS + VOLUME_METRICS + EXTRACTION_METRICS:
        if metric in ingestion:
            metrics[metric] = ingestion[metric]

    # Extract recall metrics from recall_results
    recall = raw.get("recall_results", {})
    if recall:
        no_reranker = recall.get("no_reranker", {})
        with_reranker = recall.get("with_reranker", {})

        for k in ["1", "3", "5", "10"]:
            if k in no_reranker:
                metrics[f"recall@{k}"] = no_reranker[k]
            if k in with_reranker:
                metrics[f"recall@{k}_reranker"] = with_reranker[k]

    return RunData(
        path=str(path),
        name=name,
        timestamp=raw.get("timestamp", "unknown"),
        git_commit=raw.get("latest_commit"),
        dataset=raw.get("test_config", {}).get("test_name"),
        metrics=metrics,
        raw=raw,
    )


def is_session_dir(path: str | Path) -> bool:
    """Check if a path is a nightly session directory."""
    path = Path(path)
    return (path / "session_summary.json").exists()


def load_session(path: str | Path) -> SessionData:
    """Load all results from a nightly session directory."""
    path = Path(path)

    if not path.is_dir():
        raise ValueError(f"Session path must be a directory: {path}")

    # Load session summary if available
    summary_file = path / "session_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        timestamp = summary.get("timestamp", "unknown")
    else:
        # No summary file - still try to load subdirectories
        timestamp = "unknown"

    # Load all dataset subdirectories that have results.json
    runs: dict[str, RunData] = {}
    for subdir in sorted(path.iterdir()):
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                try:
                    run_data = load_results(subdir)
                    runs[subdir.name] = run_data
                except Exception as e:
                    print(f"Warning: Could not load {subdir}: {e}")

    return SessionData(
        path=str(path),
        name=path.name,
        timestamp=timestamp,
        runs=runs,
    )


def compare_sessions(session_a: SessionData, session_b: SessionData) -> SessionComparisonResult:
    """Compare two nightly sessions dataset-by-dataset."""
    datasets_a = set(session_a.runs.keys())
    datasets_b = set(session_b.runs.keys())

    common = datasets_a & datasets_b
    only_a = sorted(datasets_a - datasets_b)
    only_b = sorted(datasets_b - datasets_a)

    comparisons: dict[str, ComparisonResult] = {}
    for dataset in sorted(common):
        comparisons[dataset] = compute_deltas(
            session_a.runs[dataset],
            session_b.runs[dataset],
        )

    return SessionComparisonResult(
        session_a=session_a,
        session_b=session_b,
        comparisons=comparisons,
        datasets_only_in_a=only_a,
        datasets_only_in_b=only_b,
    )


def compute_deltas(run_a: RunData, run_b: RunData) -> ComparisonResult:
    """Compute percentage deltas between two runs."""
    comparisons: list[MetricComparison] = []

    # Gather all metrics from both runs
    all_metric_names = set(run_a.metrics.keys()) | set(run_b.metrics.keys())

    # Sort by our defined order, then alphabetically for any extras
    def sort_key(m: str) -> tuple[int, str]:
        if m in ALL_METRICS:
            return (ALL_METRICS.index(m), m)
        return (len(ALL_METRICS), m)

    for metric_name in sorted(all_metric_names, key=sort_key):
        val_a = run_a.metrics.get(metric_name)
        val_b = run_b.metrics.get(metric_name)

        delta: float | None = None
        improved: bool | None = None

        if val_a is not None and val_b is not None and val_a != 0:
            delta = ((val_b - val_a) / abs(val_a)) * 100

            # Determine if this is an improvement
            if metric_name in LOWER_IS_BETTER:
                improved = delta < 0  # Lower is better
            else:
                improved = delta > 0  # Higher is better

        comparisons.append(
            MetricComparison(
                name=metric_name,
                value_a=val_a,
                value_b=val_b,
                delta=delta,
                improved=improved,
            )
        )

    return ComparisonResult(run_a=run_a, run_b=run_b, metrics=comparisons)


def format_value(metric_name: str, value: float | int | None) -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"

    if metric_name.endswith("_s"):
        # Time in seconds - show with units
        seconds = float(value)
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes:02}m {secs:05.2f}s"
        elif seconds >= 60:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:05.2f}s"
        return f"{seconds:.2f}s"

    if metric_name.startswith("recall@"):
        return f"{float(value):.3f}"

    if metric_name == "pages_per_second":
        return f"{float(value):.2f}"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


def format_delta(delta: float | None, improved: bool | None) -> str:
    """Format a delta value with sign."""
    if delta is None:
        return "N/A"

    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def format_cli_table(
    comparison: ComparisonResult,
    baseline_label: str | None = None,
    compare_label: str | None = None,
    note: str | None = None,
    threshold: float = 2.0,
) -> str:
    """Format comparison as a rich CLI table."""
    try:
        from rich.console import Console
        from rich.table import Table
        from io import StringIO

        console = Console(file=StringIO(), force_terminal=True, width=100)

        label_a = baseline_label or comparison.run_a.name
        label_b = compare_label or comparison.run_b.name

        console.print("\n[bold cyan]Benchmark Comparison[/bold cyan]")
        console.print(f"[dim]Baseline:[/dim] [yellow]{label_a}[/yellow]")
        console.print(f"[dim]Compare:[/dim]  [yellow]{label_b}[/yellow]")
        if note:
            console.print(f"[dim]Note:[/dim]     [italic]{note}[/italic]")
        console.print()

        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column(label_a, justify="right")
        table.add_column(label_b, justify="right")
        table.add_column("Delta", justify="right")

        for m in comparison.metrics:
            val_a_str = format_value(m.name, m.value_a)
            val_b_str = format_value(m.name, m.value_b)
            delta_str = format_delta(m.delta, m.improved)

            if m.delta is not None and abs(m.delta) >= threshold:
                if m.improved is True:
                    delta_str = f"[green]✓ {delta_str}[/green]"
                elif m.improved is False:
                    delta_str = f"[red]✗ {delta_str}[/red]"
            else:
                delta_str = f"[dim]{delta_str}[/dim]"

            table.add_row(m.name, val_a_str, val_b_str, delta_str)

        console.print(table)
        return console.file.getvalue()

    except ImportError:
        # Fallback to simple text table
        return format_simple_table(comparison)


def format_simple_table(comparison: ComparisonResult) -> str:
    """Format comparison as a simple text table (fallback)."""
    lines = []
    lines.append(f"{'Metric':<25} {'Baseline':>15} {'Compare':>15} {'Delta':>10}")
    lines.append("-" * 70)

    for m in comparison.metrics:
        val_a_str = format_value(m.name, m.value_a)
        val_b_str = format_value(m.name, m.value_b)
        delta_str = format_delta(m.delta, m.improved)
        lines.append(f"{m.name:<25} {val_a_str:>15} {val_b_str:>15} {delta_str:>10}")

    return "\n".join(lines)


def format_markdown(
    comparison: ComparisonResult,
    baseline_label: str | None = None,
    compare_label: str | None = None,
    note: str | None = None,
    threshold: float = 2.0,
) -> str:
    """Format comparison as a markdown table."""
    label_a = baseline_label or comparison.run_a.name
    label_b = compare_label or comparison.run_b.name

    lines = []
    lines.append("# Benchmark Comparison")
    lines.append("")
    lines.append(f"**Baseline:** {label_a} (`{comparison.run_a.timestamp}`)")
    lines.append(f"**Compare:** {label_b} (`{comparison.run_b.timestamp}`)")
    if note:
        lines.append("")
        lines.append(f"> {note}")
    lines.append("")

    if comparison.run_a.git_commit or comparison.run_b.git_commit:
        lines.append("## Git Commits")
        if comparison.run_a.git_commit:
            lines.append(f"- Baseline: `{comparison.run_a.git_commit[:8]}`")
        if comparison.run_b.git_commit:
            lines.append(f"- Compare: `{comparison.run_b.git_commit[:8]}`")
        lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append(f"| Metric | {label_a} | {label_b} | Delta |")
    lines.append("|--------|----------|----------|-------|")

    for m in comparison.metrics:
        val_a_str = format_value(m.name, m.value_a)
        val_b_str = format_value(m.name, m.value_b)
        delta_str = format_delta(m.delta, m.improved)

        if m.delta is not None and abs(m.delta) >= threshold:
            if m.improved is True:
                delta_str = f"✅ {delta_str}"
            elif m.improved is False:
                delta_str = f"❌ {delta_str}"

        lines.append(f"| {m.name} | {val_a_str} | {val_b_str} | {delta_str} |")

    return "\n".join(lines)


def format_slack_blocks(
    comparison: ComparisonResult,
    baseline_label: str | None = None,
    compare_label: str | None = None,
    note: str | None = None,
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Format comparison as Slack Block Kit payload."""
    # TODO: Consolidate Slack formatting into SlackSink helpers when PR scope allows.
    label_a = baseline_label or comparison.run_a.name
    label_b = compare_label or comparison.run_b.name

    rows = []
    rows.append(_slack_row_bold("Metric", f"{label_a} → {label_b}"))

    for m in comparison.metrics:
        val_a_str = format_value(m.name, m.value_a)
        val_b_str = format_value(m.name, m.value_b)
        delta_str = format_delta(m.delta, m.improved)

        if m.delta is not None and abs(m.delta) >= threshold:
            if m.improved is True:
                delta_str = f"✅ {delta_str}"
            elif m.improved is False:
                delta_str = f"❌ {delta_str}"

        rows.append(_slack_row(m.name, f"{val_a_str} → {val_b_str} ({delta_str})"))

    desc_text = (
        f"*Baseline:* `{label_a}` ({comparison.run_a.timestamp})\n*Compare:* `{label_b}` ({comparison.run_b.timestamp})"
    )
    if note:
        desc_text += f"\n\n_{note}_"

    return {
        "username": "nv-ingest Benchmark Runner",
        "icon_emoji": ":bar_chart:",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Benchmark Comparison"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": desc_text,
                },
            },
            {"type": "divider"},
            {"type": "table", "rows": rows},
        ],
    }


def _slack_row(left: str, right: str) -> list[dict[str, Any]]:
    """Create a two-column Slack table row."""
    return [
        {
            "type": "rich_text",
            "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": left}]}],
        },
        {
            "type": "rich_text",
            "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": right}]}],
        },
    ]


def _slack_row_bold(left: str, right: str) -> list[dict[str, Any]]:
    """Create a two-column Slack table row with bold left column."""
    return [
        {
            "type": "rich_text",
            "elements": [
                {"type": "rich_text_section", "elements": [{"type": "text", "text": left, "style": {"bold": True}}]}
            ],
        },
        {
            "type": "rich_text",
            "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": right}]}],
        },
    ]


def post_to_slack(
    comparison: ComparisonResult,
    webhook_url: str | None = None,
    baseline_label: str | None = None,
    compare_label: str | None = None,
    note: str | None = None,
    threshold: float = 2.0,
) -> bool:
    """Post comparison to Slack webhook."""
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        print("Error: No Slack webhook URL provided. Set SLACK_WEBHOOK_URL env var or use --webhook-url.")
        return False

    payload = format_slack_blocks(comparison, baseline_label, compare_label, note, threshold)
    sink = SlackSink({"enabled": True, "webhook_url": url})
    if sink.post_payload(payload):
        print("Successfully posted comparison to Slack.")
        return True
    return False


def format_session_cli_table(
    result: SessionComparisonResult,
    baseline_label: str | None = None,
    compare_label: str | None = None,
    note: str | None = None,
    threshold: float = 2.0,
) -> str:
    """Format session comparison as a CLI table with rich colors."""
    from rich.console import Console
    from rich.table import Table
    from io import StringIO

    console = Console(file=StringIO(), force_terminal=True, width=140)

    label_a = baseline_label or result.session_a.name
    label_b = compare_label or result.session_b.name

    console.print("\n[bold cyan]Session Comparison[/bold cyan]")
    console.print(f"[dim]Baseline:[/dim] [yellow]{label_a}[/yellow]")
    console.print(f"[dim]Compare:[/dim]  [yellow]{label_b}[/yellow]")
    if note:
        console.print(f"[dim]Note:[/dim]     [italic]{note}[/italic]")
    console.print()

    if result.datasets_only_in_a:
        console.print(f"[dim yellow]Datasets only in baseline: {', '.join(result.datasets_only_in_a)}[/dim yellow]")
    if result.datasets_only_in_b:
        console.print(f"[dim yellow]Datasets only in compare: {', '.join(result.datasets_only_in_b)}[/dim yellow]")

    for dataset, comparison in result.comparisons.items():
        table = Table(title=f"[bold]{dataset}[/bold]", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", min_width=20)
        table.add_column(label_a, justify="right", min_width=12)
        table.add_column(label_b, justify="right", min_width=12)
        table.add_column("Delta", justify="right", min_width=12)

        for m in comparison.metrics:
            val_a_str = format_value(m.name, m.value_a)
            val_b_str = format_value(m.name, m.value_b)
            delta_str = format_delta(m.delta, m.improved)

            if m.delta is not None and abs(m.delta) >= threshold:
                if m.improved is True:
                    delta_str = f"[green]✓ {delta_str}[/green]"
                elif m.improved is False:
                    delta_str = f"[red]✗ {delta_str}[/red]"
            else:
                delta_str = f"[dim]{delta_str}[/dim]"

            table.add_row(m.name, val_a_str, val_b_str, delta_str)

        console.print(table)
        console.print()

    return console.file.getvalue()


def format_session_slack_blocks(
    result: SessionComparisonResult,
    baseline_label: str | None = None,
    compare_label: str | None = None,
    note: str | None = None,
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Format session comparison as Slack Block Kit message."""
    label_a = baseline_label or result.session_a.name
    label_b = compare_label or result.session_b.name

    desc_text = (
        f"*Baseline:* `{label_a}` ({result.session_a.timestamp})\n*Compare:* `{label_b}` ({result.session_b.timestamp})"
    )
    if note:
        desc_text += f"\n\n_{note}_"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Nightly Session Comparison"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": desc_text,
            },
        },
    ]

    if result.datasets_only_in_a or result.datasets_only_in_b:
        warning_text = ""
        if result.datasets_only_in_a:
            warning_text += f"⚠️ Only in baseline: {', '.join(result.datasets_only_in_a)}\n"
        if result.datasets_only_in_b:
            warning_text += f"⚠️ Only in compare: {', '.join(result.datasets_only_in_b)}"
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": warning_text.strip()},
            }
        )

    blocks.append({"type": "divider"})

    for dataset, comparison in result.comparisons.items():
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{dataset}*"},
            }
        )

        rows = []
        rows.append(_slack_row_bold("Metric", f"{label_a} → {label_b}"))

        for m in comparison.metrics:
            val_a_str = format_value(m.name, m.value_a)
            val_b_str = format_value(m.name, m.value_b)
            delta_str = format_delta(m.delta, m.improved)

            if m.delta is not None and abs(m.delta) >= threshold:
                if m.improved is True:
                    delta_str = f"✅ {delta_str}"
                elif m.improved is False:
                    delta_str = f"❌ {delta_str}"

            rows.append(_slack_row(m.name, f"{val_a_str} → {val_b_str} ({delta_str})"))

        blocks.append({"type": "table", "rows": rows})

    return {
        "username": "nv-ingest Benchmark Runner",
        "icon_emoji": ":bar_chart:",
        "blocks": blocks,
    }


def format_session_markdown(
    result: SessionComparisonResult,
    baseline_label: str | None = None,
    compare_label: str | None = None,
    note: str | None = None,
    threshold: float = 2.0,
) -> str:
    """Format session comparison as markdown."""
    label_a = baseline_label or result.session_a.name
    label_b = compare_label or result.session_b.name

    lines = [
        "# Nightly Session Comparison",
        "",
        f"**Baseline:** `{label_a}` ({result.session_a.timestamp})",
        f"**Compare:** `{label_b}` ({result.session_b.timestamp})",
    ]

    if note:
        lines.append("")
        lines.append(f"> {note}")
    lines.append("")

    if result.datasets_only_in_a:
        lines.append(f"> ⚠️ Datasets only in baseline: {', '.join(result.datasets_only_in_a)}")
    if result.datasets_only_in_b:
        lines.append(f"> ⚠️ Datasets only in compare: {', '.join(result.datasets_only_in_b)}")

    for dataset, comparison in result.comparisons.items():
        lines.append("")
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"| Metric | {label_a} | {label_b} | Delta |")
        lines.append("|--------|----------|---------|-------|")

        for m in comparison.metrics:
            val_a_str = format_value(m.name, m.value_a)
            val_b_str = format_value(m.name, m.value_b)
            delta_str = format_delta(m.delta, m.improved)

            if m.delta is not None and abs(m.delta) >= threshold:
                if m.improved is True:
                    delta_str = f"✅ {delta_str}"
                elif m.improved is False:
                    delta_str = f"❌ {delta_str}"

            lines.append(f"| {m.name} | {val_a_str} | {val_b_str} | {delta_str} |")

    return "\n".join(lines)


def post_session_to_slack(
    result: SessionComparisonResult,
    webhook_url: str | None = None,
    baseline_label: str | None = None,
    compare_label: str | None = None,
    note: str | None = None,
    threshold: float = 2.0,
) -> bool:
    """Post session comparison to Slack webhook."""
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        print("Error: No Slack webhook URL provided. Set SLACK_WEBHOOK_URL env var or use --webhook-url.")
        return False

    payload = format_session_slack_blocks(result, baseline_label, compare_label, note, threshold)
    sink = SlackSink({"enabled": True, "webhook_url": url})
    if sink.post_payload(payload):
        print("Successfully posted session comparison to Slack.")
        return True
    return False
