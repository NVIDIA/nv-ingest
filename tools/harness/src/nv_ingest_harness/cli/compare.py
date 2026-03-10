# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""CLI for comparing benchmark runs."""

import sys
from pathlib import Path

import click

from nv_ingest_harness.compare import (
    compute_deltas,
    format_cli_table,
    format_markdown,
    load_results,
    post_to_slack,
    is_session_dir,
    load_session,
    compare_sessions,
    format_session_cli_table,
    format_session_markdown,
    post_session_to_slack,
)


@click.command()
@click.argument("run_a", type=click.Path(exists=True))
@click.argument("run_b", type=click.Path(exists=True))
@click.option(
    "--slack",
    is_flag=True,
    help="Post comparison to Slack (uses SLACK_WEBHOOK_URL env var)",
)
@click.option(
    "--webhook-url",
    type=str,
    default=None,
    help="Slack webhook URL (overrides SLACK_WEBHOOK_URL env var)",
)
@click.option(
    "--markdown",
    type=click.Path(),
    default=None,
    help="Write comparison to markdown file",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress CLI table output (useful with --slack or --markdown)",
)
@click.option(
    "--baseline-label",
    "-b",
    type=str,
    default=None,
    help="Custom label for baseline run (e.g., 'Main RC')",
)
@click.option(
    "--compare-label",
    "-c",
    type=str,
    default=None,
    help="Custom label for comparison run (e.g., 'NIM RC 1.7.1')",
)
@click.option(
    "--note",
    "-n",
    type=str,
    default=None,
    help="Note describing what's being tested (e.g., 'Testing new NIM RC drops')",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=2.0,
    help="Significance threshold %% for marking regressions (default: 2.0)",
)
def compare(
    run_a: str,
    run_b: str,
    slack: bool,
    webhook_url: str | None,
    markdown: str | None,
    quiet: bool,
    baseline_label: str | None,
    compare_label: str | None,
    note: str | None,
    threshold: float,
) -> None:
    """Compare two benchmark runs or nightly sessions side-by-side.

    RUN_A and RUN_B are paths to artifact directories.

    For single runs: paths to directories containing results.json files.
    For sessions: paths to nightly session directories (containing session_summary.json).

    Examples:

        # Compare two single runs with custom labels and a note
        nv-ingest-harness-compare artifacts/bo767_20260109_053814_UTC artifacts/nim_rc_1.7.1_20260109_204419_UTC \\
            --baseline-label "Main RC" --compare-label "NIM RC 1.7.1" \\
            --note "Testing new NIM engineering drops"

        # Compare two nightly sessions and post to Slack
        nv-ingest-harness-compare artifacts/nightly_20260109 artifacts/nightly_20260112 \\
            -b "Baseline Session" -c "Experiment Session" --slack

        # Adjust significance threshold (only mark >5% changes as regressions)
        nv-ingest-harness-compare artifacts/bo20_20260109 artifacts/bo20_20260110 --threshold 5.0
    """
    try:
        path_a = Path(run_a)
        path_b = Path(run_b)

        display_opts = {
            "baseline_label": baseline_label,
            "compare_label": compare_label,
            "note": note,
            "threshold": threshold,
        }

        is_session_a = is_session_dir(path_a)
        is_session_b = is_session_dir(path_b)

        if is_session_a != is_session_b:
            click.echo(
                "Error: Cannot compare a session with a single run. Both must be sessions or both must be single runs.",
                err=True,
            )
            sys.exit(1)

        if is_session_a:
            session_a = load_session(path_a)
            session_b = load_session(path_b)

            result = compare_sessions(session_a, session_b)

            if not quiet:
                print(format_session_cli_table(result, **display_opts))

            if markdown:
                md_content = format_session_markdown(result, **display_opts)
                Path(markdown).write_text(md_content)
                print(f"\nMarkdown written to: {markdown}")

            if slack:
                success = post_session_to_slack(result, webhook_url, **display_opts)
                if not success:
                    sys.exit(1)
        else:
            data_a = load_results(run_a)
            data_b = load_results(run_b)

            comparison = compute_deltas(data_a, data_b)

            if not quiet:
                print(format_cli_table(comparison, **display_opts))

            if markdown:
                md_content = format_markdown(comparison, **display_opts)
                Path(markdown).write_text(md_content)
                print(f"\nMarkdown written to: {markdown}")

            if slack:
                success = post_to_slack(comparison, webhook_url, **display_opts)
                if not success:
                    sys.exit(1)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    compare()


if __name__ == "__main__":
    main()
