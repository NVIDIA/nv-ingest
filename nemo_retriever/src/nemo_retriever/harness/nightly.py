# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import typer

from nemo_retriever.harness.artifacts import write_session_summary
from nemo_retriever.harness.config import DEFAULT_NIGHTLY_CONFIG_PATH, load_nightly_config
from nemo_retriever.harness.run import normalize_tags, execute_runs
from nemo_retriever.harness.slack import load_replay_report, load_session_report, post_report_to_slack


def _maybe_post_to_slack(
    *,
    report_path: Path | None,
    replay_paths: list[Path] | None,
    slack_config: dict[str, object],
    skip_slack: bool,
) -> bool:
    if skip_slack:
        typer.echo("Slack posting skipped (--skip-slack).")
        return False
    if not bool(slack_config.get("enabled", True)):
        typer.echo("Slack posting disabled in nightly config.")
        return False

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        typer.echo("Slack posting enabled but SLACK_WEBHOOK_URL is not set; skipping post.")
        return False

    report = load_replay_report(replay_paths or []) if replay_paths else load_session_report(report_path or Path("."))
    post_report_to_slack(report, slack_config, webhook_url=webhook_url)
    typer.echo(f"Posted Slack summary for session `{report.session_name}`.")
    return True


def nightly_command(
    config: str | None = typer.Option(None, "--config", help="Path to harness test config YAML."),
    runs_config: str = typer.Option(
        str(DEFAULT_NIGHTLY_CONFIG_PATH), "--runs-config", help="Path to nightly runs YAML."
    ),
    preset: str | None = typer.Option(None, "--preset", help="Force preset for all nightly runs."),
    tag: list[str] = typer.Option([], "--tag", help="Session tag to persist on each run. Repeatable."),
    skip_slack: bool = typer.Option(False, "--skip-slack", help="Skip Slack posting after the run completes."),
    replay: list[Path] = typer.Option(
        [],
        "--replay",
        help=(
            "Replay a previous session directory, run directory, or results.json file to Slack. "
            "Repeatable for run dirs."
        ),
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print nightly run plan without executing."),
) -> None:
    normalized_tags = normalize_tags(tag)
    nightly_cfg = load_nightly_config(runs_config)
    runs = nightly_cfg["runs"]
    slack_config = nightly_cfg["slack"]
    resolved_preset = preset or nightly_cfg.get("preset")

    if replay:
        _maybe_post_to_slack(
            report_path=None,
            replay_paths=list(replay),
            slack_config=slack_config,
            skip_slack=skip_slack,
        )
        raise typer.Exit(code=0)

    if dry_run:
        typer.echo("Nightly dry run:")
        for idx, run in enumerate(runs):
            tag_text = f" tags={normalized_tags}" if normalized_tags else ""
            run_preset = run.get("preset") if run.get("preset") is not None else resolved_preset
            plan_line = (
                f"  {idx + 1:03d}: name={run.get('name')} "
                f"dataset={run.get('dataset')} preset={run_preset}{tag_text}"
            )
            typer.echo(plan_line)
        typer.echo(
            "  slack: "
            f"enabled={bool(slack_config.get('enabled', True))} "
            f"title={slack_config.get('title')} "
            f"post_artifact_paths={bool(slack_config.get('post_artifact_paths', True))}"
        )
        raise typer.Exit(code=0)

    session_dir, run_results = execute_runs(
        runs=runs,
        config_file=config,
        session_prefix="nightly",
        preset_override=resolved_preset,
        tags=normalized_tags,
    )
    summary_path = write_session_summary(
        session_dir,
        run_results,
        session_type="nightly",
        config_path=str(Path(runs_config).expanduser().resolve()),
    )

    typer.echo(f"\nNightly session: {session_dir}")
    typer.echo(f"Session summary: {summary_path}")
    slack_failed = False
    try:
        _maybe_post_to_slack(
            report_path=summary_path,
            replay_paths=None,
            slack_config=slack_config,
            skip_slack=skip_slack,
        )
    except RuntimeError as exc:
        typer.echo(f"Slack post failed: {exc}")
        slack_failed = True
    failed = [r for r in run_results if not r["success"]]
    raise typer.Exit(code=0 if not failed and not slack_failed else 1)
