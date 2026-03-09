# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import typer

from nemo_retriever.harness.artifacts import write_session_summary
from nemo_retriever.harness.config import DEFAULT_NIGHTLY_CONFIG_PATH, load_runs_config
from nemo_retriever.harness.run import _normalize_tags, execute_runs


def nightly_command(
    config: str | None = typer.Option(None, "--config", help="Path to harness test config YAML."),
    runs_config: str = typer.Option(
        str(DEFAULT_NIGHTLY_CONFIG_PATH), "--runs-config", help="Path to nightly runs YAML."
    ),
    preset: str | None = typer.Option(None, "--preset", help="Force preset for all nightly runs."),
    tag: list[str] = typer.Option([], "--tag", help="Session tag to persist on each run. Repeatable."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print nightly run plan without executing."),
) -> None:
    normalized_tags = _normalize_tags(tag)
    runs = load_runs_config(runs_config)
    if dry_run:
        typer.echo("Nightly dry run:")
        for idx, run in enumerate(runs):
            tag_text = f" tags={normalized_tags}" if normalized_tags else ""
            plan_line = (
                f"  {idx + 1:03d}: name={run.get('name')} "
                f"dataset={run.get('dataset')} preset={run.get('preset')}{tag_text}"
            )
            typer.echo(plan_line)
        raise typer.Exit(code=0)

    session_dir, run_results = execute_runs(
        runs=runs,
        config_file=config,
        session_prefix="nightly",
        preset_override=preset,
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
    failed = [r for r in run_results if not r["success"]]
    raise typer.Exit(code=0 if not failed else 1)
