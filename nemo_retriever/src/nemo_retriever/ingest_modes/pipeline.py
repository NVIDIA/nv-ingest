# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Ingest pipeline commands.")


@app.command()
def run(
    input_path: Path = typer.Option(
        ...,
        "--input-dir",
        exists=True,
        file_okay=True,
        dir_okay=True,
        path_type=Path,
        help="Input file or directory to ingest.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
        help="Optional ingest config file.",
    ),
) -> None:
    """
    Skeleton entrypoint for the ingest pipeline CLI.
    """
    typer.echo("Ingest pipeline runner is not implemented yet. " f"Received input={input_path} config={config}.")


def main() -> None:
    app()
