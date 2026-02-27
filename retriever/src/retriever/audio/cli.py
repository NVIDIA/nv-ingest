# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio CLI: Typer app for `retriever audio` subcommands.

Kept in a separate module so that `retriever.audio.__init__` does not import
`__main__`. Importing __main__ from the package causes runpy to see the module
already in sys.modules when running `python -m retriever.audio`, leading to a
RuntimeWarning and unreliable CLI behavior (e.g. no output in Docker).
"""

from __future__ import annotations

import typer

from . import stage

app = typer.Typer(help="Audio extraction (chunk + ASR).")
app.add_typer(stage.app, name="stage")


def main() -> None:
    app()
