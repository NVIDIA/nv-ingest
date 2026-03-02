# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from . import commands

app = typer.Typer(help="Chart Extraction")
app.add_typer(commands.app, name="stage")


def main() -> None:
    app()
