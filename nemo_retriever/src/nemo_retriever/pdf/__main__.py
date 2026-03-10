# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from . import stage

app = typer.Typer(help="PDF Extraction")
app.add_typer(stage.app, name="stage")


def main() -> None:
    app()
