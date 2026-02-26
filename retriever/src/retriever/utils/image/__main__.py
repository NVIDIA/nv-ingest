# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from . import render

app = typer.Typer(help="Utilities for working with images (visualization, inspection, conversions)")
app.add_typer(render.app, name="render")


def main() -> None:
    app()
