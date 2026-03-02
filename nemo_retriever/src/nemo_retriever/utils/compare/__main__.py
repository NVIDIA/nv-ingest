# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from . import compare_json
from . import compare_results

app = typer.Typer(help="Comparison utilities")
app.add_typer(compare_json.app, name="json")
app.add_typer(compare_results.app, name="results")


def main() -> None:
    app()
