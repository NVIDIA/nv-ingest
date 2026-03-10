# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from nemo_retriever.harness.nightly import nightly_command
from nemo_retriever.harness.reporting import compare_command, summary_command
from nemo_retriever.harness.run import run_command, sweep_command

app = typer.Typer(help="Harness commands for benchmark orchestration.")
app.command("run")(run_command)
app.command("sweep")(sweep_command)
app.command("nightly")(nightly_command)
app.command("summary")(summary_command)
app.command("compare")(compare_command)


def main() -> None:
    app()
