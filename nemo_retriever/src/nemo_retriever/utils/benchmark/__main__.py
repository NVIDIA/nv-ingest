# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from . import all_actor, audio_extract_actor, extract_actor, ocr_actor, page_elements_actor, split_actor

app = typer.Typer(help="Actor-stage throughput benchmarks.")
app.add_typer(split_actor.app, name="split")
app.add_typer(extract_actor.app, name="extract")
app.add_typer(audio_extract_actor.app, name="audio-extract")
app.add_typer(page_elements_actor.app, name="page-elements")
app.add_typer(ocr_actor.app, name="ocr")
app.add_typer(all_actor.app, name="all")


def main() -> None:
    app()
