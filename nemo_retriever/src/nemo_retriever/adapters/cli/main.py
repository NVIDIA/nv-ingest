# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from nemo_retriever.audio import app as audio_app
from nemo_nemo_retriever.utils.benchmark import app as benchmark_app
from nemo_nemo_retriever.chart import app as chart_app
from nemo_nemo_retriever.utils.compare import app as compare_app
from nemo_nemo_retriever.html import __main__ as html_main
from nemo_nemo_retriever.utils.image import app as image_app
from nemo_nemo_retriever.local import app as local_app
from nemo_nemo_retriever.online import __main__ as online_main
from nemo_nemo_retriever.pdf import app as pdf_app
from nemo_nemo_retriever.recall import app as recall_app
from nemo_nemo_retriever.txt import __main__ as txt_main
from nemo_nemo_retriever.vector_store import app as vector_store_app
from nemo_nemo_retriever.version import get_version_info

app = typer.Typer(help="Retriever")
app.add_typer(audio_app, name="audio")
app.add_typer(image_app, name="image")
app.add_typer(pdf_app, name="pdf")
app.add_typer(local_app, name="local")
app.add_typer(chart_app, name="chart")
app.add_typer(compare_app, name="compare")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(vector_store_app, name="vector-store")
app.add_typer(recall_app, name="recall")
app.add_typer(txt_main.app, name="txt")
app.add_typer(html_main.app, name="html")
app.add_typer(online_main.app, name="online")


def _version_callback(value: bool) -> None:
    if not value:
        return
    info = get_version_info()
    typer.echo(info["full_version"])
    raise typer.Exit()


def main() -> None:
    app()


@app.callback()
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show retriever version metadata and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    _ = version
