from __future__ import annotations

import typer

from .image import app as image_app

from .pdf import app as pdf_app
from .local import app as local_app
from .chart import app as chart_app
from .compare import app as compare_app
from .benchmark import app as benchmark_app
from .vector_store import app as vector_store_app
from .recall import app as recall_app
from .txt import __main__ as txt_main
from .online import __main__ as online_main
from .version import get_version_info

app = typer.Typer(help="Retriever")
app.add_typer(image_app, name="image")
app.add_typer(pdf_app, name="pdf")
app.add_typer(local_app, name="local")
app.add_typer(chart_app, name="chart")
app.add_typer(compare_app, name="compare")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(vector_store_app, name="vector-store")
app.add_typer(recall_app, name="recall")
app.add_typer(txt_main.app, name="txt")
app.add_typer(online_main.app, name="online")


def _version_callback(value: bool) -> None:
    if not value:
        return
    info = get_version_info()
    typer.echo(info["full_version"])
    raise typer.Exit()


def main():
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
