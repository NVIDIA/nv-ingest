from __future__ import annotations

import typer

from . import stage

app = typer.Typer(help="PDF Extraction")
app.add_typer(stage.app, name="stage")


def main() -> None:
    app()
