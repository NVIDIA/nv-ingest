from __future__ import annotations

import typer

from . import compare_json

app = typer.Typer(help="Comparison utilities")
app.add_typer(compare_json.app, name="json")


def main() -> None:
    app()

