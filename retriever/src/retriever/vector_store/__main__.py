from __future__ import annotations

import typer

from . import stage

app = typer.Typer(help="Vector store utilities (LanceDB upload, etc).")
app.add_typer(stage.app, name="stage")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

