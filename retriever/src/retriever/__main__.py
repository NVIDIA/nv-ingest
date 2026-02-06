from __future__ import annotations

import typer

from .image import app as image_app

from .pdf import app as pdf_app
from .local import app as local_app
from .chart import app as chart_app
from .compare import app as compare_app
from .vector_store import app as vector_store_app
from .recall import app as recall_app

app = typer.Typer(help="Retriever")
app.add_typer(image_app, name="image")
app.add_typer(pdf_app, name="pdf")
app.add_typer(local_app, name="local")
app.add_typer(chart_app, name="chart")
app.add_typer(compare_app, name="compare")
app.add_typer(vector_store_app, name="vector-store")
app.add_typer(recall_app, name="recall")


def main():
    app()
