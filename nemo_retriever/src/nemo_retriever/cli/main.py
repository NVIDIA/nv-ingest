# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import logging

import typer

from nemo_retriever.cli.ingest import app as ingest_app
from nemo_retriever.cli.query import app as query_app
from nemo_retriever.version import get_version_info

logger = logging.getLogger(__name__)

app = typer.Typer(help="NeMo Retriever product workflows: ingest content, query an index, or operate the service.")

# Service sub-app is always available (lightweight, no GPU deps).
from nemo_retriever.service.cli import app as service_app  # noqa: E402

app.add_typer(service_app, name="service")
app.add_typer(ingest_app, name="ingest")
app.add_typer(query_app, name="query")

# Keep compatibility commands callable while hiding them from the product help
# surface. HTML and TXT are intentionally absent: they are ingest input formats,
# not standalone workflows.
_LAZY_SUBAPPS: list[tuple[str, str, str, bool]] = [
    ("eval", "nemo_retriever.tools.evaluation.cli", "app", True),
    ("benchmark", "nemo_retriever.tools.benchmark", "app", True),
    ("recall", "nemo_retriever.tools.recall", "app", True),
    ("skill-eval", "nemo_retriever.tools.skill_eval", "app", True),
]

for _name, _module, _attr, _hidden in _LAZY_SUBAPPS:
    try:
        _mod = importlib.import_module(_module)
        app.add_typer(getattr(_mod, _attr), name=_name, hidden=_hidden)
    except Exception:
        logger.debug("Skipping '%s' sub-command (import failed)", _name)


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
