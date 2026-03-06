# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level CLI entrypoint.

This package intentionally exposes only the ``evaluate`` command group.
"""

import logging
from typing import Annotated

import typer

from retrieval_bench.cli.evaluate import app as evaluate_app
from vidore_benchmark.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="CLI for retrieval pipeline benchmarking.",
    no_args_is_help=True,
)

app.add_typer(
    evaluate_app,
    name="evaluate",
    help="Evaluate retrieval pipelines on ViDoRe v3 / BRIGHT datasets",
)


@app.callback()
def main(log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning"):
    setup_logging(log_level)
    logger.info("Logging level set to `%s`", log_level)


if __name__ == "__main__":
    app()
