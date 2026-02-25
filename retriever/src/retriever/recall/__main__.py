# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from . import vdb_recall

app = typer.Typer(help="Recall utilities (query -> embed -> vector DB search).")
app.add_typer(vdb_recall.app, name="vdb-recall")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
