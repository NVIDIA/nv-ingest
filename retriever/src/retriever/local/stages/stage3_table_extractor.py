"""
Local pipeline stage: pass-through table extraction CLI.

This module intentionally contains no configuration logic. It re-exports the
`retriever.table.stage` Typer application so arguments provided to:

  `retriever local stage3-table ...`

are handled by `retriever.table.stage`.
"""

from __future__ import annotations

from retriever.table.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

