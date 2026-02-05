"""
Local pipeline stage: pass-through chart extraction CLI.

This module intentionally contains no configuration logic. It re-exports the
`retriever.chart.stage` Typer application so arguments provided to:

  `retriever local stage4 ...`

are handled by `retriever.chart.stage`.
"""

from __future__ import annotations

from retriever.chart.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

