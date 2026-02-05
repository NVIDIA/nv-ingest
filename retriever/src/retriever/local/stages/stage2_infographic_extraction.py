"""
Local pipeline stage: pass-through infographic extraction CLI.

This module intentionally contains no configuration logic. It re-exports the
`retriever.infographic.stage` Typer application so arguments provided to:

  `retriever local stage2-infographic ...`

are handled by `retriever.infographic.stage`.
"""

from __future__ import annotations

from retriever.infographic.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

