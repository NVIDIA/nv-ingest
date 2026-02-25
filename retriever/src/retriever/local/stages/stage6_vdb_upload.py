from __future__ import annotations

"""
Local pipeline stage6: thin proxy for vector store upload CLI.

This module intentionally contains no configuration logic. It re-exports the
`retriever.vector_store.stage` Typer application so arguments provided to:

  `retriever local stage6 ...`

are handled by `retriever.vector_store.stage`.
"""

from retriever.vector_store.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
