from __future__ import annotations

"""
Local pipeline stage7: thin proxy for vector DB query/recall CLI.

This module intentionally contains no configuration logic. It re-exports the
`retriever.recall.vdb_recall` Typer application so arguments provided to:

  `retriever local stage7 ...`

are handled by `retriever.recall.vdb_recall`, including a local-HF fallback path
when embedding HTTP/gRPC endpoints are not configured.
"""

from retriever.recall.vdb_recall import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()

