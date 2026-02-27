# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Local pipeline stage6: thin proxy for vector store upload CLI.

This module intentionally contains no configuration logic. It re-exports the
`nemo_retriever.vector_store.stage` Typer application so arguments provided to:

  `retriever local stage6 ...`

are handled by `nemo_retriever.vector_store.stage`.
"""

from nemo_retriever.vector_store.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
