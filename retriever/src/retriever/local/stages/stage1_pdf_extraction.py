# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Local pipeline stage1: pass-through PDF extraction CLI.

This module intentionally contains **no configuration logic**. It simply re-exports the
`retriever.pdf.stage` Typer application so any arguments provided to:

  `retriever local stage1 ...`

are handled exactly the same as:

  `retriever pdf ...`
"""

from retriever.pdf.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
