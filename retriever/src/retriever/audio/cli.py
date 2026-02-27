# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio extraction CLI (chunk + ASR, write *.audio_extraction.json sidecars).

This module intentionally contains **no configuration logic**. It simply re-exports the
`retriever.audio.stage` Typer application so any arguments provided to:

  `retriever audio ...`

are handled exactly the same as the stage commands (e.g. `extract`, `discover`).
"""

from __future__ import annotations

from retriever.audio.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
