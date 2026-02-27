# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Local pipeline stage5: thin proxy for text embedding CLI.

This module intentionally contains no configuration logic. It re-exports the
`nemo_retriever.text_embed.stage` Typer application so arguments provided to:

  `retriever local stage5 ...`

are handled by `nemo_retriever.text_embed.stage`, including local-HF fallback options
when an embedding endpoint is not configured.
"""

from nemo_retriever.text_embed.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
