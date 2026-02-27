# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for `python -m nemo_retriever.audio`. Do not import this from the package __init__."""

from __future__ import annotations

from .cli import app

if __name__ == "__main__":
    app()
