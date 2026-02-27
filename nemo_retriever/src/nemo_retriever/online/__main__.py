# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.adapters.service.cli import app, main, serve_cmd, stream_pdf_cmd, submit_cmd

__all__ = ["app", "main", "serve_cmd", "stream_pdf_cmd", "submit_cmd"]
