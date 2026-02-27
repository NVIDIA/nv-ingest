# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Checkpointable local pipeline stages.

Each stage is a standalone Typer CLI module intended to be run independently,
reading previous artifacts from disk and writing new artifacts alongside inputs.
"""

# Intentionally empty: stages are imported explicitly in `nemo_retriever.local.__main__`.
