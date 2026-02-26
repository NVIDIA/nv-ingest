# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


def find_stage_inputs(input_dir: Path, *, suffix: str) -> list[Path]:
    files = [path for path in input_dir.iterdir() if path.is_file() and path.name.endswith(suffix)]
    return sorted(files)


def build_stage_output_path(input_path: Path, *, stage_suffix: str, output_dir: Path | None) -> Path:
    output_name = input_path.stem + stage_suffix + input_path.suffix
    if output_dir is None:
        return input_path.with_name(output_name)
    return output_dir / output_name
