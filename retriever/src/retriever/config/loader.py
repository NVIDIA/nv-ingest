# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from retriever.ingest_config import (
    load_ingest_config_file,
    load_ingest_config_section,
    resolve_ingest_config_path,
)


def resolve_config_path(explicit: Optional[Path]) -> Tuple[Optional[Path], str]:
    return resolve_ingest_config_path(explicit)


def load_config_file(explicit: Optional[Path], *, verbose: bool = True) -> Tuple[Dict[str, Any], Optional[Path], str]:
    return load_ingest_config_file(explicit, verbose=verbose)


def load_config_section(
    explicit: Optional[Path],
    *,
    section: str,
    verbose: bool = True,
    warn_if_missing_section: bool = True,
) -> Dict[str, Any]:
    return load_ingest_config_section(
        explicit,
        section=section,
        verbose=verbose,
        warn_if_missing_section=warn_if_missing_section,
    )
