# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Filesystem writer strategy for the NV-Ingest data writer.
"""

import json
from typing import List

# Type imports (will be resolved at runtime)
try:
    from nv_ingest_api.data_handlers.data_writer import FilesystemDestinationConfig
except ImportError:
    # Handle circular import during development
    FilesystemDestinationConfig = None


class FilesystemWriterStrategy:
    """Strategy for writing to filesystem destinations."""

    def is_available(self) -> bool:
        """Check if fsspec is available."""
        try:
            import fsspec

            return fsspec is not None
        except ImportError:
            return False

    def write(self, data_payload: List[str], config: FilesystemDestinationConfig) -> None:
        """Write payloads to filesystem using fsspec."""
        if not self.is_available():
            from nv_ingest_api.data_handlers.errors import DependencyError

            raise DependencyError(
                "fsspec library is not available. Install fsspec for filesystem destination support: "
                "pip install fsspec"
            )

        import fsspec

        # Combine all fragments into a single JSON array
        combined_data = [json.loads(payload) for payload in data_payload]

        with fsspec.open(config.path, "w") as f:
            json.dump(combined_data, f)
