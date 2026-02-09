"""
Batch runmode.

Intended for large-scale batch execution over large inputs on multiple workers.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, List, Optional

from typing import Union

import ray
import ray.data as rd

from ..ingest import Ingestor


class BatchIngestor(Ingestor):
    RUN_MODE = "batch"

    def __init__(self, documents: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(documents=documents, **kwargs)

        logging.basicConfig(level=logging.INFO)

        # Initialize Ray locally for distributed execution.
        ray.init(address="local", ignore_reinit_error=True) 

        # Builder-style task configuration recorded for later execution.
        self._input_documents: List[str] = [] # List of original input documents.
        self._rd_dataset: rd.Dataset = None # Ray Data dataset created from input documents.
        self._tasks: List[tuple[str, dict[str, Any]]] = []

    def files(self, documents: Union[str, List[str]]) -> "BatchIngestor":
        """
        Add local files for batch processing.

        This runmode assumes all inputs are local. Any glob pattern or explicit
        path must resolve to at least one existing file.
        """
        if isinstance(documents, str):
            documents = [documents]

        for pattern in documents:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"Invalid document pattern: {pattern!r}")

            # Expand globs (supports ** when recursive=True).
            matches = glob.glob(pattern, recursive=True)
            if matches:
                files = [os.path.abspath(p) for p in matches if os.path.isfile(p)]
                if not files:
                    raise FileNotFoundError(f"Pattern resolved, but no files found: {pattern!r}")
                self._input_documents.extend(files)
                continue

            # No glob matches: treat as explicit path.
            if os.path.isfile(pattern):
                self._input_documents.append(os.path.abspath(pattern))
                continue

            raise FileNotFoundError(f"No local files found for: {pattern!r}")

        self._rd_dataset = rd.read_binary_files(self._input_documents, include_paths=True)

        return self

    def extract(self, **kwargs: Any) -> "BatchIngestor":
        """
        Configure extraction for batch processing (builder only).

        This does not run extraction yet; it records configuration so the batch
        executor can build a concrete pipeline later.
        """
        print(f"kwargs: {kwargs}")
        self._tasks.append(("extract", dict(kwargs)))
        return self

