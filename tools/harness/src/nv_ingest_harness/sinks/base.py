# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for Sinks."""

from abc import ABC, abstractmethod
from typing import Any


class Sink(ABC):
    """Abstract base class for benchmark result sinks."""

    @abstractmethod
    def __init__(self, sink_config: dict[str, Any]): ...

    @abstractmethod
    def initialize(self, session_name: str, env_data: dict[str, Any]) -> None: ...

    @abstractmethod
    def process_result(
        self,
        result: dict[str, Any],
        entry_config: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    def finalize(self) -> None: ...
