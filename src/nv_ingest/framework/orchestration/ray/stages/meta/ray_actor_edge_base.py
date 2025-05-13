# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Dict


# TODO(Devin): Early prototype. Not currently used anywhere


class RayActorEdge(ABC):
    """
    Abstract base class for a Ray actor edge used in a RayPipeline.

    Parameters
    ----------
    max_size : int
        The maximum size of the edge's internal queue.
    multi_reader : bool
        Whether the edge supports multiple concurrent readers.
    multi_writer : bool
        Whether the edge supports multiple concurrent writers.
    """

    def __init__(self, max_size: int, multi_reader: bool = False, multi_writer: bool = False) -> None:
        self.max_size = max_size
        self.multi_reader = multi_reader
        self.multi_writer = multi_writer

    @abstractmethod
    def write(self, item: Any) -> bool:
        """
        Write an item into the edge.

        Parameters
        ----------
        item : Any
            The item to enqueue.

        Returns
        -------
        bool
            True if the item was enqueued successfully.
        """
        pass

    @abstractmethod
    def read(self) -> Any:
        """
        Read an item from the edge.

        Returns
        -------
        Any
            The next item in the edge.
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """
        Get current statistics for the edge.

        Returns
        -------
        Dict[str, int]
            A dictionary containing statistics (e.g. write_count, read_count, queue_full_count, current_size).
        """
        pass
