# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Writer strategies package for NV-Ingest data destinations.

This package contains strategy implementations for writing data to various
destinations including Redis, filesystem, HTTP, and Kafka.
"""

from .redis import RedisWriterStrategy
from .filesystem import FilesystemWriterStrategy
from .http import HttpWriterStrategy
from .kafka import KafkaWriterStrategy

# Strategy registry
WRITER_STRATEGIES = {
    "redis": RedisWriterStrategy(),
    "filesystem": FilesystemWriterStrategy(),
    "http": HttpWriterStrategy(),  # Now properly instantiates with session pooling
    "kafka": KafkaWriterStrategy(),
}


def get_writer_strategy(destination_type: str):
    """
    Get the writer strategy for a destination type.

    Parameters
    ----------
    destination_type : str
        The destination type (e.g., "redis", "filesystem", "http", "kafka")

    Returns
    -------
    WriterStrategy
        The appropriate writer strategy

    Raises
    ------
    ValueError
        If the destination type is not supported
    """
    if destination_type not in WRITER_STRATEGIES:
        supported = list(WRITER_STRATEGIES.keys())
        raise ValueError(f"Unsupported destination type: {destination_type}. Supported: {supported}")

    return WRITER_STRATEGIES[destination_type]


__all__ = [
    "RedisWriterStrategy",
    "FilesystemWriterStrategy",
    "HttpWriterStrategy",
    "KafkaWriterStrategy",
    "get_writer_strategy",
]
