# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Error definitions for the NV-Ingest data writer.

This module defines the exception hierarchy used by the IngestDataWriter
for classifying and handling different types of errors that can occur
during data writing operations.
"""


class DataWriterError(Exception):
    """Base exception for data writer errors."""

    pass


class TransientError(DataWriterError):
    """Errors that may succeed on retry (e.g., network timeouts, temporary server issues)."""

    pass


class PermanentError(DataWriterError):
    """Errors that will not succeed on retry (e.g., auth failures, config errors)."""

    pass


class ConnectionError(TransientError):
    """Connection-related transient errors (timeouts, unreachable hosts, DNS failures)."""

    pass


class AuthenticationError(PermanentError):
    """Authentication/authorization failures (invalid credentials, insufficient permissions)."""

    pass


class ConfigurationError(PermanentError):
    """Configuration-related errors (invalid settings, missing required parameters)."""

    pass


class DependencyError(ConfigurationError):
    """Error raised when required dependencies are not available."""

    pass
