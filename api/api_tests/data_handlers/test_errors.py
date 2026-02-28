# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nv_ingest_api.data_handlers import errors as err


class TestErrorsModule:
    """Black-box tests for error class hierarchy and behavior."""

    def test_error_hierarchy_is_correct(self):
        """Derived exceptions should be instances of their parents and base class."""
        base = err.DataWriterError("base")
        transient = err.TransientError("transient")
        permanent = err.PermanentError("permanent")
        conn = err.ConnectionError("conn")
        auth = err.AuthenticationError("auth")
        cfg = err.ConfigurationError("cfg")
        dep = err.DependencyError("dep")

        # Base types
        assert isinstance(base, Exception)
        assert isinstance(transient, err.DataWriterError)
        assert isinstance(permanent, err.DataWriterError)
        assert isinstance(conn, err.TransientError)
        assert isinstance(auth, err.PermanentError)
        assert isinstance(cfg, err.PermanentError)
        assert isinstance(dep, err.ConfigurationError)

    def test_catching_specific_then_general(self):
        """Catching should work from specific to general without leaking exceptions."""
        # Specific catch
        try:
            raise err.ConnectionError("network down")
        except err.ConnectionError as e:
            assert "network down" in str(e)
        except Exception:  # pragma: no cover - would indicate catch order error
            pytest.fail("Caught by wrong handler")

        # General catch covers derived
        try:
            raise err.AuthenticationError("denied")
        except err.PermanentError as e:
            assert "denied" in str(e)
        else:  # pragma: no cover
            pytest.fail("PermanentError did not catch AuthenticationError")

    def test_dependency_error_is_configuration_error(self):
        """DependencyError should be a ConfigurationError and a DataWriterError."""
        exc = err.DependencyError("missing lib")
        assert isinstance(exc, err.ConfigurationError)
        assert isinstance(exc, err.DataWriterError)
        assert isinstance(exc, Exception)
