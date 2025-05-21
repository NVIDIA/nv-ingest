# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch

import nv_ingest_client.cli.util.system as module_under_test
import pytest
from nv_ingest_client.cli.util.system import configure_logging

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


# Testing configure_logging
@patch("logging.Logger.setLevel")
@patch("logging.basicConfig")
def test_configure_logging(mock_basicConfig, mock_setLevel):
    logger = logging.getLogger(__name__)
    configure_logging(logger, "DEBUG")
    mock_basicConfig.assert_called_with(level=logging.DEBUG)
    mock_setLevel.assert_called_with(logging.DEBUG)

    with pytest.raises(ValueError):
        configure_logging(logger, "INVALID")
