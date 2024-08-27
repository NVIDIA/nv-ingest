# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import mrc
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def fetch_and_validate_module_config(builder: mrc.Builder, schema_class):
    """
    Validates the configuration of a module using a specified Pydantic schema class.

    Parameters
    ----------
    builder : object
        The builder object used to access the current module's configuration.
    schema_class : Pydantic BaseModel
        The schema class to be used for validating the module configuration.

    Raises
    ------
    ValueError
        If the module configuration fails validation according to the schema class.
    """
    module_config = builder.get_current_module_config()
    try:
        validated_config = schema_class(**module_config)
    except ValidationError as e:
        error_messages = "; ".join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
        log_error_message = f"Invalid configuration: {error_messages}"
        logger.error(log_error_message)
        raise ValueError(log_error_message)

    return validated_config
