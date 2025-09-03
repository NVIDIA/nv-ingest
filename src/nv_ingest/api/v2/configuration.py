# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: skip-file
import logging
from typing import Dict, Any
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Path, Body, status
from pydantic import BaseModel, Field
from typing import Optional, Any
from fastapi import Path, Body, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

# Simulated in-memory configuration store
CONFIG_STORE: Dict[str, Any] = {
    "max_upload_size_mb": 100,
    "enable_tracing": True,
    "default_timeout_sec": 60,
    "service_name": "nv-ingest",
    "log_level": "INFO",
}


class ConfigResponse(BaseModel):
    config: Dict[str, Any] = Field(..., description="Key/value pairs of configuration settings.")


class ConfigKeyValue(BaseModel):
    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="Configuration value")


@router.get(
    "/config",
    response_model=ConfigResponse,
    tags=["Configuration"],
    summary="Get all configuration key/value pairs",
    status_code=status.HTTP_200_OK,
)
async def get_all_config():
    """
    Returns all configuration key/value pairs.
    """
    return ConfigResponse(config=CONFIG_STORE)


@router.get(
    "/config/{key}",
    response_model=ConfigKeyValue,
    tags=["Configuration"],
    summary="Get a specific configuration value by key",
    status_code=status.HTTP_200_OK,
)
async def get_config_key(key: str = Path(..., description="Configuration key to retrieve")):
    """
    Returns the value for a specific configuration key.
    """
    if key not in CONFIG_STORE:
        raise HTTPException(status_code=404, detail=f"Configuration key '{key}' not found.")
    return ConfigKeyValue(key=key, value=CONFIG_STORE[key])


@router.post(
    "/config",
    response_model=ConfigKeyValue,
    tags=["Configuration"],
    summary="Create a new configuration key/value pair",
    status_code=status.HTTP_201_CREATED,
)
async def create_config(config: ConfigKeyValue = Body(..., description="Key/value pair to create")):
    """
    Creates a new configuration key/value pair.
    """
    if config.key in CONFIG_STORE:
        raise HTTPException(status_code=409, detail=f"Configuration key '{config.key}' already exists.")
    CONFIG_STORE[config.key] = config.value
    return config


@router.put(
    "/config/{key}",
    response_model=ConfigKeyValue,
    tags=["Configuration"],
    summary="Update an existing configuration value",
    status_code=status.HTTP_200_OK,
)
async def update_config(
    key: str = Path(..., description="Configuration key to update"),
    value: Any = Body(..., embed=True, description="New value for the configuration key"),
):
    """
    Updates the value for an existing configuration key.
    """
    if key not in CONFIG_STORE:
        raise HTTPException(status_code=404, detail=f"Configuration key '{key}' not found.")
    CONFIG_STORE[key] = value
    return ConfigKeyValue(key=key, value=value)


@router.delete(
    "/config/{key}",
    response_model=ConfigKeyValue,
    tags=["Configuration"],
    summary="Delete a configuration key/value pair",
    status_code=status.HTTP_200_OK,
)
async def delete_config(key: str = Path(..., description="Configuration key to delete")):
    """
    Deletes a configuration key/value pair.
    """
    if key not in CONFIG_STORE:
        raise HTTPException(status_code=404, detail=f"Configuration key '{key}' not found.")
    value = CONFIG_STORE.pop(key)
    return ConfigKeyValue(key=key, value=value)
