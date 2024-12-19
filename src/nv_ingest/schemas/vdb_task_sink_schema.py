# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import typing

import pymilvus
from pydantic import field_validator, ConfigDict, BaseModel
from pydantic import Field
from typing_extensions import Annotated

logger = logging.getLogger(__name__)


def build_default_milvus_config(embedding_size: int = 1024) -> typing.Dict[str, typing.Any]:
    """
    Builds the configuration for Milvus.

    This function creates a dictionary configuration for a Milvus collection.
    It includes the index configuration and the schema configuration, with
    various fields like id, title, link, summary, page_content, and embedding.

    Parameters
    ----------
    embedding_size : int
        The size of the embedding vector.

    Returns
    -------
    typing.Dict[str, Any]
        A dictionary containing the configuration settings for Milvus.
    """

    milvus_resource_kwargs = {
        "index_conf": {
            "field_name": "vector",
            "metric_type": "L2",
            "index_type": "GPU_CAGRA",
            "params": {
                "intermediate_graph_degree": 128,
                "graph_degree": 64,
                "build_algo": "NN_DESCENT",
            },
        },
        "schema_conf": {
            "enable_dynamic_field": True,
            "schema_fields": [
                pymilvus.FieldSchema(
                    name="pk",
                    dtype=pymilvus.DataType.INT64,
                    description="Primary key for the collection",
                    is_primary=True,
                    auto_id=True,
                ).to_dict(),
                pymilvus.FieldSchema(
                    name="text", dtype=pymilvus.DataType.VARCHAR, description="Extracted content", max_length=65_535
                ).to_dict(),
                pymilvus.FieldSchema(
                    name="vector",
                    dtype=pymilvus.DataType.FLOAT_VECTOR,
                    description="Embedding vectors",
                    dim=embedding_size,
                ).to_dict(),
                pymilvus.FieldSchema(
                    name="source",
                    dtype=pymilvus.DataType.JSON,
                    description="Source document and raw data extracted content",
                ).to_dict(),
                pymilvus.FieldSchema(
                    name="content_metadata",
                    dtype=pymilvus.DataType.JSON,
                    description="Content metadata",
                ).to_dict(),
            ],
            "description": "NV-INGEST collection schema",
        },
    }

    return milvus_resource_kwargs


class VdbTaskSinkSchema(BaseModel):
    recreate: bool = False
    service: str = "milvus"
    is_service_serialized: bool = False
    default_resource_name: str = "nv_ingest_collection"
    resource_schemas: dict = {default_resource_name: build_default_milvus_config()}
    resource_kwargs: dict = Field(default_factory=dict)
    service_kwargs: dict = {}
    batch_size: int = 5120
    write_time_interval: float = 1.0
    retry_interval: float = 60.0
    raise_on_failure: bool = False
    progress_engines: Annotated[int, Field(ge=1)] = 1

    @field_validator("service", mode="before")
    @classmethod
    def validate_service(cls, to_validate):  # pylint: disable=no-self-argument
        if not to_validate:
            raise ValueError("Service must be a service name or a serialized instance of VectorDBService")
        return to_validate

    @field_validator("default_resource_name", mode="before")
    @classmethod
    def validate_resource_name(cls, to_validate):  # pylint: disable=no-self-argument
        if not to_validate:
            raise ValueError("Resource name must not be None or Empty.")
        return to_validate

    model_config = ConfigDict(extra="forbid")
