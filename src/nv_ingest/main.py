# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from fastapi import FastAPI

from .api.main import app as app_v1

app = FastAPI(
    title="NV-Ingest Microservice",
    description="Service for ingesting heterogenous datatypes",
    version="0.1.0",
    contact={
        "name": "NVIDIA Corporation",
        "url": "https://nvidia.com",
    },
    openapi_tags=[
        {"name": "Health", "description": "Health checks"},
    ],
)


app.mount("/v1", app_v1)
