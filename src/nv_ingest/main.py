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
import datetime
import os
import re

from .api.main import app as app_v1


# TODO: Lets move this to a common utility, this is also used in both client and runtime setup.py files ...
def get_version():
    release_type = os.getenv("NV_INGEST_RELEASE_TYPE", "dev")
    version = os.getenv("NV_INGEST_VERSION")
    rev = os.getenv("NV_INGEST_REV", "0")

    if not version:
        version = f"{datetime.datetime.now().strftime('%Y.%m.%d')}"

    # Ensure the version is PEP 440 compatible
    pep440_regex = r"^\d{4}\.\d{1,2}\.\d{1,2}$"
    if not re.match(pep440_regex, version):
        raise ValueError(f"Version '{version}' is not PEP 440 compatible")

    # Construct the final version string
    if release_type == "dev":
        final_version = f"{version}.dev{rev}"
    elif release_type == "release":
        final_version = f"{version}.post{rev}" if int(rev) > 0 else version
    else:
        raise ValueError(f"Invalid release type: {release_type}")

    return final_version


app = FastAPI(
    title="NV-Ingest Microservice",
    description="Service for ingesting heterogenous datatypes",
    version=get_version(),
    contact={
        "name": "NVIDIA Corporation",
        "url": "https://nvidia.com",
    },
    openapi_tags=[
        {"name": "Health", "description": "Health checks"},
    ],
)


app.mount("/v1", app_v1)
