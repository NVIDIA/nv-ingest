# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import datetime
import os
import re


def get_version():
    release_type = os.getenv("NV_INGEST_RELEASE_TYPE", "dev")
    version = os.getenv("NV_INGEST_VERSION")
    rev = os.getenv("NV_INGEST_REV", "0")

    if not version:
        version = f"{datetime.datetime.now().strftime('%Y.%m.%d')}"

    # We only check this for dev, we assume for release the user knows what they are doing
    if release_type != "release":
        # Ensure the version is PEP 440 compatible
        pep440_regex = r"^\d{4}\.\d{1,2}\.\d{1,2}$"
        if not re.match(pep440_regex, version):
            raise ValueError(f"Version '{version}' is not PEP 440 compatible")

    # Construct the final version string
    if release_type == "dev":
        # If rev is not specified and defaults to 0 lets create a more meaningful development
        # identifier that is pep440 compliant
        if int(rev) == 0:
            rev = datetime.datetime.now().strftime("%Y%m%d")
        final_version = f"{version}.dev{rev}"
    elif release_type == "release":
        final_version = f"{version}.post{rev}" if int(rev) > 0 else version
    else:
        raise ValueError(f"Invalid release type: {release_type}")

    return final_version
