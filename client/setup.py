# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from setuptools import find_packages
from setuptools import setup


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "src", "version.py")
    with open(version_file) as f:
        # Execute the content of version.py
        exec(f.read())
    return locals()["get_version"]()


setup(
    author="Jeremy Dyer",
    author_email="jdyer@nvidia.com",
    description="Python client for the nv-ingest service",
    license="Apache-2.0",
    name="nv-ingest-client",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    version=get_version(),
)
