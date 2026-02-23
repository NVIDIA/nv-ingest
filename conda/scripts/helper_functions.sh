#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fail on errors and undefined variables
set -euo pipefail

validate_conda_build_environment() {
    ##############################
    # Validate Dependencies
    ##############################

    # Ensure conda is installed
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found in PATH. Please ensure Conda is installed and available."
        exit 1
    fi

    # Ensure conda-build is installed
    if ! command -v conda-build &> /dev/null; then
        echo "Error: conda-build not found in PATH. Install it via: conda install conda-build"
        exit 1
    fi

    # Ensure git is installed
    if ! command -v git &> /dev/null; then
        echo "Error: git not found in PATH. Please ensure Git is installed and available."
        exit 1
    fi
}

get_package_version() {
    local version_script=$1
    python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('version', '$version_script')
version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version)
print(version.get_version())
"
}

determine_git_root() {
    ##############################
    # Determine Git Root
    ##############################

    if git rev-parse --is-inside-work-tree &> /dev/null; then
        echo "$(git rev-parse --show-toplevel)"
    else
        echo "Error: Not inside a Git repository. Unable to determine the Git root."
        exit 1
    fi
}
