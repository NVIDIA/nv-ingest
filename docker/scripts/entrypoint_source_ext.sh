# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Run preparation tasks here

if [ "$INSTALL_ADOBE_SDK" = "true" ]; then
  echo "Checking if Adobe PDF Services SDK is installed..."

  # Check if pdfservices-sdk is installed
  if ! python -c "import pkg_resources; pkg_resources.require('pdfservices-sdk~=4.0.0')" 2>/dev/null; then
    echo "Installing Adobe PDF Services SDK..."
    pip install "pdfservices-sdk~=4.0.0"
  fi
fi

# If MEM_TRACE is set in the environment, use mamba to install memray
if [ -n "$MEM_TRACE" ]; then
  echo "MEM_TRACE is set. Installing memray via mamba..."
  mamba install -y conda-forge::memray
fi
