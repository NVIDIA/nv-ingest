#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Run preparation tasks here
# Helper: truthy check (1/true/on/yes, case-insensitive)
is_truthy() {
  [ -n "$1" ] || return 1
  case "$(printf "%s" "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|on|yes) return 0 ;;
    *) return 1 ;;
  esac
}

# Ensure micromamba is available if we need to install conda packages
need_micromamba=0

if is_truthy "${INSTALL_ADOBE_SDK}"; then
  echo "Checking if Adobe PDF Services SDK is installed..."

  # Check if pdfservices-sdk is installed
  if ! python -c "import pkg_resources; pkg_resources.require('pdfservices-sdk~=4.0.0')" 2>/dev/null; then
    echo "Installing Adobe PDF Services SDK..."
    pip install "pdfservices-sdk~=4.0.0"
  fi
fi

# Install audio dependencies
if ! python -c "import pkg_resources; pkg_resources.require('librosa')" 2>/dev/null; then
  echo "Installing librosa using conda..."
  need_micromamba=1
  micromamba install -y -n nv_ingest_runtime -c conda-forge librosa
fi

# If MEM_TRACE is set in the environment, use mamba to install memray
if is_truthy "${MEM_TRACE}" || is_truthy "${INGEST_MEM_TRACE}"; then
  echo "MEM_TRACE is set. Installing memray via mamba..."
  need_micromamba=1
  micromamba install -y -n nv_ingest_runtime -c conda-forge memray || {
    echo "Fallback: installing memray via pip..."
    pip install memray
  }
fi
