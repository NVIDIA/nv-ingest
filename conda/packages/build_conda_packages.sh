#!/bin/bash

# TODO(Devin) - Generalize so we can run from anywhere.

# Fail on errors
set -e

# Input Arguments
OUTPUT_DIR=${1:-"./output_conda_channel"}
NV_INGEST_DIR="./conda/packages/nv_ingest"
NV_INGEST_CLIENT_DIR="./conda/packages/nv_ingest_client"

# Validate OUTPUT_DIR
if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <output_directory>"
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR/linux-64"

# Build nv_ingest
echo "Building nv_ingest..."
conda build "$NV_INGEST_DIR" -c nvidia/label/dev -c rapidsai -c nvidia -c conda-forge -c pytorch --output-folder="$OUTPUT_DIR"

# Build nv_ingest_client
echo "Building nv_ingest_client..."
conda build "$NV_INGEST_CLIENT_DIR" --output-folder="$OUTPUT_DIR"

# Index the output directory
echo "Indexing conda channel..."
conda index "$OUTPUT_DIR"

echo "Artifacts successfully built and placed in $OUTPUT_DIR"
