#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --type <dev|release> --lib <api|client|service>"
    exit 1
}

# Get the directory of the current script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Parse options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type) TYPE="$2"; shift ;;
        --lib) LIBRARY="$2"; shift  ;;
        *) usage ;;
    esac
    shift
done

# Validate input
if [[ -z "$TYPE" || -z "$LIBRARY" ]]; then
    usage
fi

# Get current date
DATE=$(date +'%Y.%m.%d')

# Set the version based on the build type
if [[ "$TYPE" == "dev" ]]; then
    VERSION_SUFFIX="${DATE}-dev"
elif [[ "$TYPE" == "release" ]]; then
    VERSION_SUFFIX="${DATE}"
else
    echo "Invalid type: $TYPE"
    usage
fi

NV_INGEST_VERSION_OVERRIDE="${VERSION_SUFFIX}"
export NV_INGEST_VERSION_OVERRIDE

# Set library-specific variables and paths
if [[ "$LIBRARY" == "api" ]]; then
    SETUP_PATH="$SCRIPT_DIR/../../api/pyproject.toml"
    (cd "$(dirname "$SETUP_PATH")" && python -m build)
elif [[ "$LIBRARY" == "client" ]]; then
    SETUP_PATH="$SCRIPT_DIR/../../client/pyproject.toml"
    (cd "$(dirname "$SETUP_PATH")" && python -m build)
elif [[ "$LIBRARY" == "service" ]]; then
    SETUP_PATH="$SCRIPT_DIR/../../src/pyproject.toml"
    (cd "$(dirname "$SETUP_PATH")" && python -m build)
else
    echo "Invalid library: $LIBRARY"
    usage
fi
