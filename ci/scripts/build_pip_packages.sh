#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --type <dev|release> --lib <client|service>"
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

# Set library-specific variables and paths
if [[ "$LIBRARY" == "client" ]]; then
    NV_INGEST_CLIENT_VERSION_OVERRIDE="${VERSION_SUFFIX}"
    export NV_INGEST_CLIENT_VERSION_OVERRIDE
    SETUP_PATH="$SCRIPT_DIR/../../client/setup.py"
elif [[ "$LIBRARY" == "service" ]]; then
    NV_INGEST_SERVICE_VERSION_OVERRIDE="${VERSION_SUFFIX}"
    export NV_INGEST_SERVICE_VERSION_OVERRIDE
    SETUP_PATH="$SCRIPT_DIR/../../setup.py"
else
    echo "Invalid library: $LIBRARY"
    usage
fi

# Build the wheel
(cd "$(dirname "$SETUP_PATH")" && python setup.py sdist bdist_wheel)
