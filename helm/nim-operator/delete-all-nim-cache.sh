#!/bin/bash

# Accept input parameters for namespaces
NAMESPACE=${1:-nv-ingest}

echo "Deleting all NimCache in namespace: $NAMESPACE"

kubectl delete nimcache --all -n $NAMESPACE
