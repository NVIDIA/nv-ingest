#!/bin/bash

# Accept input parameters for namespaces
NAMESPACE=${1:-nv-ingest}

echo "Deleting all NimServices in namespace: $NAMESPACE"

kubectl delete nimservice --all -n $NAMESPACE
