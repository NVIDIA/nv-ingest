#!/bin/bash

# Create a temporary file for the helm-docs output
helm-docs --output-file=README.md.suffix

# Split the original README.md at "### Deployment parameters"
awk '/### Deployment parameters/{exit} {print}' README.md > README.md.prefix

# Combine the files
cat README.md.prefix README.md.suffix > README.md

# Clean up temporary files
rm README.md.prefix README.md.suffix
