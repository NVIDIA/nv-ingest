# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Define paths
SPHINX_BUILD_DIR=sphinx_docs/build
SPHINX_SOURCE_DIR=sphinx_docs/source
SPHINX_OUTPUT_DIR=docs/user-guide/developer-guide/api-docs

# Default target
.PHONY: all
all: docs

# Run apidoc against nv-ingest, nv-ingest-api, and nv-ingest-client codebase
.PHONY: sphinx-apidoc
sphinx-apidoc:
	@echo "🏃 Running sphinx-apidoc against nv-ingest"
	sphinx-apidoc -o sphinx_docs/source/nv-ingest ../src/nv_ingest

	@echo "🏃 Running sphinx-apidoc against nv-ingest-api"
	sphinx-apidoc -o sphinx_docs/source/nv-ingest-api ../api/src/nv_ingest_api

	@echo "🏃 Running sphinx-apidoc against nv-ingest-client"
	sphinx-apidoc -o sphinx_docs/source/nv-ingest-client ../client/src/nv_ingest_client

# Build Sphinx documentation
.PHONY: sphinx
sphinx:
	@echo "📖 Building Sphinx documentation..."
	sphinx-build -b html $(SPHINX_SOURCE_DIR) $(SPHINX_BUILD_DIR)/html

# Copy Sphinx HTML output to the final location
.PHONY: copy-sphinx
copy-sphinx: sphinx
	@echo "📂 Copying Sphinx docs to $(SPHINX_OUTPUT_DIR)..."
	mkdir -p $(SPHINX_OUTPUT_DIR)
	cp -r $(SPHINX_BUILD_DIR)/html/* $(SPHINX_OUTPUT_DIR)/

# Run MkDocs to build final documentation
.PHONY: mkdocs
mkdocs:
	@echo "📚 Building MkDocs site..."
	mkdocs build

# Full docs pipeline
.PHONY: docs
docs: sphinx-apidoc sphinx copy-sphinx mkdocs
	@echo "✅ Documentation build complete!"

# Clean up built docs
.PHONY: clean
clean:
	@echo "🧹 Cleaning up built documentation..."
	rm -rf $(SPHINX_BUILD_DIR) $(SPHINX_OUTPUT_DIR) site
