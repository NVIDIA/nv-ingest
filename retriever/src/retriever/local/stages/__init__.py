"""
Checkpointable local pipeline stages.

Each stage is a standalone Typer CLI module intended to be run independently,
reading previous artifacts from disk and writing new artifacts alongside inputs.
"""

# Intentionally empty: stages are imported explicitly in `retriever.local.__main__`.
