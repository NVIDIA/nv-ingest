# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile
import shutil
from typing import List, Tuple, Optional


def get_git_root(file_path):
    """
    Attempts to get the root of the git repository for the given file.

    Parameters:
        file_path (str): The path of the file to determine the git root for.

    Returns:
        str: The absolute path to the git repository root if found,
             otherwise None.
    """
    try:
        # Get the absolute directory of the file.
        directory = os.path.dirname(os.path.abspath(file_path))
        # Run the git command to get the repository's top-level directory.
        git_root = (
            subprocess.check_output(["git", "-C", directory, "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT)
            .strip()
            .decode("utf-8")
        )
        return git_root
    except subprocess.CalledProcessError:
        # In case the file is not inside a git repository.
        return None
    except Exception:
        return None


def find_root_by_pattern(pattern, start_dir=None):
    """
    Backtracks up the directory tree looking for the first directory
    where the specified pattern exists.

    Parameters:
        pattern (str): The relative path to check for (e.g., "data/test.pdf").
        start_dir (str, optional): The starting directory. Defaults to the current working directory.

    Returns:
        str: The absolute path of the first directory where pattern exists,
             or "./" if not found.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    current_dir = os.path.abspath(start_dir)

    while True:
        candidate = os.path.join(current_dir, pattern)
        if os.path.exists(candidate):
            return current_dir

        # Determine the parent directory.
        parent_dir = os.path.dirname(current_dir)
        # If we're at the filesystem root, break.
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    return "./"


def create_test_workspace(prefix: str = "client_test_") -> str:
    """
    Create a temporary directory for test workspace.

    Parameters:
        prefix (str): Prefix for the temporary directory name.

    Returns:
        str: Path to the created temporary directory.
    """
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_test_workspace(workspace_path: str) -> None:
    """
    Clean up a test workspace directory.

    Parameters:
        workspace_path (str): Path to the workspace directory to clean up.
    """
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)


def create_test_file(directory: str, filename: str, content: str = None) -> str:
    """
    Create a test file with optional content.

    Parameters:
        directory (str): Directory where the file should be created.
        filename (str): Name of the file to create.
        content (str, optional): Content to write to the file. Defaults to a simple test message.

    Returns:
        str: Full path to the created file.
    """
    if content is None:
        content = f"This is a test file: {filename}"

    file_path = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)

    with open(file_path, "w") as f:
        f.write(content)

    return file_path


def create_test_documents(workspace: str, file_specs: List[Tuple[str, Optional[str]]] = None) -> List[str]:
    """
    Create multiple test documents in a workspace.

    Parameters:
        workspace (str): Path to the workspace directory.
        file_specs (List[Tuple[str, Optional[str]]], optional): List of (filename, content) tuples.
            If None, creates default test documents.

    Returns:
        List[str]: List of paths to created files.
    """
    if file_specs is None:
        file_specs = [
            ("test.pdf", "This is a test PDF document."),
            ("test.txt", "This is a test text document."),
            ("test.html", "<html><body>This is a test HTML document.</body></html>"),
            ("test.json", '{"message": "This is a test JSON document."}'),
            ("test.md", "# This is a test Markdown document"),
            ("test.sh", "#!/bin/bash\necho 'This is a test shell script'"),
        ]

    created_files = []
    for filename, content in file_specs:
        file_path = create_test_file(workspace, filename, content)
        created_files.append(file_path)

    return created_files


def create_jsonl_test_file(directory: str, filename: str, data_entries: List[dict]) -> str:
    """
    Create a JSONL test file with the given data entries.

    Parameters:
        directory (str): Directory where the file should be created.
        filename (str): Name of the JSONL file to create.
        data_entries (List[dict]): List of dictionaries to write as JSON lines.

    Returns:
        str: Full path to the created JSONL file.
    """
    import json

    file_path = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)

    with open(file_path, "w") as f:
        for entry in data_entries:
            f.write(json.dumps(entry) + "\n")

    return file_path


class TestWorkspace:
    """
    Context manager for creating and cleaning up test workspaces.

    Usage:
        with TestWorkspace() as workspace:
            # Use workspace.path for test operations
            test_file = workspace.create_file("test.txt", "content")
    """

    def __init__(self, prefix: str = "client_test_"):
        self.prefix = prefix
        self.path = None

    def __enter__(self):
        self.path = create_test_workspace(self.prefix)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path:
            cleanup_test_workspace(self.path)

    def create_file(self, filename: str, content: str = None) -> str:
        """Create a test file in this workspace."""
        if self.path is None:
            raise RuntimeError("Workspace not initialized. Use within 'with' statement.")
        return create_test_file(self.path, filename, content)

    def create_documents(self, file_specs: List[Tuple[str, Optional[str]]] = None) -> List[str]:
        """Create multiple test documents in this workspace."""
        if self.path is None:
            raise RuntimeError("Workspace not initialized. Use within 'with' statement.")
        return create_test_documents(self.path, file_specs)

    def create_jsonl_file(self, filename: str, data_entries: List[dict]) -> str:
        """Create a JSONL test file in this workspace."""
        if self.path is None:
            raise RuntimeError("Workspace not initialized. Use within 'with' statement.")
        return create_jsonl_test_file(self.path, filename, data_entries)
