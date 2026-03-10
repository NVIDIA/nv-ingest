# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat


def has_permissions(path: str, read: bool = False, write: bool = False) -> bool:
    """
    Checks if the current user has specified permissions on a path.

    Parameters
    ----------
    path : str
        The filesystem path to check permissions on.
    read : bool, optional
        Whether to check for read permission.
    write : bool, optional
        Whether to check for write permission.

    Returns
    -------
    bool
        True if the path has the specified permissions, False otherwise.
    """
    if not os.path.exists(path):
        return False

    current_permissions = os.stat(path).st_mode
    has_read = not read or bool(current_permissions & stat.S_IRUSR)
    has_write = not write or bool(current_permissions & stat.S_IWUSR)

    return has_read and has_write


def ensure_directory_with_permissions(directory_path: str):
    """
    Ensures that a directory exists and the current user has read/write permissions.
    If the directory does not exist, attempts to create it after checking the parent directory for write permission.

    Parameters
    ----------
    directory_path : str
        The path to the directory to check or create.

    Returns
    -------
    bool
        True if the directory exists and has the correct permissions, or if it was successfully created.
        False if the directory cannot be created or does not have the correct permissions.
    """
    if directory_path is None:
        return

    try:
        if not os.path.exists(directory_path):
            parent_directory = os.path.dirname(directory_path)
            if not has_permissions(parent_directory, write=True):
                raise OSError(f"Parent directory {parent_directory} does not have write permissions")

            os.makedirs(directory_path)

        if not has_permissions(directory_path, read=True, write=True):
            raise OSError(f"Directory {directory_path} does not have read/write permissions")
    except OSError as err:
        raise OSError(f"Error checking or creating directory: {err}")
