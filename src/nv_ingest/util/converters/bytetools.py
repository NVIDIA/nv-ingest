# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64


def bytesfromhex(hex_input):
    """
    Function to convert hex to bytes.

    Parameters
    ----------
    hex_input : hex
        Hex string to store bytes in cuDF.

    Returns
    -------
    bytes
        Hex encoded object converted to bytes.
    """

    return bytes.fromhex(hex_input)


def hexfrombytes(bytes_input):
    """
    Function to bytes to hex string.

    Parameters
    ----------
    bytes_input : bytes
        Raw bytes of object.

    Returns
    -------
    hex
        Hex string to store bytes in cuDF.
    """

    return bytes_input.hex()


def bytesfrombase64(base64_input):
    """
    Function to convert base64 encoded string to bytes.

    Parameters
    ----------
    base64_input : hex
        Base64 encoded string to store bytes in cuDF.

    Returns
    -------
    bytes
        Base64 encoded string converted to bytes.
    """

    return base64.b64decode(base64_input)


def base64frombytes(bytes_input, encoding="utf-8"):
    """
    Function to bytes to base64 string.

    Parameters
    ----------
    bytes_input : bytes
        Raw bytes of object.

    Returns
    -------
    base64
        base64 encoded string to store bytes in cuDF.
    """

    return base64.b64encode(bytes_input).decode(encoding)
