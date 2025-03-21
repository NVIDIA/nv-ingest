# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


def cm_ensure_payload_not_null(control_message: IngestControlMessage):
    """
    Ensures that the payload of a IngestControlMessage is not None.

    Parameters
    ----------
    control_message : IngestControlMessage
        The IngestControlMessage to check.

    Raises
    ------
    ValueError
        If the payload is None.
    """

    if control_message.payload() is None:
        raise ValueError("Payload cannot be None")


def cm_set_failure(control_message: IngestControlMessage, reason: str) -> IngestControlMessage:
    """
    Sets the failure metadata on a IngestControlMessage.

    Parameters
    ----------
    control_message : IngestControlMessage
        The IngestControlMessage to set the failure metadata on.
    reason : str
        The reason for the failure.

    Returns
    -------
    control_message : IngestControlMessage
        The modified IngestControlMessage with the failure metadata set.
    """

    control_message.set_metadata("cm_failed", True)
    control_message.set_metadata("cm_failed_reason", reason)

    return control_message
