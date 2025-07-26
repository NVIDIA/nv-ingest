#!/usr/bin/env python3
"""
Random UDF function for nv-ingest pipeline.

This module contains a UDF function that adds random metadata to ingestion data.
"""

# noqa
# flake8: noqa


def add_two_metadata(control_message: "IngestControlMessage") -> "IngestControlMessage":
    import logging

    logger = logging.getLogger(__name__)
    logger.info("UDF: Adding two metadata to control message")
    new_cm = add_random_metadata(control_message)
    new_cm = add_random_metadata_2(new_cm)

    return new_cm


def add_random_metadata(control_message: "IngestControlMessage") -> "IngestControlMessage":
    """
    UDF function that adds a random number to each row's metadata.custom_content.udf_random

    This function:
    1. Accesses the IngestControlMessage payload (pandas DataFrame)
    2. Iterates through each row's metadata
    3. Adds a random number to metadata.custom_content.udf_random
    4. Persists the changes back to the control message

    Parameters
    ----------
    control_message : IngestControlMessage
        The control message containing the payload and metadata

    Returns
    -------
    IngestControlMessage
        The modified control message with random metadata added
    """
    import random
    import logging

    logger = logging.getLogger(__name__)
    logger.info("UDF: Adding random metadata to control message")

    # Get the payload DataFrame
    payload = control_message.payload()
    if payload is not None:
        df = payload
        logger.info(f"UDF: Processing DataFrame with {len(df)} rows")

        # Iterate through each row and modify metadata
        for idx, row in df.iterrows():
            # Get existing metadata or create new
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Ensure custom_content exists
            if "custom_content" not in metadata:
                metadata["custom_content"] = {}
            elif not isinstance(metadata["custom_content"], dict):
                metadata["custom_content"] = {}

            # Add random number
            random_value = random.randint(1, 1000)
            metadata["custom_content"]["udf_random"] = random_value

            # Update the row metadata
            df.at[idx, "metadata"] = metadata
            logger.debug(f"UDF: Added random value {random_value} to row {idx}")

        # Persist changes back to control message
        control_message.payload(df)
        logger.info("UDF: Successfully added random metadata to all rows")
    else:
        logger.warning("UDF: No payload found in control message")

    return control_message


def add_random_metadata_2(control_message: "IngestControlMessage") -> "IngestControlMessage":
    """
    UDF function that adds a random number to each row's metadata.custom_content.udf_random

    This function:
    1. Accesses the IngestControlMessage payload (pandas DataFrame)
    2. Iterates through each row's metadata
    3. Adds a random number to metadata.custom_content.udf_random
    4. Persists the changes back to the control message

    Parameters
    ----------
    control_message : IngestControlMessage
        The control message containing the payload and metadata

    Returns
    -------
    IngestControlMessage
        The modified control message with random metadata added
    """
    import random
    import logging

    logger = logging.getLogger(__name__)
    logger.info("UDF: Adding random metadata to control message")

    # Get the payload DataFrame
    payload = control_message.payload()
    if payload is not None:
        df = payload
        logger.info(f"UDF: Processing DataFrame with {len(df)} rows")

        # Iterate through each row and modify metadata
        for idx, row in df.iterrows():
            # Get existing metadata or create new
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Ensure custom_content exists
            if "custom_content" not in metadata:
                metadata["custom_content"] = {}
            elif not isinstance(metadata["custom_content"], dict):
                metadata["custom_content"] = {}

            # Add random number
            random_value = random.randint(1, 1000)
            metadata["custom_content"]["udf_random_2"] = random_value

            # Update the row metadata
            df.at[idx, "metadata"] = metadata
            logger.debug(f"UDF: Added random value {random_value} to row {idx}")

        # Persist changes back to control message
        control_message.payload(df)
        logger.info("UDF: Successfully added random metadata to all rows")
    else:
        logger.warning("UDF: No payload found in control message")

    return control_message


def add_timestamp_metadata(control_message: "IngestControlMessage") -> "IngestControlMessage":
    """
    Alternative UDF function that adds timestamp metadata.

    This demonstrates how multiple UDF functions can be defined in the same file.
    """
    import datetime
    import logging

    logger = logging.getLogger(__name__)
    logger.info("UDF: Adding timestamp metadata to control message")

    payload = control_message.payload()
    if payload is not None:
        df = payload
        current_time = datetime.datetime.now().isoformat()

        for idx, row in df.iterrows():
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            if "custom_content" not in metadata:
                metadata["custom_content"] = {}
            elif not isinstance(metadata["custom_content"], dict):
                metadata["custom_content"] = {}

            metadata["custom_content"]["processed_at"] = current_time
            df.at[idx, "metadata"] = metadata

        control_message.payload(df)
        logger.info(f"UDF: Added timestamp {current_time} to all rows")

    return control_message
