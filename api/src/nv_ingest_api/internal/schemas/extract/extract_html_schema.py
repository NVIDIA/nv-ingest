# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import ConfigDict, BaseModel

logger = logging.getLogger(__name__)


class HtmlExtractorSchema(BaseModel):
    """
    Configuration schema for the Html extractor settings.

    Parameters
    ----------
    max_queue_size : int, default=1
        The maximum number of items allowed in the processing queue.

    n_workers : int, default=16
        The number of worker threads to use for processing.

    raise_on_failure : bool, default=False
        A flag indicating whether to raise an exception on processing failure.

    """

    max_queue_size: int = 1
    n_workers: int = 16
    raise_on_failure: bool = False

    model_config = ConfigDict(extra="forbid")
