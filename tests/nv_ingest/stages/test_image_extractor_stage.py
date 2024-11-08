# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
from unittest.mock import patch, Mock
import pandas as pd

from nv_ingest.stages.extractors.image_extractor_stage import process_image

MODULE_UNDER_TEST = 'nv_ingest.stages.extractors.image_extractor_stage'


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'source_id': ['id1', 'id2'],
        'content': ['base64image1', 'base64image2']
    })


@pytest.fixture
def task_props():
    return {'method': 'image', 'params': {}}


@pytest.fixture
def validated_config():
    return Mock()


@pytest.fixture
def trace_info():
    return {}
