# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_api.internal.primitives.nim.default_values import *


def test_yolox_max_batch_size():
    assert 1 <= YOLOX_MAX_BATCH_SIZE <= 128, "YOLOX_MAX_BATCH_SIZE should be between 1 and 128"


def test_yolox_max_dimensions():
    assert 64 <= YOLOX_MAX_WIDTH <= 4096, "YOLOX_MAX_WIDTH should be between 64 and 4096"
    assert 64 <= YOLOX_MAX_HEIGHT <= 4096, "YOLOX_MAX_HEIGHT should be between 64 and 4096"


def test_yolox_conf_threshold():
    assert 0.001 <= YOLOX_CONF_THRESHOLD <= 1.0, "YOLOX_CONF_THRESHOLD should be between 0.001 and 1.0"


def test_yolox_iou_threshold():
    assert 0.1 <= YOLOX_IOU_THRESHOLD <= 1.0, "YOLOX_IOU_THRESHOLD should be between 0.1 and 1.0"


def test_yolox_min_score():
    assert 0.0 <= YOLOX_MIN_SCORE <= 1.0, "YOLOX_MIN_SCORE should be between 0.0 and 1.0"


def test_yolox_score_consistency():
    assert (
        YOLOX_CONF_THRESHOLD <= YOLOX_MIN_SCORE <= YOLOX_FINAL_SCORE
    ), "Thresholds should be in order: CONF_THRESHOLD <= MIN_SCORE <= FINAL_SCORE"
