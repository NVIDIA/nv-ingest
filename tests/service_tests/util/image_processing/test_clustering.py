# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_api.util.image_processing.clustering import (
    boxes_are_close_or_overlap,
    group_bounding_boxes,
    combine_groups_into_bboxes,
    remove_superset_bboxes,
)


def test_boxes_are_close_or_overlap():
    from_box = [0, 0, 10, 10]
    to_box = [15, 15, 25, 25]
    assert not boxes_are_close_or_overlap(from_box, to_box, threshold=1)
    assert boxes_are_close_or_overlap(from_box, to_box, threshold=5)


def test_group_bounding_boxes():
    boxes = [[0, 0, 10, 10], [10, 10, 20, 20], [100, 100, 110, 110]]
    # The second and third boxes should group together
    groups = group_bounding_boxes(boxes, threshold=2)
    assert len(groups) == 2
    assert sorted(groups[0]) == [0, 1]
    assert sorted(groups[1]) == [2]


def test_combine_groups_into_bboxes():
    boxes = [[0, 0, 1, 1], [2, 2, 3, 3], [1, 1, 2, 2]]
    groups = [[0], [1, 2]]
    combined = combine_groups_into_bboxes(boxes, groups)
    assert len(combined) == 2
    assert combined[1] == [1, 1, 3, 3]


def test_remove_superset_bboxes():
    bboxes = [[0, 0, 10, 10], [2, 2, 4, 4], [3, 3, 5, 5]]
    # The first box encloses the second but not the third strictly
    out = remove_superset_bboxes(bboxes)
    assert len(out) == 2
    assert [0, 0, 10, 10] not in out
