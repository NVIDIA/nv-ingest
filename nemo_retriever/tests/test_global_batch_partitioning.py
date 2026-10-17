# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.graph.executor import _repartition_global_batch


class FakeDataset:
    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks
        self.repartition_calls: list[dict] = []

    def num_blocks(self) -> int:
        return self._num_blocks

    def repartition(self, **kwargs):
        self.repartition_calls.append(kwargs)
        return self


def test_grouped_global_batch_keeps_existing_partition_count() -> None:
    dataset = FakeDataset(478)
    assert _repartition_global_batch(dataset, ["source_path"], 1) is dataset
    assert dataset.repartition_calls == [
        {"num_blocks": 478, "keys": ["source_path"], "shuffle": True}
    ]


def test_grouped_global_batch_can_expand_for_actor_concurrency() -> None:
    dataset = FakeDataset(1)
    _repartition_global_batch(dataset, ["source_path"], (1, 4, 1))
    assert dataset.repartition_calls == [
        {"num_blocks": 4, "keys": ["source_path"], "shuffle": True}
    ]


def test_ungrouped_global_batch_still_coalesces_to_one_block() -> None:
    dataset = FakeDataset(478)
    _repartition_global_batch(dataset, [], 8)
    assert dataset.repartition_calls == [{"num_blocks": 1}]
