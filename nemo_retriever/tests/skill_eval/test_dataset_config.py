# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from nemo_retriever.tools.skill_eval.dataset import load_config


@pytest.mark.parametrize("text, expected", [("", {}), ("agent: claude\ntrials: 2\n", {"agent": "claude", "trials": 2})])
def test_load_config_mapping(tmp_path, text, expected):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    assert load_config(path) == expected


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found:"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_requires_top_level_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- item\n")
    with pytest.raises(ValueError, match="YAML config must be a mapping/object at top-level:"):
        load_config(path)


def test_load_config_rejects_malformed_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("agent: [\n")
    with pytest.raises(yaml.YAMLError):
        load_config(path)
