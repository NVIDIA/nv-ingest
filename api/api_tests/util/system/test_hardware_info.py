# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import nv_ingest_api.util.system.hardware_info as module_under_test

from unittest.mock import patch

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@pytest.mark.parametrize("hyperthread_weight", [0.0, 0.5, 1.0])
def test_init_accepts_valid_hyperthread_weights(hyperthread_weight):
    probe = module_under_test.SystemResourceProbe(hyperthread_weight)
    assert probe.hyperthread_weight == (hyperthread_weight if module_under_test.psutil else 1.0)


def test_init_rejects_invalid_hyperthread_weight():
    with pytest.raises(ValueError):
        module_under_test.SystemResourceProbe(-0.1)
    with pytest.raises(ValueError):
        module_under_test.SystemResourceProbe(1.1)


@patch(f"{MODULE_UNDER_TEST}.os.cpu_count", return_value=8)
@patch(f"{MODULE_UNDER_TEST}.psutil.cpu_count", return_value=8)
@patch(f"{MODULE_UNDER_TEST}.platform.system", return_value="Linux")
@patch(f"{MODULE_UNDER_TEST}.os.sched_getaffinity", return_value=set(range(4)))
@patch(f"{MODULE_UNDER_TEST}.os.path.exists", return_value=False)  # Simulate no cgroup files
def test_probe_detects_affinity_and_os_counts(
    mock_exists, mock_affinity, mock_platform, mock_psutil_cpu_count, mock_os_cpu_count
):
    probe = module_under_test.SystemResourceProbe()
    details = probe.get_details()

    assert details["os_logical_cores"] == 8
    assert details["os_sched_affinity_cores"] == 4
    assert details["effective_cores"] is not None


@patch(f"{MODULE_UNDER_TEST}.os.path.exists", side_effect=lambda path: path == module_under_test.CGROUP_V2_CPU_FILE)
@patch(f"{MODULE_UNDER_TEST}.SystemResourceProbe._read_file_str", return_value="100000 100000")
@patch(f"{MODULE_UNDER_TEST}.os.cpu_count", return_value=8)
def test_probe_detects_cgroup_v2(mock_cpu_count, mock_read_file_str, mock_exists):
    probe = module_under_test.SystemResourceProbe()
    details = probe.get_details()

    assert details["cgroup_type"] == "v2"
    assert details["cgroup_quota_cores"] == 1.0
    assert details["effective_cores"] == 1.0
    assert "cgroup_v2_quota" in details["detection_method"]


@patch(f"{MODULE_UNDER_TEST}.os.path.exists", side_effect=lambda path: path == module_under_test.CGROUP_V1_CPU_DIR)
@patch(
    f"{MODULE_UNDER_TEST}.SystemResourceProbe._read_file_int",
    side_effect=lambda path: 100000 if "quota" in path else 100000 if "period" in path else None,
)
@patch(f"{MODULE_UNDER_TEST}.os.cpu_count", return_value=8)
def test_probe_detects_cgroup_v1(mock_cpu_count, mock_read_file_int, mock_exists):
    probe = module_under_test.SystemResourceProbe()
    details = probe.get_details()

    assert details["cgroup_type"] == "v1"
    assert details["cgroup_quota_cores"] == 1.0
    assert details["effective_cores"] == 1.0
    assert "cgroup_v1_quota" in details["detection_method"]


@patch(f"{MODULE_UNDER_TEST}.psutil", None)  # Simulate missing psutil
@patch(f"{MODULE_UNDER_TEST}.os.cpu_count", return_value=16)
@patch(f"{MODULE_UNDER_TEST}.platform.system", return_value="Linux")
@patch(f"{MODULE_UNDER_TEST}.os.sched_getaffinity", return_value=set(range(8)))
def test_probe_no_psutil_fallback_to_os_count(mock_affinity, mock_platform, mock_cpu_count):
    probe = module_under_test.SystemResourceProbe(hyperthread_weight=0.5)
    details = probe.get_details()

    assert details["os_logical_cores"] == 16
    assert details["effective_cores"] >= 8
    assert details["hyperthread_weight_applied"] == 1.0  # Forced to 1.0 due to missing psutil


@patch(f"{MODULE_UNDER_TEST}.psutil.cpu_count", return_value=None)
@patch(f"{MODULE_UNDER_TEST}.os.cpu_count", return_value=None)
@patch(f"{MODULE_UNDER_TEST}.platform.system", return_value="Linux")
@patch(f"{MODULE_UNDER_TEST}.os.sched_getaffinity", return_value=set())
def test_probe_fallback_to_default(mock_affinity, mock_platform, mock_os_cpu_count, mock_psutil_cpu_count):
    probe = module_under_test.SystemResourceProbe()
    details = probe.get_details()

    assert details["raw_limit_value"] == 1.0
