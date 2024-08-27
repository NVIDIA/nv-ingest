# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from statistics import mean
from statistics import median

from nv_ingest.util.telemetry.global_stats import GlobalStats


def setup_function():
    GlobalStats._instance = None


def test_singleton():
    gs1 = GlobalStats.get_instance()
    gs2 = GlobalStats.get_instance()
    assert gs1 is gs2, "GlobalStats should be a singleton"


def test_set_stat():
    gs = GlobalStats.get_instance()
    gs.set_stat("submitted_jobs", 5)
    assert gs.get_stat("submitted_jobs") == 5, "Stat 'submitted_jobs' should be set to 5"


def test_increment_stat():
    gs = GlobalStats.get_instance()
    gs.increment_stat("submitted_jobs", 2)
    assert gs.get_stat("submitted_jobs") == 2, "Stat 'submitted_jobs' should be incremented by 2"
    gs.increment_stat("submitted_jobs")
    assert gs.get_stat("submitted_jobs") == 3, "Stat 'submitted_jobs' should be incremented by 1"


def test_append_job_stat():
    gs = GlobalStats.get_instance()
    job_name = "job_1"
    values = [10, 20, 30]

    for value in values:
        gs.append_job_stat(job_name, value)

    assert gs.get_job_stat(job_name, "mean") == mean(values), "Mean should be calculated correctly"
    assert gs.get_job_stat(job_name, "median") == median(values), "Median should be calculated correctly"


def test_append_job_stat_window_size():
    gs = GlobalStats.get_instance()
    gs.max_jobs = 3  # Set a small window size for testing
    job_name = "job_1"
    values = [10, 20, 30, 40]  # 4 values, but window size is 3

    for value in values:
        gs.append_job_stat(job_name, value)

    expected_values = deque([20, 30, 40])
    assert list(gs.job_stats[job_name]["values"]) == list(
        expected_values
    ), "Oldest value should be removed when window size is exceeded"
    assert gs.get_job_stat(job_name, "mean") == mean(
        expected_values
    ), "Mean should be calculated for the values within the window size"
    assert gs.get_job_stat(job_name, "median") == median(
        expected_values
    ), "Median should be calculated for the values within the window size"


def test_reset_all_stats():
    gs = GlobalStats.get_instance()
    gs.set_stat("submitted_jobs", 5)
    gs.append_job_stat("job_1", 10)
    gs.reset_all_stats()

    assert gs.get_stat("submitted_jobs") == 0, "All global stats should be reset to 0"
    assert gs.get_job_stat("job_1", "values") == deque(), "All job stats should be reset"


def test_get_all_stats():
    gs = GlobalStats.get_instance()
    gs.set_stat("submitted_jobs", 5)
    gs.append_job_stat("job_1", 10)
    gs.append_job_stat("job_1", 20)

    all_stats = gs.get_all_stats()
    assert all_stats["global_stats"]["submitted_jobs"] == 5, "Global stats should be retrieved correctly"
    assert all_stats["job_stats"]["job_1"]["mean"] == 15, "Job stats should be retrieved correctly"
    assert all_stats["job_stats"]["job_1"]["median"] == 15, "Job stats should be retrieved correctly"
