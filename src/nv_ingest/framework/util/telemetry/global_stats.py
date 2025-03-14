# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
from collections import deque
from statistics import mean
from statistics import median

TELEMETRY_WINDOW_SIZE = os.getenv("TELEMETRY_WINDOW_SIZE", 100)


class GlobalStats:
    """
    Singleton class for maintaining global and job-specific statistics within a pipeline.

    This class is designed to keep track of various statistics, including the number of submitted and completed jobs,
    as well as dynamic statistics (mean and median) for job-specific metrics over a sliding window.

    Usage
    -----
    global_stats = GlobalStats.get_instance()

    # Setting and incrementing global statistics
    global_stats.set_stat("submitted_jobs", 5)
    global_stats.increment_stat("completed_jobs")

    # Appending job-specific statistics
    global_stats.append_job_stat("job_1", 100)
    global_stats.append_job_stat("job_1", 200)

    # Retrieving statistics
    submitted_jobs = global_stats.get_stat("submitted_jobs")
    job_1_mean = global_stats.get_job_stat("job_1", "mean")

    Methods
    -------
    get_instance():
        Returns the singleton instance of the GlobalStats class.

    reset_all_stats():
        Resets all global and job-specific statistics.

    set_stat(stat_name, value):
        Sets a specific global statistic to the given value.

    increment_stat(stat_name, value=1):
        Increments a specific global statistic by the given value (default is 1).

    append_job_stat(job_name, value):
        Appends a value to the job-specific statistics and updates the mean and median for the job.

    get_stat(stat_name):
        Retrieves the value of a specific global statistic.

    get_job_stat(job_name, stat_name):
        Retrieves the value of a specific job-specific statistic (mean or median).

    get_all_stats():
        Returns a dictionary containing all global and job-specific statistics.

    Attributes
    ----------
    max_jobs : int
        The maximum number of jobs to retain in the statistics window (default is 100).

    stats : dict
        Dictionary to hold global statistics.

    job_stats : defaultdict
        Dictionary to hold job-specific statistics, with each job having its own deque for values, mean, and median.

    Example
    -------
    >>> global_stats = GlobalStats.get_instance()
    >>> global_stats.increment_stat("submitted_jobs", 10)
    >>> global_stats.append_job_stat("job_1", 50)
    >>> global_stats.append_job_stat("job_1", 70)
    >>> print(global_stats.get_stat("submitted_jobs"))
    10
    >>> print(global_stats.get_job_stat("job_1", "mean"))
    60.0
    >>> print(global_stats.get_job_stat("job_1", "median"))
    60.0
    """

    _instance = None

    @staticmethod
    def get_instance():
        if GlobalStats._instance is None:
            GlobalStats()
        return GlobalStats._instance

    def __init__(self):
        if GlobalStats._instance is not None:
            raise Exception("This class is a singleton. Use `GlobalStats.get_instance()`.")
        GlobalStats._instance = self

        self.max_jobs = TELEMETRY_WINDOW_SIZE

        self.reset_all_stats()

    def reset_all_stats(self):
        self.stats = {
            "submitted_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
        }
        self.job_stats = defaultdict(lambda: {"values": deque(), "mean": 0.0, "median": 0.0})

    def set_stat(self, stat_name, value):
        self.stats[stat_name] = value

    def increment_stat(self, stat_name, value=1):
        self.stats[stat_name] += value

    def append_job_stat(self, job_name, value):
        job_stat = self.job_stats[job_name]
        values = job_stat["values"]

        if len(values) >= self.max_jobs:
            values.popleft()
        values.append(value)

        job_stat["mean"] = mean(values)
        job_stat["median"] = median(values)

    def get_stat(self, stat_name):
        if stat_name not in self.stats:
            raise ValueError(f"Key {stat_name} does not exist.")
        return self.stats[stat_name]

    def get_job_stat(self, job_name, stat_name):
        return self.job_stats[job_name][stat_name]

    def get_all_stats(self):
        return {
            "global_stats": self.stats.copy(),
            "job_stats": {job_name: stats.copy() for job_name, stats in self.job_stats.items()},
        }

    def __str__(self):
        return str(self.get_all_stats())
