# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backoff strategies for retry logic.

This module implements the Strategy pattern for different backoff algorithms
used in retry scenarios. Each strategy encapsulates its own parameters and
logic for calculating delays between retry attempts.
"""

import random
from abc import ABC, abstractmethod
from typing import Literal


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies."""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0):
        """
        Initialize the backoff strategy.

        Parameters
        ----------
        base_delay : float
            Base delay in seconds for the first retry
        max_delay : float
            Maximum delay in seconds (capped to prevent excessive waits)
        """
        self.base_delay = base_delay
        self.max_delay = max_delay

    @abstractmethod
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay for the given attempt number.

        Parameters
        ----------
        attempt : int
            Current attempt number (0-based, so first retry is attempt 0)

        Returns
        -------
        float
            Delay in seconds before the next retry attempt
        """
        pass

    def _add_jitter(self, delay: float) -> float:
        """Add random jitter (±25%) to the delay to prevent thundering herd."""
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)


class ExponentialBackoffStrategy(BackoffStrategy):
    """Exponential backoff strategy with jitter."""

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Formula: delay = base_delay * (2 ^ attempt) + jitter
        """
        delay = self.base_delay * (2**attempt)
        delay = min(delay, self.max_delay)
        return self._add_jitter(delay)


class LinearBackoffStrategy(BackoffStrategy):
    """Linear backoff strategy with jitter."""

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate linear backoff delay.

        Formula: delay = base_delay * (attempt + 1) + jitter
        """
        delay = self.base_delay * (attempt + 1)
        delay = min(delay, self.max_delay)
        return self._add_jitter(delay)


class FixedBackoffStrategy(BackoffStrategy):
    """Fixed delay backoff strategy with jitter."""

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate fixed backoff delay.

        Formula: delay = base_delay + jitter
        """
        delay = self.base_delay
        delay = min(delay, self.max_delay)
        return self._add_jitter(delay)


# Strategy factory and registry
BackoffStrategyType = Literal["exponential", "linear", "fixed"]

STRATEGY_CLASSES = {
    "exponential": ExponentialBackoffStrategy,
    "linear": LinearBackoffStrategy,
    "fixed": FixedBackoffStrategy,
}


def create_backoff_strategy(
    strategy_type: BackoffStrategyType, base_delay: float = 1.0, max_delay: float = 60.0
) -> BackoffStrategy:
    """
    Factory function to create backoff strategy instances.

    Parameters
    ----------
    strategy_type : BackoffStrategyType
        Type of backoff strategy to create
    base_delay : float
        Base delay in seconds
    max_delay : float
        Maximum delay in seconds

    Returns
    -------
    BackoffStrategy
        Configured backoff strategy instance

    Raises
    ------
    ValueError
        If strategy_type is not supported
    """
    if strategy_type not in STRATEGY_CLASSES:
        supported = list(STRATEGY_CLASSES.keys())
        raise ValueError(f"Unsupported strategy type: {strategy_type}. Supported: {supported}")

    strategy_class = STRATEGY_CLASSES[strategy_type]
    return strategy_class(base_delay=base_delay, max_delay=max_delay)
