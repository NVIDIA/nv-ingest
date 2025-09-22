# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import patch

from nv_ingest_api.data_handlers.backoff_strategies import (
    ExponentialBackoffStrategy,
    LinearBackoffStrategy,
    FixedBackoffStrategy,
    create_backoff_strategy,
)


class TestBackoffStrategies:
    """Black-box tests for backoff strategies and factory."""

    @patch("nv_ingest_api.data_handlers.backoff_strategies.random.random", return_value=0.5)
    def test_exponential_backoff_no_jitter_and_cap(self, _):
        """Exponential delay doubles per attempt and respects max_delay cap; jitter neutralized."""
        strat = ExponentialBackoffStrategy(base_delay=1.0, max_delay=5.0)
        # With jitter neutral (random=0.5), delay is exact
        assert strat.calculate_delay(0) == 1.0  # 1 * 2^0
        assert strat.calculate_delay(1) == 2.0  # 1 * 2^1
        assert strat.calculate_delay(2) == 4.0  # 1 * 2^2
        # Next would be 8.0 but capped at 5.0
        assert strat.calculate_delay(3) == 5.0
        assert strat.calculate_delay(10) == 5.0

    @patch("nv_ingest_api.data_handlers.backoff_strategies.random.random", return_value=0.5)
    def test_linear_backoff_no_jitter_and_cap(self, _):
        """Linear delay grows linearly and respects max_delay cap; jitter neutralized."""
        strat = LinearBackoffStrategy(base_delay=1.5, max_delay=4.0)
        # attempt 0 => 1.5, attempt 1 => 3.0, attempt 2 => 4.0 (cap)
        assert strat.calculate_delay(0) == 1.5
        assert strat.calculate_delay(1) == 3.0
        assert strat.calculate_delay(2) == 4.0
        assert strat.calculate_delay(5) == 4.0

    @patch("nv_ingest_api.data_handlers.backoff_strategies.random.random", return_value=0.5)
    def test_fixed_backoff_no_jitter_and_cap(self, _):
        """Fixed delay equals base_delay up to cap; jitter neutralized."""
        strat = FixedBackoffStrategy(base_delay=2.0, max_delay=10.0)
        for attempt in range(0, 5):
            assert strat.calculate_delay(attempt) == 2.0

        # Cap smaller than base
        strat2 = FixedBackoffStrategy(base_delay=2.0, max_delay=1.0)
        assert strat2.calculate_delay(0) == 1.0

    def test_factory_creates_correct_types(self):
        """Factory returns appropriate strategy instances with parameters passed through."""
        exp = create_backoff_strategy("exponential", base_delay=0.7, max_delay=9.0)
        lin = create_backoff_strategy("linear", base_delay=0.3, max_delay=2.0)
        fix = create_backoff_strategy("fixed", base_delay=5.0, max_delay=5.0)

        assert isinstance(exp, ExponentialBackoffStrategy)
        assert isinstance(lin, LinearBackoffStrategy)
        assert isinstance(fix, FixedBackoffStrategy)

        assert exp.base_delay == 0.7 and exp.max_delay == 9.0
        assert lin.base_delay == 0.3 and lin.max_delay == 2.0
        assert fix.base_delay == 5.0 and fix.max_delay == 5.0

    def test_factory_unsupported_strategy_raises(self):
        """Unsupported strategy type should raise ValueError listing supported types."""
        with pytest.raises(ValueError) as exc:
            create_backoff_strategy("unknown", base_delay=1.0, max_delay=2.0)  # type: ignore[arg-type]
        msg = str(exc.value)
        assert "Unsupported strategy type" in msg
        # Must list supported values
        assert "exponential" in msg and "linear" in msg and "fixed" in msg

    def test_jitter_bounds_are_reasonable(self):
        """Jitter should stay within ±25% and never below 0.1 seconds."""
        strat = FixedBackoffStrategy(base_delay=2.0, max_delay=100.0)
        # We won't mock random here; just sample a few times
        samples = [strat.calculate_delay(0) for _ in range(50)]
        for s in samples:
            assert 0.1 <= s
            # Must stay within 25% of base (since fixed)
            assert 1.5 <= s <= 2.5

    @patch("nv_ingest_api.data_handlers.backoff_strategies.random.random", return_value=0.0)
    def test_jitter_min_floor_applies(self, _):
        """When jitter would push below 0.1s, the 0.1s floor should apply."""
        # base_delay=0.1, negative jitter of 25% yields 0.075, floor to 0.1
        strat = FixedBackoffStrategy(base_delay=0.1, max_delay=100.0)
        assert strat.calculate_delay(0) == 0.1
