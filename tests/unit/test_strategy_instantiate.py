from __future__ import annotations

import pytest

from backtester.core.exceptions import StrategyError
from backtester.strategies.instantiate import build_strategy_and_params


def test_build_known_strategy_with_defaults():
    strat, params = build_strategy_and_params("sma_cross", {})
    assert strat.strategy_id == "sma_cross"
    assert params.fast == 20  # default


def test_build_known_strategy_with_overrides():
    strat, params = build_strategy_and_params("sma_cross", {"fast": 5, "slow": 25})
    assert params.fast == 5 and params.slow == 25


def test_build_unknown_strategy_raises():
    with pytest.raises(KeyError):
        build_strategy_and_params("does_not_exist", {})


def test_build_unknown_param_key_raises():
    with pytest.raises(StrategyError, match="unknown"):
        build_strategy_and_params("sma_cross", {"not_a_field": 1})
