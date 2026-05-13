from __future__ import annotations

import pytest

from backtester.config.models import (
    DataConfig, ExecutionConfig, PortfolioConfig, RunConfig, WFOConfig,
)
from backtester.config.validation import validate_run_config
from backtester.core.exceptions import ConfigError


def _make(**over):
    base = RunConfig(
        run_name="x",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 30},
        data=DataConfig(symbols=["SPY"], timeframe="1d", start="2020-01-01", end="2024-01-01"),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(),
    )
    for k, v in over.items():
        setattr(base, k, v)
    return base


def test_valid_config_passes():
    validate_run_config(_make())


def test_empty_symbols_rejected():
    rc = _make()
    rc.data.symbols = []
    with pytest.raises(ConfigError, match="symbols"):
        validate_run_config(rc)


def test_start_after_end_rejected():
    rc = _make()
    rc.data.start = "2025-01-01"
    rc.data.end = "2024-01-01"
    with pytest.raises(ConfigError, match="start"):
        validate_run_config(rc)


def test_negative_cash_rejected():
    rc = _make()
    rc.execution.initial_cash = -1
    with pytest.raises(ConfigError, match="initial_cash"):
        validate_run_config(rc)


def test_wfo_requires_windows():
    rc = _make(wfo=WFOConfig(enabled=True))
    with pytest.raises(ConfigError, match="train_bars"):
        validate_run_config(rc)


def test_wfo_valid_when_windows_set():
    rc = _make(wfo=WFOConfig(enabled=True, train_bars=252, test_bars=63, step_bars=63))
    validate_run_config(rc)
