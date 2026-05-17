from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.core.types import StrategyContext


def test_params_type_and_defaults():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    assert MLSupertrendStrategy.params_type() is MLSupertrendParams
    p = MLSupertrendParams()
    assert p.signal_mode == "reversal"
    assert p.require_new_extreme is True
    assert p.min_bars_between_signals == 10
    assert p.sensitivity == 30
    assert p.atr_period == 24
    assert p.multiplier == 1.4
    assert p.source_type == "hlcc4"
    assert p.use_atr is True
    assert p.enable_rsi is True
    assert p.rsi_len == 14
    assert p.rsi_lookback_top == 50
    assert p.rsi_lookback_bot == 50
    assert p.rsi_top == 70
    assert p.rsi_bot == 30
    assert p.vol_lookback == 3
    assert p.vol_multiplier == 1.2
    assert p.require_vol_spike is False
    assert p.enable_major_levels_only is False
    assert p.major_level_threshold == 4.5
    assert p.size == 1.0


def test_strategy_identity_and_warmup():
    from strategies.ml_supertrend import MLSupertrendStrategy, MLSupertrendParams

    s = MLSupertrendStrategy()
    assert s.strategy_id == "ml_supertrend"
    assert s.timeframe == "1d"
    # warmup = max(atr_period, sensitivity, rsi_len, vol_lookback) + 1
    assert s.warmup_bars(MLSupertrendParams()) == 31
    assert s.warmup_bars(MLSupertrendParams(atr_period=60, sensitivity=10)) == 61
