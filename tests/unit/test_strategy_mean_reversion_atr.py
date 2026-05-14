import pandas as pd
import pytest


def _ohlcv(closes, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def test_strategy_is_registered():
    from backtester.strategies.registry import get_strategy_class
    cls = get_strategy_class("mean_reversion_atr")
    assert cls.strategy_id == "mean_reversion_atr"


def test_strategy_uses_multi_symbol_and_per_bar():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy
    assert MeanReversionAtrStrategy.uses_multi_symbol is True
    assert MeanReversionAtrStrategy.uses_per_bar is True


def test_params_defaults_match_prd():
    from strategies.mean_reversion_atr import MeanReversionAtrParams
    p = MeanReversionAtrParams()
    assert p.entry_atr_mult == 1.25
    assert p.mean_lookback == 10
    assert p.atr_lookback == 20
    assert p.time_stop_days == 7
    assert p.runner_time_stop_days == 12
    assert p.runner_ceiling_atr_mult == 1.25
    assert p.runtime_trend_threshold == 0.0025


def test_indicators_emits_mean10_atr20_slope200():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    s = MeanReversionAtrStrategy()
    data = _ohlcv([100.0 + 0.1 * i for i in range(250)])
    ind = s.indicators(data, MeanReversionAtrParams())
    assert "mean10" in ind.columns
    assert "atr20" in ind.columns
    assert "slope_log_200d" in ind.columns
    assert pd.isna(ind["mean10"].iloc[0])
    assert pd.isna(ind["atr20"].iloc[0])


def test_warmup_bars_correct():
    from strategies.mean_reversion_atr import MeanReversionAtrStrategy, MeanReversionAtrParams
    s = MeanReversionAtrStrategy()
    p = MeanReversionAtrParams()
    assert s.warmup_bars(p) >= 200  # slope_log_200d is the deepest lookback
