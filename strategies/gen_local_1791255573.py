from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalPressureParams:
    baseline_bars: int = 20
    pressure_bars: int = 8
    entry_threshold: float = 0.12
    conviction_scale: float = 0.5
    min_size: float = 0.2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SeasonalPressureParams]):
    strategy_id = "gen_local_1791255573"

    @classmethod
    def params_type(cls):
        return SeasonalPressureParams

    def warmup_bars(self, params):
        return int(params.baseline_bars + params.pressure_bars + 1)

    def indicators(self, data, params):
        if params.baseline_bars < 2 or params.pressure_bars < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < params.entry_threshold and 0 < params.conviction_scale):
            raise ValueError("Threshold and conviction scale must be positive.")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["return_scale"] = out["return"].rolling(params.baseline_bars).std(ddof=0)
        out["volume_baseline"] = data["volume"].shift(1).rolling(params.baseline_bars).mean()
        out["relative_volume"] = (data["volume"] / out["volume_baseline"].replace(0, np.nan)).clip(0, 3)
        out["normalized_move"] = (out["return"] / out["return_scale"].replace(0, np.nan)).clip(-2, 2)
        out["seasonal_window"] = ((data.index.day >= 25) | (data.index.day <= 7)).astype(float)
        out["impulse"] = out["normalized_move"] * out["relative_volume"] * out["seasonal_window"]
        out["pressure"] = out["impulse"].rolling(params.pressure_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        pressure = indicators["pressure"].fillna(0)
        df = pd.DataFrame(index=data.index)
        # The same predicate enters and exits; no time, price, or risk exit.
        df["signal"] = (pressure > params.entry_threshold).astype(int)
        strength = (pressure - params.entry_threshold).clip(lower=0)
        df["size"] = params.min_size + (params.max_size - params.min_size) * strength / (strength + params.conviction_scale)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SeasonalPressureParams()
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2020-01-01", periods=160)
    close = 100 * np.exp(np.cumsum(0.003 + 0.006 * np.sin(np.arange(160))))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = (indicators["pressure"].fillna(0) > params.entry_threshold).astype(int).shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(result["signal"], expected, check_names=False)
    assert result["signal"].eq(1).any() and result["signal"].diff().eq(-1).any()
    assert result["size"].between(params.min_size, params.max_size).all()
    prefix = data.iloc[:100]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(result.iloc[:100], prefix_result)


if __name__ == "__main__":
    _self_check()
