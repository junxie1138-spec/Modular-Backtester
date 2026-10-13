from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    memory_bars: int = 64
    trend_bars: int = 18
    percentile_bars: int = 126
    entry_percentile: float = 0.80

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if min(self.memory_bars, self.trend_bars, self.percentile_bars) < 2:
            raise ValueError("lookbacks must be at least two")
        if not 0.5 < self.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between 0.5 and one")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791323075"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.memory_bars + params.trend_bars + params.percentile_bars + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        fractional_close = close.copy()
        weight = 1.0
        # ponytail: finite fractional memory; increase memory_bars for longer tails.
        for lag in range(1, params.memory_bars):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        innovation = fractional_close.diff()
        drift = innovation.rolling(params.trend_bars).mean()
        downside = innovation.clip(upper=0).pow(2).rolling(params.trend_bars).mean()
        upside = innovation.clip(lower=0).pow(2).rolling(params.trend_bars).mean()
        energy = innovation.pow(2).rolling(params.trend_bars).mean()
        countertrend = downside.where(drift >= 0, upside)
        denominator = (countertrend + 0.05 * energy).pow(0.5).replace(0, np.nan)
        strength = (drift / denominator).replace([np.inf, -np.inf], np.nan)
        threshold = strength.abs().shift(1).rolling(params.percentile_bars).quantile(params.entry_percentile)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "innovation": innovation,
            "drift": drift,
            "downside": downside,
            "upside": upside,
            "energy": energy,
            "countertrend": countertrend,
            "denominator": denominator,
            "strength": strength,
            "threshold": threshold,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        strength = indicators["strength"]
        threshold = indicators["threshold"]
        valid = strength.notna() & threshold.notna()
        candidate = np.select(
            [valid & (strength > threshold), valid & (strength < -threshold)],
            [1, -1],
            default=0,
        ).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(candidate):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({
        "strength": [np.nan, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0],
        "threshold": [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
