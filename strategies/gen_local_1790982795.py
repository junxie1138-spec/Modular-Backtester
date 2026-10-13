from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    correlation_floor: float = 0.15
    volume_ratio: float = 1.15
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1790982795"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["previous_return"] = out["return"].shift(1)
        out["two_day_return"] = out["return"].shift(2)
        out["autocorrelation_one"] = out["return"].rolling(6, min_periods=6).corr(out["previous_return"]).replace([np.inf, -np.inf], np.nan)
        out["autocorrelation_two"] = out["return"].rolling(6, min_periods=6).corr(out["two_day_return"]).replace([np.inf, -np.inf], np.nan)
        out["volume_baseline"] = data["volume"].shift(1).rolling(5, min_periods=5).mean()
        out["relative_volume"] = data["volume"].div(out["volume_baseline"].where(out["volume_baseline"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["previous_relative_volume"] = out["relative_volume"].shift(1)
        out["previous_volume"] = data["volume"].shift(1)
        out["entry"] = (
            (out["autocorrelation_one"] < -params.correlation_floor)
            & (out["autocorrelation_two"] > params.correlation_floor)
            & (out["previous_return"] > 0)
            & (out["return"] < 0)
            & (out["return"] >= -0.75 * out["previous_return"])
            & (out["previous_relative_volume"] >= params.volume_ratio)
            & (out["relative_volume"] < 1.0)
            & (data["volume"] > 0)
            & (data["volume"] < out["previous_volume"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveParams) -> SignalFrame:
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("holding_bars must be 3, 4, or 5")
        if not np.isfinite(params.correlation_floor) or not 0 <= params.correlation_floor < 1:
            raise ValueError("correlation_floor must be finite and in [0, 1)")
        if not np.isfinite(params.volume_ratio) or params.volume_ratio <= 0:
            raise ValueError("volume_ratio must be finite and positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        remaining = 0
        for i in range(len(data)):
            if remaining > 0:
                remaining -= 1
                signals[i] = int(remaining > 0)
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                remaining = params.holding_bars
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveParams()
    index = pd.date_range("2020-01-01", periods=20)
    prices = np.arange(100.0, 120.0)
    data = pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 100.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    derived["entry"] = True
    for hold in (3, 4, 5):
        params.holding_bars = hold
        result = strategy.generate_signals(data, derived, None, params).data
        first_fill = strategy.warmup_bars(params)
        assert result["signal"].iloc[:first_fill].eq(0).all()
        assert result["signal"].iloc[first_fill:first_fill + hold].eq(1).all()
        assert result["signal"].iloc[first_fill + hold] == 0
        assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
