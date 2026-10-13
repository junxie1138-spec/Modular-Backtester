from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeShockParams:
    lookback: int = 20
    shock_threshold: float = 1.5


class GeneratedStrategy(BaseStrategy[RangeShockParams]):
    strategy_id = "gen_local_1790870596"

    @classmethod
    def params_type(cls):
        return RangeShockParams

    def warmup_bars(self, params: RangeShockParams) -> int:
        return params.lookback + max(3, params.lookback // 4) + 1

    def indicators(self, data: pd.DataFrame, params: RangeShockParams) -> pd.DataFrame:
        if params.lookback < 8 or not np.isfinite(params.shock_threshold) or params.shock_threshold <= 0:
            raise ValueError("lookback must be >= 8 and shock_threshold must be finite and positive")
        fast = max(3, params.lookback // 4)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["range_fraction"] = (data["high"] - data["low"]) / out["previous_close"]
        out["log_range"] = np.log(out["range_fraction"].where(out["range_fraction"] > 0))
        out["upper_excursion"] = (data["high"] - out["previous_close"]).clip(lower=0)
        out["lower_excursion"] = (out["previous_close"] - data["low"]).clip(lower=0)
        out["excursion_total"] = out["upper_excursion"] + out["lower_excursion"]
        out["upper_share"] = out["upper_excursion"] / out["excursion_total"].where(out["excursion_total"] > 0)
        out["recent_log_range"] = out["log_range"].rolling(fast).mean()
        out["baseline_log_range"] = out["log_range"].shift(fast).rolling(params.lookback).mean()
        out["baseline_std"] = out["log_range"].shift(fast).rolling(params.lookback).std(ddof=0)
        out["range_shock"] = (out["recent_log_range"] - out["baseline_log_range"]) / (out["baseline_std"].clip(lower=0.05) * np.sqrt(1.0 / fast + 1.0 / params.lookback))
        out["recent_share"] = out["upper_share"].rolling(fast).mean()
        out["baseline_share"] = out["upper_share"].shift(fast).rolling(params.lookback).mean()
        out["share_shift"] = out["recent_share"] - out["baseline_share"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeShockParams) -> SignalFrame:
        active = indicators["range_shock"].gt(params.shock_threshold)
        bullish = active & indicators["share_shift"].gt(0.15) & indicators["recent_share"].gt(0.5)
        bearish = active & indicators["share_shift"].lt(-0.15) & indicators["recent_share"].lt(0.5)
        candidate = np.select([bullish.fillna(False), bearish.fillna(False)], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(candidate):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeShockParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "range_shock": [np.nan, 3, 0, 3, 3, 0, 3, 0],
        "share_shift": [np.nan, 0.3, 0, -0.3, -0.3, 0, 0.3, 0],
        "recent_share": [np.nan, 0.8, 0.5, 0.2, 0.2, 0.5, 0.8, 0.5],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    index = pd.date_range("2020-01-01", periods=100)
    close = pd.Series(100 + np.arange(100) * 0.1, index=index)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000}, index=index)
    indicators = strategy.indicators(data, params)
    assert indicators.index.equals(data.index)
    assert indicators["range_shock"].first_valid_index() == index[strategy.warmup_bars(params) - 1]
    assert not np.isinf(indicators.to_numpy()).any()
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].isin([-1, 0, 1]).all()


if __name__ == "__main__":
    _self_check()
