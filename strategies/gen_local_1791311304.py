from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    peak_window: int = 20
    min_depth: float = 0.01
    max_range_fraction: float = 0.35


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791311304"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(200, params.peak_window + 3))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if params.peak_window < 2 or not 0.0 < params.min_depth < 1.0 or not 0.0 < params.max_range_fraction <= 1.0:
            raise ValueError("Invalid compression parameters")
        out = pd.DataFrame(index=data.index)
        out["close"] = data["close"]
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["peak"] = data["close"].rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = 1.0 - data["close"] / out["peak"].where(out["peak"] > 0.0)
        out["depth_change"] = out["depth"].diff()
        out["lagged_depth_change"] = out["depth_change"].shift(2)
        distance = out["peak"] - data["close"]
        out["range_fraction"] = (data["high"] - data["low"]) / distance.where(distance > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        entry_condition = (
            indicators.notna().all(axis=1)
            & (indicators["close"] > indicators["ma200"])
            & (indicators["depth"] >= params.min_depth)
            & (indicators["range_fraction"] >= 0.0)
            & (indicators["range_fraction"] <= params.max_range_fraction)
            & (indicators["depth_change"] < 0.0)
            & (indicators["lagged_depth_change"] > 0.0)
        )
        df = pd.DataFrame(index=data.index)
        # A false entry condition exits; there are no independent exit triggers.
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    close = np.full(241, 190.0)
    close[200:234] = 210.0
    close[234:] = [206.0, 204.0, 202.0, 202.4, 202.8, 203.2, 203.6]
    data = pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=len(close)),
    )
    strategy = GeneratedStrategy()
    params = CompressionParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert signals["signal"].tail(4).tolist() == [0, 1, 1, 0]
    assert signals["signal"].iloc[:200].eq(0).all()
    assert signals["signal"].isin([0, 1]).all()
    assert signals["size"].gt(0.0).all()
    prefix = data.iloc[:-1]
    prefix_signals = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(prefix_signals, signals.iloc[:-1])
