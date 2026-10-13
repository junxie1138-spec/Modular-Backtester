from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 24
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791262799"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.lookback)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.lookback < 3 or params.holding_bars < 1:
            raise ValueError("lookback must be >= 3 and holding_bars must be >= 1")
        result = pd.DataFrame(index=data.index)
        prices = data["close"].astype(float)
        log_close = np.log(prices.where(prices.gt(0))).replace([np.inf, -np.inf], np.nan)
        degree = pd.Series(0.0, index=data.index)
        upward_degree = degree.copy()
        long_degree = degree.copy()
        minimum_slope = pd.Series(np.inf, index=data.index)
        long_start = max(2, params.lookback // 2)
        # A distant vertex is visible iff every nearer backward slope is larger.
        for distance in range(1, params.lookback):
            slope = (log_close - log_close.shift(distance)) / distance
            visible = slope.notna() & slope.lt(minimum_slope)
            degree += visible.astype(float)
            upward_degree += (visible & slope.gt(0)).astype(float)
            if distance >= long_start:
                long_degree += visible.astype(float)
            minimum_slope = np.minimum(minimum_slope, slope)
        result["log_close"] = log_close
        result["degree"] = degree
        result["upward_fraction"] = upward_degree.div(degree.replace(0, np.nan)).fillna(0.0)
        result["long_reach"] = long_degree / (params.lookback - long_start)
        result["net_change"] = log_close - log_close.shift(params.lookback - 1)
        valid = log_close.rolling(params.lookback, min_periods=params.lookback).count().eq(params.lookback)
        result["entry"] = (
            valid
            & degree.ge(int(np.ceil(np.sqrt(params.lookback))))
            & result["upward_fraction"].ge(0.8)
            & result["long_reach"].ge(0.5)
            & result["net_change"].gt(0)
        ).fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be >= 1")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        exit_at = -1
        for bar in range(len(data)):
            if bar < exit_at:
                signals[bar] = 1
            elif bar == exit_at:
                signals[bar] = 0
            elif entries[bar]:
                signals[bar] = 1
                exit_at = bar + params.holding_bars
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = VisibilityParams(lookback=12, holding_bars=5)
    close = np.exp(0.0001 * np.arange(80, dtype=float) ** 2)
    data = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=80))
    strategy = GeneratedStrategy()
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, None, params).data["signal"].to_numpy()
    expected = np.zeros(80, dtype=int)
    for entry in range(params.lookback, 80, params.holding_bars + 1):
        expected[entry:entry + params.holding_bars] = 1
    assert np.array_equal(signals, expected)
    assert indicators["degree"].iloc[params.lookback - 1] == params.lookback - 1
    flat = data.assign(open=1.0, high=1.0, low=1.0, close=1.0)
    assert not strategy.indicators(flat, params)["entry"].any()
