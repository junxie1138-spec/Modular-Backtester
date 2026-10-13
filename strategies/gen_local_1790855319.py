from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilitySeasonParams:
    lookback: int = 16
    hub_threshold: float = 0.25
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[VisibilitySeasonParams]):
    strategy_id = "gen_local_1790855319"

    @classmethod
    def params_type(cls):
        return VisibilitySeasonParams

    def warmup_bars(self, params: VisibilitySeasonParams) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: VisibilitySeasonParams) -> pd.DataFrame:
        if params.lookback < 3 or params.holding_bars < 1 or not 0 < params.hub_threshold <= 1:
            raise ValueError("Require lookback >= 3, holding_bars >= 1, and 0 < hub_threshold <= 1.")
        close = data["close"].where(data["close"] > 0)
        ready = close.rolling(params.lookback, min_periods=params.lookback).count().eq(params.lookback)
        peak_degree = pd.Series(1.0, index=data.index)
        valley_degree = pd.Series(1.0, index=data.index)
        # Horizontal visibility: all intermediate prices lie below both endpoints.
        # Mirroring prices reverses the inequality and identifies valley hubs.
        for lag in range(2, params.lookback):
            past = close.shift(lag)
            middle = close.shift(1).rolling(lag - 1, min_periods=lag - 1)
            peak_degree += (middle.max() < np.minimum(close, past)).astype(float)
            valley_degree += (middle.min() > np.maximum(close, past)).astype(float)
        peak_degree = peak_degree.where(ready)
        valley_degree = valley_degree.where(ready)
        total_degree = peak_degree + valley_degree
        balance = (valley_degree - peak_degree) / total_degree
        seasonal = pd.Series((data.index.day >= 24) | (data.index.day <= 5), index=data.index)
        eligible = ready & seasonal & total_degree.ge(5)
        setup = pd.Series(np.select(
            [eligible & balance.ge(params.hub_threshold),
             eligible & balance.le(-params.hub_threshold)],
            [1, -1], default=0
        ), index=data.index, dtype=int)
        return pd.DataFrame({
            "ready": ready, "peak_degree": peak_degree,
            "valley_degree": valley_degree, "total_degree": total_degree,
            "hub_balance": balance, "seasonal_window": seasonal, "setup": setup
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VisibilitySeasonParams
    ) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive.")
        setup = indicators["setup"].fillna(0).astype(int)
        confirmed = setup.where(setup.ne(0) & setup.eq(setup.shift(1)), 0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(confirmed.to_numpy()):
            if position != 0 and i - entry_bar >= params.holding_bars:
                position = 0
                continue
            if position == 0 and candidate != 0:
                position = int(candidate)
                entry_bar = i
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = VisibilitySeasonParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=params.holding_bars + 3, freq="B")
    data = pd.DataFrame(index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        setup = [direction, direction] + [-direction] * (len(index) - 2)
        indicators = pd.DataFrame({"setup": setup}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, 0] + [direction] * params.holding_bars + [0]
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
