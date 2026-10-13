from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    fractional_window: int = 32
    percentile_window: int = 126
    tail_quantile: float = 0.10
    recovery_quantile: float = 0.25
    refractory_bars: int = 2
    setup_bars: int = 5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791208084"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.fractional_window + params.percentile_window
                   + params.refractory_bars + params.setup_bars + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0 < params.fractional_order < 1
                and 0 < params.tail_quantile < params.recovery_quantile < 0.5
                and params.fractional_window >= 2
                and params.percentile_window >= 2
                and params.refractory_bars >= 1
                and params.setup_bars >= 1
                and params.hold_bars >= 1):
            raise ValueError("Invalid fractional filter, quantiles, or bar windows.")
        out = pd.DataFrame(index=data.index)
        prices = np.log(data["close"].where(data["close"] > 0)).to_numpy(dtype=float)
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, len(weights)):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        filtered = np.full(len(data), np.nan)
        # ponytail: finite fractional memory; enlarge fractional_window if truncation matters.
        if len(data) >= len(weights):
            filtered[len(weights) - 1:] = np.convolve(prices, weights, mode="valid")
        out["fractional_close"] = filtered
        history = out["fractional_close"].shift(1).rolling(params.percentile_window)
        out["lower_tail"] = history.quantile(params.tail_quantile)
        out["upper_tail"] = history.quantile(1 - params.tail_quantile)
        out["lower_recovery"] = history.quantile(params.recovery_quantile)
        out["upper_recovery"] = history.quantile(1 - params.recovery_quantile)
        out["down_spike"] = out["fractional_close"] < out["lower_tail"]
        out["up_spike"] = out["fractional_close"] > out["upper_tail"]
        for side in ("down", "up"):
            spike = out[side + "_spike"].astype(float)
            old_spike = spike.shift(params.refractory_bars).rolling(params.setup_bars).max()
            recent_spike = spike.rolling(params.refractory_bars).max()
            out[side + "_ready"] = old_spike.eq(1) & recent_spike.eq(0)
        fd = out["fractional_close"]
        long_entry = (out["down_ready"] & fd.ge(out["lower_recovery"])
                      & fd.shift(1).lt(out["lower_recovery"].shift(1)))
        short_entry = (out["up_ready"] & fd.le(out["upper_recovery"])
                       & fd.shift(1).gt(out["upper_recovery"].shift(1)))
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        positions = np.zeros(len(data), dtype=int)
        active = 0
        exit_bar = -1
        for i, entry in enumerate(indicators["entry"].fillna(0).to_numpy()):
            if active:
                if i == exit_bar:
                    active = 0
            elif entry:
                active = int(entry)
                exit_bar = i + params.hold_bars
            positions[i] = active
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(hold_bars=3)
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=10))
    indicators = pd.DataFrame({"entry": [1, -1, -1, -1, 0, -1, 1, 1, 1, 0]},
                              index=data.index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, -1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
