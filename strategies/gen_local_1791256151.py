from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RankRecoveryParams:
    rank_window: int = 6
    rank_cutoff: float = 0.3333333333333333
    hold_bars: int = 4

    def __post_init__(self):
        if not 2 <= self.rank_window <= 7:
            raise ValueError("rank_window must be between 2 and 7")
        if not 0.0 < self.rank_cutoff < 0.5:
            raise ValueError("rank_cutoff must be between 0 and 0.5")
        if self.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")


class GeneratedStrategy(BaseStrategy[RankRecoveryParams]):
    strategy_id = "gen_local_1791256151"

    @classmethod
    def params_type(cls):
        return RankRecoveryParams

    def warmup_bars(self, params):
        return int(3 + params.rank_window)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].where(data["close"] > 0).shift(1)
        out["peak"] = prior_close.rolling(3, min_periods=3).max()
        out["trough"] = prior_close.rolling(3, min_periods=3).min()
        for anchor, prefix in (("peak", "down"), ("trough", "up")):
            for price in ("open", "close"):
                name = prefix + "_" + price
                out[name] = (data[price] / out[anchor] - 1.0).replace([np.inf, -np.inf], np.nan)
                out[name + "_rank"] = out[name].rolling(
                    params.rank_window, min_periods=params.rank_window
                ).rank(pct=True)
        out["session_return"] = (data["close"] / data["open"].where(data["open"] > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        long_entry = (
            (out["down_open_rank"] <= params.rank_cutoff)
            & (out["down_close_rank"] >= 1.0 - params.rank_cutoff)
            & (out["down_close"] < 0.0)
            & (out["session_return"] > 0.0)
        )
        short_entry = (
            (out["up_open_rank"] >= 1.0 - params.rank_cutoff)
            & (out["up_close_rank"] <= params.rank_cutoff)
            & (out["up_close"] > 0.0)
            & (out["session_return"] < 0.0)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if i < self.warmup_bars(params) - 1:
                continue
            if position:
                if i == exit_bar:
                    position = 0
            elif entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    for hold in (3, 4, 5):
        params = RankRecoveryParams(hold_bars=hold)
        warmup = strategy.warmup_bars(params)
        data = pd.DataFrame(
            100.0, index=pd.date_range("2020-01-01", periods=warmup + hold + 3),
            columns=["open", "high", "low", "close", "volume"]
        )
        assert warmup <= 10
        assert (strategy.indicators(data, params)["entry"] == 0).all()
        ctx = StrategyContext("SPY", "1d", warmup)
        for side in (-1, 1):
            entries = pd.DataFrame({"entry": side}, index=data.index)
            signal = strategy.generate_signals(data, entries, ctx, params).data["signal"]
            assert (signal.iloc[:warmup] == 0).all()
            assert (signal.iloc[warmup:warmup + hold] == side).all()
            assert signal.iloc[warmup + hold] == 0
