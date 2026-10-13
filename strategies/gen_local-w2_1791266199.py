from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VarianceQueueParams:
    window: int = 18
    fast_window: int = 4
    pressure_threshold: float = 0.25
    cooldown_ratio: float = 0.95


class GeneratedStrategy(BaseStrategy[VarianceQueueParams]):
    strategy_id = "gen_local-w2_1791266199"

    @classmethod
    def params_type(cls):
        return VarianceQueueParams

    def warmup_bars(self, params):
        return int(max(params.window, params.fast_window) + 2)

    def indicators(self, data, params):
        if not (2 <= params.fast_window <= params.window):
            raise ValueError("Require 2 <= fast_window <= window")
        if not (0 < params.pressure_threshold < 1 and 0 < params.cooldown_ratio <= 1):
            raise ValueError("Invalid pressure_threshold or cooldown_ratio")
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        out["up_energy"] = out["return"].clip(lower=0).pow(2).rolling(params.window, min_periods=params.window).sum()
        out["down_energy"] = out["return"].clip(upper=0).pow(2).rolling(params.window, min_periods=params.window).sum()
        out["total_energy"] = out["up_energy"] + out["down_energy"]
        out["pressure"] = (out["down_energy"] - out["up_energy"]) / out["total_energy"].where(out["total_energy"] > 0)
        out["fast_vol"] = out["return"].rolling(params.fast_window, min_periods=params.fast_window).std(ddof=0)
        out["slow_vol"] = out["return"].rolling(params.window, min_periods=params.window).std(ddof=0)
        out["cooling"] = (out["fast_vol"] < params.cooldown_ratio * out["slow_vol"]) & (out["slow_vol"] > 0)
        out["candidate"] = np.select(
            [out["cooling"] & (out["pressure"] > params.pressure_threshold),
             out["cooling"] & (out["pressure"] < -params.pressure_threshold)],
            [1, -1], default=0
        ).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i, (candidate, confirmed) in enumerate(zip(
            indicators["candidate"].fillna(0).to_numpy(),
            indicators["confirmed"].fillna(0).to_numpy()
        )):
            if position and candidate == -position:
                position = 0
            elif position == 0 and confirmed:
                position = int(confirmed)
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = VarianceQueueParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"close": 100.0}, index=index)
    candidate = pd.Series([0, 1, 1, 0, -1, -1, 0, 1, 1], index=index)
    indicators = pd.DataFrame({
        "candidate": candidate,
        "confirmed": candidate.where(candidate.eq(candidate.shift(1)), 0)
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1, -1, 0]
    flat = pd.DataFrame({"close": 100.0}, index=pd.date_range("2020-01-01", periods=40))
    derived = strategy.indicators(flat, params)
    assert derived["candidate"].eq(0).all()
    assert strategy.generate_signals(flat, derived, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
