from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalRelaxationParams:
    frac_order: float = 0.4
    frac_window: int = 64
    memory_window: int = 84
    entry_score: float = 1.25
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[FractionalRelaxationParams]):
    strategy_id = "gen_local_1791206828"

    @classmethod
    def params_type(cls):
        return FractionalRelaxationParams

    def warmup_bars(self, params):
        return int(params.frac_window + params.memory_window + 1)

    def indicators(self, data, params):
        if not (0.0 < params.frac_order < 1.0):
            raise ValueError("frac_order must be between zero and one")
        if params.frac_window < 2 or params.memory_window < 3:
            raise ValueError("indicator windows are too short")
        if params.hold_bars < 1 or params.entry_score <= 0.0:
            raise ValueError("hold_bars and entry_score must be positive")
        if not (0.0 < params.min_size <= params.max_size <= 1.0):
            raise ValueError("sizes must satisfy 0 < min_size <= max_size <= 1")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        fd = close.copy()
        weight = 1.0
        # ponytail: fixed fractional memory; increase frac_window for longer tails.
        for lag in range(1, params.frac_window):
            weight *= (lag - 1.0 - params.frac_order) / lag
            fd = fd + weight * close.shift(lag)
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fd
        prior = fd.shift(1)
        window = prior.rolling(params.memory_window, min_periods=params.memory_window)
        out["anchor"] = window.mean()
        out["dispersion"] = window.std(ddof=0).replace(0.0, np.nan)
        out["autocorrelation"] = window.corr(fd.shift(2)).clip(-1.0, 1.0)
        out["displacement"] = (fd - out["anchor"]) / out["dispersion"]
        out["relaxation_score"] = -out["displacement"] * (
            1.0 - out["autocorrelation"].clip(0.0, 1.0).pow(params.hold_bars)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        rho = indicators["autocorrelation"]
        score = indicators["relaxation_score"]
        eligible = (rho.between(0.1, 0.95) & score.ge(params.entry_score)).fillna(False)
        entry_sizes = (
            params.min_size
            + (params.max_size - params.min_size)
            * (score / (2.0 * params.entry_score)).clip(0.0, 1.0)
        ).fillna(params.min_size)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        entry_bar = -1
        locked_size = params.min_size
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.hold_bars:
                    entry_bar = -1
                else:
                    signals[i] = 1
                    sizes[i] = locked_size
            elif bool(eligible.iloc[i]):
                entry_bar = i
                locked_size = float(entry_sizes.iloc[i])
                signals[i] = 1
                sizes[i] = locked_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = FractionalRelaxationParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=2 * p.hold_bars + 5)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame(
        {"autocorrelation": 0.5, "relaxation_score": 2.0}, index=index
    )
    result = strategy.generate_signals(data, indicators, None, p).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:p.hold_bars + 1].eq(1).all()
    assert result["signal"].iloc[p.hold_bars + 1] == 0
    assert result["signal"].iloc[p.hold_bars + 2] == 1
    assert result["size"].iloc[1:p.hold_bars + 1].nunique() == 1
    assert result["size"].between(p.min_size, p.max_size).all()
