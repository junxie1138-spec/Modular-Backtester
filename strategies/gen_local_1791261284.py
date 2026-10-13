from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_threshold: float = 0.15
    trail_atr: float = 2.5
    hold_bars: int = 18
    atr_period: int = 14
    calendar_days: int = 7
    gap_memory: int = 3


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791261284"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(200, params.atr_period + 2, params.gap_memory + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.atr_period >= 1 and params.gap_memory >= 1
                and params.hold_bars >= 1 and 1 <= params.calendar_days <= 28
                and np.isfinite(params.gap_threshold) and params.gap_threshold > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["gap_atr"] = out["gap"] / out["prior_atr"].where(out["prior_atr"] > 0)
        out["prior_gap_pressure"] = out["gap"].shift(1).rolling(
            params.gap_memory, min_periods=params.gap_memory
        ).sum()
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["month_start"] = data.index.day <= params.calendar_days
        out["entry"] = (
            out["month_start"]
            & (out["gap_atr"] >= params.gap_threshold)
            & (out["prior_gap_pressure"] < 0)
            & (data["close"] > out["ma200"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if active:
                if not valid:
                    active = False
                else:
                    # Entry executes after the decision bar; track held closes only.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    if close <= stop or i - entry_bar >= params.hold_bars:
                        active = False
            elif valid and entries[i]:
                active = True
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    dates = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [150, 100, 103, 102, 100.5, 101, 102]}, index=dates)
    derived = pd.DataFrame({
        "atr": [1, 1, 1, 10, 10, 1, 1],
        "entry": [True, False, False, False, False, False, False],
    }, index=dates)
    strategy = object.__new__(GeneratedStrategy)
    params = Params(trail_atr=2.0)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    params.hold_bars = 3
    derived["atr"] = 100.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    derived["entry"] = False
    assert not strategy.generate_signals(data, derived, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
