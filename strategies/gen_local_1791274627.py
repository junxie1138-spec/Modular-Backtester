from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FloorBreakoutParams:
    streak_bars: int = 3
    breakout_bars: int = 12
    atr_bars: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.75
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.streak_bars, self.breakout_bars, self.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not np.isfinite(self.breakeven_pct) or self.breakeven_pct <= 0:
            raise ValueError("breakeven_pct must be positive and finite")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")


class GeneratedStrategy(BaseStrategy[FloorBreakoutParams]):
    strategy_id = "gen_local_1791274627"

    @classmethod
    def params_type(cls):
        return FloorBreakoutParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.breakout_bars, params.atr_bars, params.streak_bars) + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["ceiling"] = data["high"].rolling(params.breakout_bars, min_periods=params.breakout_bars).max().shift(1)
        out["rising_floor"] = (data["low"] > out["previous_low"]).fillna(False)
        resets = (~out["rising_floor"]).cumsum()
        out["floor_streak"] = out["rising_floor"].astype(int).groupby(resets).cumsum().clip(upper=params.streak_bars)
        out["entry"] = (
            out["floor_streak"].ge(params.streak_bars)
            & data["close"].gt(out["ceiling"])
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    # The prior close's order fills at this bar's open.
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * atrs[i - 1]
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                if np.isfinite(closes[i]) and closes[i] > 0:
                    peak = max(peak, closes[i])
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atrs[i])
                    # Close-confirmed exits execute at the following open.
                    if closes[i] <= stop:
                        active = False
                if age >= params.hold_bars:
                    active = False
                signals[i] = int(active)
            elif i >= self.warmup_bars(params) - 1 and entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = FloorBreakoutParams(streak_bars=2, breakout_bars=2, atr_bars=2)
    index = pd.date_range("2020-01-01", periods=9)
    for closes, expected in [
        ([100, 100, 100, 101, 99, 100, 100, 100, 100], [0, 0, 0, 1, 1, 0, 0, 0, 0]),
        ([100, 100, 100, 101, 104, 102, 100, 100, 100], [0, 0, 0, 1, 1, 1, 0, 0, 0]),
        ([100, 100, 100, 101, 102, 103, 104, 105, 106], [0, 0, 0, 1, 1, 1, 1, 0, 0]),
    ]:
        data = pd.DataFrame({"open": 100.0, "close": closes}, index=index)
        indicators = pd.DataFrame({"atr": 1.0, "entry": [False, False, True] + [False] * 6}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
