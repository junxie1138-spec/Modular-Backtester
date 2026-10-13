from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_bars: int = 5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1790969427"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return 64 + max(1, int(params.streak_bars))

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if params.streak_bars < 1 or params.hold_bars < 1:
            raise ValueError("streak_bars and hold_bars must be positive")
        out = pd.DataFrame(index=data.index)
        spread = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["close_location"] = ((data["close"] - data["low"]) / spread).replace([np.inf, -np.inf], np.nan)
        out["prior_high"] = data["high"].shift(1).rolling(63, min_periods=63).max()
        out["prior_low"] = data["low"].shift(1).rolling(63, min_periods=63).min()
        width = (out["prior_high"] - out["prior_low"]).where(lambda x: x > 0)
        out["range_position"] = ((data["close"] - out["prior_low"]) / width).replace([np.inf, -np.inf], np.nan)
        valid = out["close_location"].notna() & out["range_position"].notna()
        strong = valid & out["close_location"].ge(0.70)
        weak = valid & out["close_location"].le(0.30)
        out["strong_streak"] = strong.astype(int).groupby((~strong).cumsum()).cumsum()
        out["weak_streak"] = weak.astype(int).groupby((~weak).cumsum()).cumsum()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StreakParams) -> SignalFrame:
        if params.streak_bars < 1 or params.hold_bars < 1:
            raise ValueError("streak_bars and hold_bars must be positive")
        long_entry = indicators["strong_streak"].eq(params.streak_bars) & indicators["range_position"].between(0.0, 0.35)
        short_entry = indicators["weak_streak"].eq(params.streak_bars) & indicators["range_position"].between(0.65, 1.0)
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_bar = i + int(params.hold_bars)
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({"strong_streak": 0, "weak_streak": 0, "range_position": 0.5}, index=index)
    derived.iloc[0] = [5, 0, 0.2]
    derived.iloc[5] = [0, 5, 0.8]
    derived.iloc[20] = [0, 5, 0.8]
    result = GeneratedStrategy().generate_signals(data, derived, None, StreakParams()).data
    expected = np.zeros(45, dtype=int)
    expected[1:19] = 1
    expected[21:39] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()
