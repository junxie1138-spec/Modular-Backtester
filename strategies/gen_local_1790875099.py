from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    entry_streak: int = 3
    reversal_streak: int = 3

    def __post_init__(self):
        if not all(isinstance(n, int) and 1 <= n <= 9 for n in (self.entry_streak, self.reversal_streak)):
            raise ValueError("Streak lengths must be integers from 1 to 9.")


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1790875099"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params):
        return max(params.entry_streak, params.reversal_streak) + 1

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["low_change"] = data["low"].diff()
        out["high_change"] = data["high"].diff()
        for name, flag, limit in (
            ("floor_streak", out["low_change"].gt(0), params.entry_streak),
            ("ceiling_streak", out["high_change"].lt(0), params.reversal_streak),
        ):
            alive = pd.Series(True, index=data.index)
            count = pd.Series(0, index=data.index, dtype=int)
            for lag in range(limit):
                alive = alive & flag.shift(lag, fill_value=False)
                count = count + alive.astype(int)
            out[name] = count
        return out

    def generate_signals(self, data, indicators, ctx, params):
        enter = indicators["floor_streak"].ge(params.entry_streak)
        reverse = indicators["ceiling_streak"].ge(params.reversal_streak)
        regime = pd.Series(np.nan, index=data.index, dtype=float)
        regime.loc[enter & ~reverse] = 1.0
        regime.loc[reverse] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = regime.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    close = pd.Series(
        [100., 101., 102., 103., 102., 103., 102., 101., 100., 101.],
        index=pd.date_range("2020-01-01", periods=10),
    )
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.,
    })
    strategy = GeneratedStrategy()
    params = StreakParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    for end in range(1, len(data) + 1):
        prefix = data.iloc[:end]
        actual = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
        assert actual.equals(result.iloc[:end])
    for entry in (2, 3, 4):
        for reversal in (2, 3, 4):
            assert strategy.warmup_bars(StreakParams(entry, reversal)) <= 10
    missing = data * np.nan
    assert strategy.generate_signals(
        missing, strategy.indicators(missing, params), ctx, params
    ).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
