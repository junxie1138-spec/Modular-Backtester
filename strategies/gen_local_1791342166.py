from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PatternCapacityParams:
    min_patterns: int = 6
    edge: float = 0.75
    profit_target: float = 0.02
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[PatternCapacityParams]):
    strategy_id = "gen_local_1791342166"

    @classmethod
    def params_type(cls):
        return PatternCapacityParams

    def warmup_bars(self, params):
        return 8

    def indicators(self, data, params):
        if not 4 <= params.min_patterns <= 6:
            raise ValueError("min_patterns must be between 4 and 6")
        if not 0.5 < params.edge < 1 or params.profit_target <= 0 or params.max_hold < 1:
            raise ValueError("Invalid edge, profit_target or max_hold")
        a, b, c = data["close"].shift(2), data["close"].shift(1), data["close"]
        pattern = pd.Series(np.select(
            [(a < b) & (b < c), (a < c) & (c < b),
             (b < a) & (a < c), (b < c) & (c < a),
             (c < a) & (a < b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=np.nan), index=data.index)
        out = pd.DataFrame({"pattern": pattern}, index=data.index)
        counts = pd.DataFrame(index=data.index)
        novel = pd.Series(False, index=data.index)
        for code in range(6):
            present = pattern.eq(code).astype(float).where(pattern.notna())
            counts[code] = present.shift(1).rolling(5, min_periods=5).sum()
            out["count_" + str(code)] = counts[code]
            novel |= pattern.eq(code) & counts[code].eq(0)
        ready = counts.notna().all(axis=1) & pattern.notna()
        out["novel"] = novel & ready
        out["complexity"] = (counts.gt(0).sum(axis=1) + novel.astype(int)).where(ready)
        lo = data["low"].rolling(8, min_periods=8).min()
        hi = data["high"].rolling(8, min_periods=8).max()
        out["range_low"], out["range_high"] = lo, hi
        out["position"] = (c - lo) / (hi - lo).where(hi > lo)
        event = out["novel"] & out["complexity"].ge(params.min_patterns)
        out["entry"] = np.select(
            [event & out["position"].le(1 - params.edge),
             event & out["position"].ge(params.edge)], [1, -1], default=0)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        side, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        side = 0
                if side:
                    gain = side * (closes[i] / entry_price - 1)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                        side = 0
                decisions[i] = side
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                entry_bar = i + 1
                decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=14)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 103.0
    data.loc[index[5], "close"] = 97.0
    entries = pd.DataFrame({"entry": [0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = PatternCapacityParams(max_hold=3)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], "open"] = 104.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].iloc[3] == 1
    assert strategy.warmup_bars(params) <= 10
    assert strategy.indicators(data, params)["entry"].iloc[:7].eq(0).all()


if __name__ == "__main__":
    _self_check()
