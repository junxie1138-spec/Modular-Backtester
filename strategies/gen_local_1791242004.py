from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    lookback: int = 30
    profit_target: float = 0.025


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = "gen_local_1791242004"

    @classmethod
    def params_type(cls):
        return BreakoutParams

    def warmup_bars(self, params):
        return params.lookback + 1

    def indicators(self, data, params):
        if params.lookback < 2 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("lookback must be >= 2 and profit_target must be positive and finite")
        n = params.lookback
        result = pd.DataFrame(index=data.index)
        result["low_percentile"] = data["low"].rolling(n, min_periods=n).rank(pct=True)
        result["prior_close_ceiling"] = data["close"].shift(1).rolling(n, min_periods=n).max()
        result["prior_high_ceiling"] = data["high"].shift(1).rolling(n, min_periods=n).max()
        result["entry"] = (
            (result["low_percentile"] >= 0.95)
            & (data["low"] > result["prior_close_ceiling"])
            & (data["close"] > result["prior_high_ceiling"])
            & (data["close"] > data["open"])
        ).fillna(False)
        return result

    def generate_signals(self, data, indicators, ctx, params):
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        holding = False
        entry_price = None
        held_bars = 0
        for i in range(len(data)):
            if holding:
                # Yesterday's decision fills at today's open.
                if entry_price is None:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        raise ValueError("Entry open must be positive and finite")
                    entry_price = opens[i]
                held_bars += 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                # Exit decisions fill at the following open, like entries.
                if target_hit or held_bars >= 8:
                    holding = False
                    entry_price = None
                    held_bars = 0
                else:
                    signals[i] = 1
            elif entries[i]:
                holding = True
                entry_price = None
                held_bars = 0
                signals[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=11)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True] + [False] * 10}, index=index)
    params = BreakoutParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0, 0]
    data.loc[index[2], "close"] = 103.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1] + [0] * 8
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
