from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    profit_target: float = 0.04
    hold_bars: int = 18

    def __post_init__(self):
        if not np.isfinite(self.profit_target) or not 0.0 < self.profit_target < 1.0:
            raise ValueError("profit_target must be between zero and one")
        if not isinstance(self.hold_bars, int) or not 15 <= self.hold_bars <= 20:
            raise ValueError("hold_bars must be an integer from 15 to 20")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791260306"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 63

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        for label, window in (("fast", 10), ("slow", 63)):
            out[label + "_low"] = data["low"].rolling(window, min_periods=window).min()
            out[label + "_high"] = data["high"].rolling(window, min_periods=window).max()
            out[label + "_width"] = out[label + "_high"] - out[label + "_low"]
            out[label + "_position"] = (
                (data["close"] - out[label + "_low"])
                / out[label + "_width"].where(out[label + "_width"] > 0.0)
            ).replace([np.inf, -np.inf], np.nan)
        out["roc"] = (
            data["close"] / data["close"].shift(5).where(data["close"].shift(5) > 0.0) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"] - out["roc"].shift(5)
        out["previous_acceleration"] = out["acceleration"].shift(1)
        long_entry = (
            (out["fast_position"] >= 0.8) & (out["slow_position"] < 0.5)
            & (out["roc"] > 0.0) & (out["acceleration"] > 0.0)
            & (out["previous_acceleration"] <= 0.0)
        )
        short_entry = (
            (out["fast_position"] <= 0.2) & (out["slow_position"] > 0.5)
            & (out["roc"] < 0.0) & (out["acceleration"] < 0.0)
            & (out["previous_acceleration"] >= 0.0)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        direction = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if direction:
                # Record the fill only when its opening bar has arrived.
                if entry_bar < 0:
                    entry_price = opens[i]
                    entry_bar = i
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        direction = 0
                        entry_bar = -1
                        continue
                gain = direction * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    direction = 0
                    entry_bar = -1
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                direction = int(entries[i])
                entry_bar = -1
            signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=25)
    for direction in (1, -1):
        data = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
            index=index,
        )
        entries = pd.DataFrame({"entry": [direction] + [0] * 24}, index=index)
        result = strategy.generate_signals(data, entries, None, params).data
        assert result["signal"].iloc[0] == 0
        assert (result["signal"].iloc[1:19] == direction).all()
        assert (result["signal"].iloc[19:] == 0).all()
        data.loc[index[2], "close"] = 105.0 if direction == 1 else 95.0
        result = strategy.generate_signals(data, entries, None, params).data
        assert result["signal"].iloc[2] == direction
        assert result["signal"].iloc[3] == 0
        assert (result["size"] > 0.0).all()
    flat = strategy.indicators(data.iloc[:5], params)
    assert (flat["entry"] == 0).all()


if __name__ == "__main__":
    _self_check()
