from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CalendarQueueParams:
    z_entry: float = 1.0
    profit_target: float = 0.035
    hold_bars: int = 18

    def __post_init__(self):
        if not np.isfinite(self.z_entry) or self.z_entry <= 0:
            raise ValueError("z_entry must be finite and positive")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not isinstance(self.hold_bars, int) or self.hold_bars < 1:
            raise ValueError("hold_bars must be a positive integer")


class GeneratedStrategy(BaseStrategy[CalendarQueueParams]):
    strategy_id = "gen_local-w2_1791214632"

    @classmethod
    def params_type(cls):
        return CalendarQueueParams

    @staticmethod
    def warmup_bars(params: CalendarQueueParams) -> int:
        return 200

    @staticmethod
    def indicators(data: pd.DataFrame, params: CalendarQueueParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma20"] = close.rolling(20, min_periods=20).mean()
        out["std20"] = close.rolling(20, min_periods=20).std(ddof=0)
        out["z"] = ((close - out["ma20"]) / out["std20"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["calendar_day"] = data.index.day
        ready = np.isfinite(out["z"]) & np.isfinite(out["ma200"]) & (close > 0)
        buy = ready & (out["calendar_day"] <= 7) & (out["z"] <= -params.z_entry) & (close > out["ma200"])
        sell = ready & out["calendar_day"].between(18, 24) & (out["z"] >= params.z_entry) & (close < out["ma200"])
        out["entry_side"] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CalendarQueueParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry_side"].to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    side = 0
                else:
                    gain = side * (closes[i] / entry_price - 1.0)
                    age = i - entry_bar + 1
                    if gain >= params.profit_target or age >= params.hold_bars:
                        side = 0
                # An exit keeps this close's decision flat; no immediate re-entry.
            elif entries[i]:
                side = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-01-01", periods=9)
    data = pd.DataFrame({"open": 100.0, "high": 130.0, "low": 95.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[0], "close"] = 130.0
    data.loc[index[1], "close"] = 104.0
    entries = pd.DataFrame({"entry_side": [1, 0, 0, 0, -1, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = CalendarQueueParams(profit_target=0.03, hold_bars=3)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["entry_side"].eq(0).all()
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:end], entries.iloc[:end], None, params).data
        assert prefix["signal"].equals(result["signal"].iloc[:end])


if __name__ == "__main__":
    _self_check()
