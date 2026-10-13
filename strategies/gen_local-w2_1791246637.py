from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeFailureParams:
    lookback: int = 6
    expansion: float = 1.6
    rejection: float = 0.2
    profit_target: float = 0.015
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[RangeFailureParams]):
    strategy_id = "gen_local-w2_1791246637"

    @classmethod
    def params_type(cls) -> type[RangeFailureParams]:
        return RangeFailureParams

    def warmup_bars(self, params: RangeFailureParams) -> int:
        if not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        if not (params.expansion > 1 and 0 < params.rejection < 1
                and params.profit_target > 0 and params.max_hold >= 1):
            raise ValueError("Invalid entry or exit parameters")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: RangeFailureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        n = params.lookback
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"])
        out["baseline"] = out["range"].rolling(n, min_periods=n).median().shift(1)
        out["ceiling"] = data["high"].rolling(n, min_periods=n).max().shift(1)
        out["floor"] = data["low"].rolling(n, min_periods=n).min().shift(1)
        out["expansion"] = out["range"] / out["baseline"]
        out["lower_excess"] = (out["floor"] - data["low"]) / out["range"]
        out["upper_excess"] = (data["high"] - out["ceiling"]) / out["range"]
        inside = data["close"].ge(out["floor"]) & data["close"].le(out["ceiling"])
        expanded = out["expansion"].ge(params.expansion)
        long_entry = (expanded & inside & out["lower_excess"].ge(params.rejection)
                      & data["high"].le(out["ceiling"]))
        short_entry = (expanded & inside & out["upper_excess"].ge(params.rejection)
                       & data["low"].ge(out["floor"]))
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0)
        out["reset"] = out["expansion"].le(1.0).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeFailureParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        resets = indicators["reset"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        entry_price = np.nan
        armed = True
        for i in range(len(data)):
            if resets[i]:
                armed = True
            if position:
                if entry_bar == -1:
                    # Record the execution open only when its bar is reached.
                    entry_price, entry_bar = opens[i], i
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                if ((np.isfinite(gain) and gain >= params.profit_target)
                        or i - entry_bar + 1 >= params.max_hold):
                    position, entry_bar = 0, -1
                    continue
            elif armed and entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position, entry_bar = int(entries[i]), -1
                armed = False
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = RangeFailureParams()
    for direction, take_profit in [(1, False), (1, True), (-1, True)]:
        data = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0,
             "close": 100.0, "volume": 1000.0},
            index=pd.date_range("2020-01-01", periods=17))
        data.iloc[6, data.columns.get_loc("low" if direction == 1 else "high")] = (
            96.0 if direction == 1 else 104.0)
        if take_profit:
            data.iloc[7, data.columns.get_loc("close")] = 100.0 + direction * 2.0
            data.iloc[7, data.columns.get_loc("high" if direction == 1 else "low")] = (
                103.0 if direction == 1 else 97.0)
        ind = strategy.indicators(data, params)
        result = strategy.generate_signals(data, ind, None, params).data
        assert strategy.warmup_bars(params) <= 10
        assert (result["signal"].iloc[:7] == 0).all()
        assert result["signal"].iloc[7] == direction
        exit_bar = 8 if take_profit else 15
        assert (result["signal"].iloc[7:exit_bar] == direction).all()
        assert result["signal"].iloc[exit_bar] == 0
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    demo()
