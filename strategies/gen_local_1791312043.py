from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeParams:
    compression_ratio: float = 0.75
    profit_target: float = 0.012


class GeneratedStrategy(BaseStrategy[RangeParams]):
    strategy_id = "gen_local_1791312043"

    @classmethod
    def params_type(cls) -> type[RangeParams]:
        return RangeParams

    def warmup_bars(self, params: RangeParams) -> int:
        return 25

    def indicators(self, data: pd.DataFrame, params: RangeParams) -> pd.DataFrame:
        if not np.isfinite(params.compression_ratio) or not 0 < params.compression_ratio < 1:
            raise ValueError("compression_ratio must be between zero and one")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["daily_range"] = (data["high"] - data["low"]) / close
        out["short_range"] = out["daily_range"].rolling(4, min_periods=4).mean()
        out["baseline_range"] = out["daily_range"].rolling(20, min_periods=20).mean().shift(4)
        out["compression"] = (out["short_range"] / out["baseline_range"].replace(0, np.nan)).shift(1)
        out["box_high"] = data["high"].rolling(4, min_periods=4).max().shift(1)
        out["box_low"] = data["low"].rolling(4, min_periods=4).min().shift(1)
        width = (out["box_high"] - out["box_low"]).replace(0, np.nan)
        out["edge_location"] = (close.shift(1) - out["box_low"]) / width
        out["roc"] = close / close.shift(2) - 1.0
        out["acceleration"] = out["roc"].diff()
        out["previous_acceleration"] = out["acceleration"].shift(1)
        out["close_change"] = close.diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        compressed = out["compression"].lt(params.compression_ratio)
        out["long_entry"] = (
            compressed
            & out["edge_location"].le(0.35)
            & out["roc"].lt(0)
            & out["previous_acceleration"].le(0)
            & out["acceleration"].gt(0)
            & out["close_change"].gt(0)
        ).fillna(False)
        out["short_entry"] = (
            compressed
            & out["edge_location"].ge(0.65)
            & out["roc"].gt(0)
            & out["previous_acceleration"].ge(0)
            & out["acceleration"].lt(0)
            & out["close_change"].lt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeParams,
    ) -> SignalFrame:
        entries = np.select(
            [indicators["long_entry"].fillna(False).to_numpy(dtype=bool),
             indicators["short_entry"].fillna(False).to_numpy(dtype=bool)],
            [1, -1], default=0,
        )
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            # Yesterday's decision takes effect at today's open.
            desired = decisions[i - 1] if i else 0
            if desired != side:
                side = int(desired)
                if side:
                    entry_price = opens[i]
                    entry_bar = i
            if side:
                valid_price = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                gain = side * (closes[i] / entry_price - 1.0) if valid_price and valid_close else np.nan
                # Exit decisions execute next open, after at most four sessions.
                exit_now = (
                    not valid_price
                    or gain >= params.profit_target
                    or i - entry_bar + 1 >= 4
                )
                decisions[i] = 0 if exit_now else side
            elif np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = int(entries[i])
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"long_entry": [True, False, False, False, False, False, False], "short_entry": False}, index=index)
    strategy = GeneratedStrategy()
    params = RangeParams()
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 102.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    entries["long_entry"] = False
    entries.loc[index[0], "short_entry"] = True
    data.loc[index[2], "close"] = 98.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
