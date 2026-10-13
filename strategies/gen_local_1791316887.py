from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRejectionParams:
    volume_lookback: int = 20
    volume_multiple: float = 1.1
    minimum_gap: float = 0.004
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapRejectionParams]):
    strategy_id = "gen_local_1791316887"

    @classmethod
    def params_type(cls) -> type[GapRejectionParams]:
        return GapRejectionParams

    @staticmethod
    def warmup_bars(params: GapRejectionParams) -> int:
        return int(params.volume_lookback) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRejectionParams) -> pd.DataFrame:
        if params.volume_lookback < 2 or params.volume_multiple <= 0 or params.minimum_gap <= 0:
            raise ValueError("Volume lookback must be at least two; thresholds must be positive.")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0, np.nan)
        previous_high = data["high"].shift(1)
        previous_low = data["low"].shift(1)
        out["gap"] = (data["open"] / previous_close - 1).replace([np.inf, -np.inf], np.nan)
        out["volume_baseline"] = data["volume"].rolling(params.volume_lookback, min_periods=params.volume_lookback).mean().shift(1)
        out["relative_volume"] = (data["volume"] / out["volume_baseline"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["volume_confirmed"] = out["relative_volume"].ge(params.volume_multiple).fillna(False)
        out["up_gap_rejected"] = (
            out["gap"].ge(params.minimum_gap)
            & data["open"].gt(previous_high)
            & data["close"].lt(previous_high)
            & data["close"].gt(previous_low)
            & data["close"].lt(data["open"])
            & out["volume_confirmed"]
        ).fillna(False)
        out["down_gap_rejected"] = (
            out["gap"].le(-params.minimum_gap)
            & data["open"].lt(previous_low)
            & data["close"].gt(previous_low)
            & data["close"].lt(previous_high)
            & data["close"].gt(data["open"])
            & out["volume_confirmed"]
        ).fillna(False)
        out["long_entry"] = (
            out["down_gap_rejected"].shift(1, fill_value=False)
            & data["close"].gt(data["close"].shift(1))
            & data["close"].gt(data["open"])
            & out["volume_confirmed"]
        ).fillna(False)
        out["short_entry"] = (
            out["up_gap_rejected"].shift(1, fill_value=False)
            & data["close"].lt(data["close"].shift(1))
            & data["close"].lt(data["open"])
            & out["volume_confirmed"]
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRejectionParams) -> SignalFrame:
        if params.profit_target <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Profit target must be positive; hold_bars must be between three and five.")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position and entry_bar == -1:
                # The preceding close's decision fills at this session's open.
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    position = 0
                    continue
                entry_price = opens[i]
                entry_bar = i
            if position:
                gain = position * (closes[i] / entry_price - 1)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_stop = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_stop:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
            elif longs[i] != shorts[i]:
                position = 1 if longs[i] else -1
                entry_bar = -1
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"open": 100.0, "close": [100.0, 102.0, 100.0, 100.0, 100.0, 100.0]}, index=index)
    entries = pd.DataFrame({"long_entry": [True, False, False, False, False, False], "short_entry": False}, index=index)
    result = GeneratedStrategy.generate_signals(prices, entries, None, GapRejectionParams())
    assert result.data["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(prices, entries, None, GapRejectionParams(profit_target=0.5))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    prices["close"] = [100.0, 98.0, 100.0, 100.0, 100.0, 100.0]
    entries["short_entry"] = entries["long_entry"]
    entries["long_entry"] = False
    result = GeneratedStrategy.generate_signals(prices, entries, None, GapRejectionParams())
    assert result.data["signal"].tolist() == [0, -1, 0, 0, 0, 0]
