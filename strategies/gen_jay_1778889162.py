from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTideParams:
    range_rank_window: int = 20
    compression_thresh: float = 0.30
    trough_lookback: int = 5
    velocity_window: int = 3
    close_pos_thresh: float = 0.55
    hold_bars: int = 4
    base_size: float = 0.40
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeTideParams]):
    strategy_id = "gen_jay_1778889162"

    @classmethod
    def params_type(cls):
        return RangeTideParams

    @staticmethod
    def warmup_bars(params: RangeTideParams) -> int:
        return params.range_rank_window + params.trough_lookback + params.velocity_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeTideParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        bar_range = data["high"] - data["low"]

        # Rolling percentile rank of bar range
        ind["range_rank"] = bar_range.rolling(params.range_rank_window).rank(pct=True)

        # Rate of change of range_rank: positive = range expanding
        ind["range_velocity"] = ind["range_rank"].diff(params.velocity_window)

        # Rolling minimum of range_rank to detect a recent trough
        ind["trough_min"] = ind["range_rank"].rolling(params.trough_lookback).min()

        # Close position within bar: 0 = at low, 1 = at high
        safe_range = bar_range.where(bar_range > 0, np.nan)
        ind["close_pos"] = (data["close"] - data["low"]) / safe_range

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeTideParams,
    ) -> SignalFrame:
        ind = indicators
        n = len(data)
        signal_arr = np.zeros(n, dtype=int)
        size_arr = np.full(n, params.base_size)

        had_trough = (ind["trough_min"] < params.compression_thresh).fillna(False).to_numpy()
        expanding = (ind["range_velocity"] > 0).fillna(False).to_numpy()
        bullish = (ind["close_pos"] >= params.close_pos_thresh).fillna(False).to_numpy()
        raw_entry = had_trough & expanding & bullish

        # Velocity-scaled size: stronger expansion force => larger position
        vel = ind["range_velocity"].clip(0, 1).fillna(0).to_numpy()
        scaled_size = params.base_size + (params.max_size - params.base_size) * vel

        in_trade = False
        entry_bar = -1
        entry_size = params.base_size

        for i in range(n):
            if in_trade:
                if i - entry_bar >= params.hold_bars:
                    signal_arr[i] = 0
                    in_trade = False
                else:
                    signal_arr[i] = 1
                    size_arr[i] = entry_size
            else:
                if raw_entry[i]:
                    signal_arr[i] = 1
                    entry_size = float(scaled_size[i])
                    size_arr[i] = entry_size
                    in_trade = True
                    entry_bar = i

        df = pd.DataFrame({"signal": signal_arr, "size": size_arr}, index=data.index)

        # Mandatory 1-bar shift: decide on bar N close, fill at bar N+1 open
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.base_size)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
