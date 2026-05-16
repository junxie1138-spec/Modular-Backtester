from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    hold_bars: int = 15


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778916213"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # rolling(lb) then rolling(lb).rank() stacks two lb-length windows
        return params.lookback * 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        lb = params.lookback

        roll_high = data["high"].rolling(lb, min_periods=lb).max()
        roll_low = data["low"].rolling(lb, min_periods=lb).min()
        roll_span = roll_high - roll_low

        # Normalized range size: spring compression metric
        norm_range = roll_span / data["close"]

        # Close relative position within rolling range [0=at low, 1=at high]
        close_pos = (data["close"] - roll_low) / roll_span.where(roll_span > 0, np.nan)

        # Percentile rank of current range vs. its own recent lb-bar history
        # Low value means range is historically compressed (spring coiled)
        range_pct = norm_range.rolling(lb, min_periods=lb).rank(pct=True)

        # Percentile rank of close position vs. its own recent lb-bar history
        # Low value means close is historically near the floor of the range (tension loaded)
        close_pos_pct = close_pos.rolling(lb, min_periods=lb).rank(pct=True)

        ind["range_pct"] = range_pct
        ind["close_pos_pct"] = close_pos_pct

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        hold = params.hold_bars

        # Joint spring-tension entry: range compressed AND close near range floor
        entry_mask = (
            (indicators["range_pct"] < 0.35) & (indicators["close_pos_pct"] < 0.35)
        ).fillna(False)

        raw_signal = np.zeros(n, dtype=int)
        in_trade = False
        bars_held = 0

        for i in range(n):
            if in_trade:
                if bars_held >= hold:
                    in_trade = False
                    bars_held = 0
                    # raw_signal[i] stays 0: position exits
                else:
                    raw_signal[i] = 1
                    bars_held += 1
            else:
                if entry_mask.iloc[i]:
                    raw_signal[i] = 1
                    in_trade = True
                    bars_held = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index)
        # Shift by one bar: decision at bar N close, fill at bar N+1 open
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
