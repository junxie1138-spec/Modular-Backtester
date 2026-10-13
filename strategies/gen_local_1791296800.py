from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuarterTurnParams:
    rank_lookback: int = 252
    return_days: int = 10
    rank_cutoff: float = 0.20
    entry_window: int = 7
    atr_period: int = 14
    trail_atr: float = 2.8
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[QuarterTurnParams]):
    strategy_id = "gen_local_1791296800"

    @classmethod
    def params_type(cls) -> type[QuarterTurnParams]:
        return QuarterTurnParams

    @classmethod
    def warmup_bars(cls, params: QuarterTurnParams) -> int:
        return max(params.return_days + params.rank_lookback, params.atr_period + 1)

    def indicators(self, data: pd.DataFrame, params: QuarterTurnParams) -> pd.DataFrame:
        close = data["close"]
        quarter = pd.Series(data.index.to_period("Q"), index=data.index)
        quarter_start = quarter.ne(quarter.shift(1))
        trailing_return = (close / close.shift(params.return_days) - 1).replace([np.inf, -np.inf], np.nan)
        rank = trailing_return.rolling(params.rank_lookback, min_periods=params.rank_lookback).rank(pct=True)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)

        result = pd.DataFrame(index=data.index)
        result["quarter_age"] = quarter.groupby(quarter).cumcount()
        result["prior_quarter_rank"] = rank.shift(1).where(quarter_start).ffill().fillna(1.0)
        result["two_up"] = close.diff().gt(0) & close.diff().shift(1).gt(0)
        result["atr"] = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean().fillna(0.0)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QuarterTurnParams,
    ) -> SignalFrame:
        target = np.zeros(len(data), dtype=int)
        entered_this_quarter = False
        held = 0
        high_water = 0.0
        stop = -np.inf

        for i in range(len(data)):
            age = int(indicators["quarter_age"].iloc[i])
            if age == 0:
                entered_this_quarter = False

            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            active = i > 0 and target[i - 1] == 1

            if active:
                if held == 0:
                    high_water = close
                    stop = -np.inf
                held += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close >= stop and held < params.hold_bars:
                    target[i] = 1
            else:
                held = 0
                if (
                    not entered_this_quarter
                    and 2 <= age < params.entry_window
                    and indicators["prior_quarter_rank"].iloc[i] <= params.rank_cutoff
                    and bool(indicators["two_up"].iloc[i])
                    and atr > 0
                ):
                    target[i] = 1
                    entered_this_quarter = True

        df = pd.DataFrame({"signal": target, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
