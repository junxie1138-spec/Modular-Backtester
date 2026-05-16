from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class NeapSpringParams:
    lookback: int = 20
    trail_mult: float = 2.5


class GeneratedStrategy(BaseStrategy["NeapSpringParams"]):
    strategy_id = "gen_jay_1778916649"

    @classmethod
    def params_type(cls):
        return NeapSpringParams

    @staticmethod
    def warmup_bars(params: NeapSpringParams) -> int:
        return params.lookback + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: NeapSpringParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        gap = data["open"] - data["close"].shift(1)
        ind["gap"] = gap
        ind["gap_cumsum"] = gap.rolling(params.lookback, min_periods=params.lookback).sum()

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.lookback, min_periods=params.lookback).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: NeapSpringParams,
    ) -> SignalFrame:
        close = data["close"].values
        gap = indicators["gap"].values
        gap_cumsum = indicators["gap_cumsum"].values
        atr = indicators["atr"].values
        n = len(close)

        ENTRY_GAP_THRESH = 0.3

        raw_signal = np.zeros(n, dtype=int)
        in_trade = False
        hwm = 0.0

        for i in range(1, n):
            g = gap[i]
            a = atr[i]

            if np.isnan(g) or np.isnan(a):
                if in_trade:
                    raw_signal[i] = 1
                continue

            if not in_trade:
                cs_prev = gap_cumsum[i - 1]
                if np.isnan(cs_prev):
                    continue
                if cs_prev < 0.0 and g > ENTRY_GAP_THRESH * a:
                    in_trade = True
                    hwm = close[i]
                    raw_signal[i] = 1
            else:
                hwm = max(hwm, close[i])
                if close[i] < hwm - a * params.trail_mult:
                    in_trade = False
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
