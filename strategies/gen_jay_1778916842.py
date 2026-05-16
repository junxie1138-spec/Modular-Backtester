from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTideParams:
    fast_range_ema: int = 3
    slow_range_ema: int = 7
    intraday_threshold: float = 0.60
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy["RangeTideParams"]):
    strategy_id = "gen_jay_1778916842"

    @classmethod
    def params_type(cls):
        return RangeTideParams

    @staticmethod
    def warmup_bars(params: RangeTideParams) -> int:
        return min(params.slow_range_ema + 1, 10)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeTideParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        close_safe = data["close"].replace(0, np.nan)
        norm_range = (data["high"] - data["low"]) / close_safe
        ind["norm_range"] = norm_range

        # Fast and slow exponential tides of normalized range
        ind["fast_tide"] = norm_range.ewm(span=params.fast_range_ema, adjust=False).mean()
        ind["slow_tide"] = norm_range.ewm(span=params.slow_range_ema, adjust=False).mean()

        # Intraday position: fraction of day's range where close landed
        hl = (data["high"] - data["low"]).replace(0, np.nan)
        ind["intraday_pos"] = (data["close"] - data["low"]) / hl

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeTideParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)

        # Phase lock: fast range tide below slow range tide (compression node)
        phase_lock = (indicators["fast_tide"] < indicators["slow_tide"]).fillna(False).values
        # Bullish absorption: close in upper portion of the bar's range
        bullish_pos = (indicators["intraday_pos"] >= params.intraday_threshold).fillna(False).values
        entry = phase_lock & bullish_pos

        # Fixed-bar exit: hold exactly hold_bars bars, no early exit
        in_until = -1
        for i in range(n):
            if i > in_until and entry[i]:
                end = min(i + params.hold_bars, n)
                signal[i:end] = 1
                in_until = end - 1

        # Mandatory 1-bar shift: decision on bar N executes on bar N+1
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
