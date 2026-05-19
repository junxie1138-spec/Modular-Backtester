from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTensionParams:
    range_window: int = 10
    compression_window: int = 100
    compression_pct: float = 0.35
    tension_threshold: float = 0.55
    atr_window: int = 14
    trail_k: float = 1.8
    ma_window: int = 200
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GapTensionParams]):
    strategy_id = "gen_a2_1779152646"

    @classmethod
    def params_type(cls):
        return GapTensionParams

    @staticmethod
    def warmup_bars(params: GapTensionParams) -> int:
        return int(max(params.ma_window,
                       params.compression_window + params.range_window,
                       params.atr_window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapTensionParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        prev_close = close.shift(1)
        gap = open_ - prev_close

        intraday_range = (high - low).clip(lower=0.0)
        avg_range = intraday_range.rolling(params.range_window).mean()
        # avoid division by zero -> NaN, which comparisons treat as False
        avg_range_safe = avg_range.replace(0.0, np.nan)

        range_rank = avg_range.rolling(params.compression_window).rank(pct=True)

        tension = gap / avg_range_safe

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        ma = close.rolling(params.ma_window).mean()

        ind = pd.DataFrame(index=data.index)
        ind["gap"] = gap
        ind["avg_range"] = avg_range
        ind["range_rank"] = range_rank
        ind["tension"] = tension
        ind["atr"] = atr
        ind["ma"] = ma
        return ind

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapTensionParams) -> SignalFrame:
        close = data["close"]
        open_ = data["open"]
        ma = indicators["ma"]

        compressed = indicators["range_rank"] < params.compression_pct
        loaded = indicators["tension"] > params.tension_threshold
        confirm = close > open_
        regime = close > ma

        entry = (compressed & loaded & confirm & regime).fillna(False).to_numpy()

        close_arr = close.to_numpy(dtype=float)
        atr_arr = indicators["atr"].to_numpy(dtype=float)

        n = len(data)
        signal = np.zeros(n, dtype=np.int64)
        position = 0
        hwm = 0.0
        trail_k = float(params.trail_k)

        for i in range(n):
            if position == 0:
                if entry[i]:
                    position = 1
                    hwm = close_arr[i]
                    signal[i] = 1
            else:
                c = close_arr[i]
                if c > hwm:
                    hwm = c
                a = atr_arr[i]
                if not np.isnan(a) and c < hwm - trail_k * a:
                    position = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
