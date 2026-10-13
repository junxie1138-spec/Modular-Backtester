from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    lookback: int = 20
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local_1790714801"

    @classmethod
    def params_type(cls):
        return GapParams

    @classmethod
    def warmup_bars(cls, params: GapParams) -> int:
        return params.lookback + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        gap = data["open"] / data["close"].shift(1) - 1.0
        gap_autocorr = gap.rolling(params.lookback).corr(gap.shift(1))
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        return pd.DataFrame({"gap": gap, "gap_autocorr": gap_autocorr, "atr": atr}, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapParams) -> SignalFrame:
        gap = indicators["gap"]
        corr = indicators["gap_autocorr"]
        raw = pd.Series(0, index=data.index, dtype=int)
        ready = gap.notna() & corr.notna()
        raw.loc[ready & (corr > 0.10) & (gap > 0)] = 1
        raw.loc[ready & (corr > 0.10) & (gap < 0)] = -1

        # Lock the entry ATR stop at the signal bar; evaluate it on subsequent closes.
        signal = raw.to_numpy(dtype=int)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        stop_atr = params.stop_atr
        position = 0
        stop = np.nan
        for i in range(len(signal)):
            if position and ((position > 0 and close[i] < stop) or (position < 0 and close[i] > stop)):
                signal[i] = 0
                position = 0
            elif position and signal[i] == 0:
                position = 0
            if not position and signal[i] != 0 and np.isfinite(atr[i]):
                position = signal[i]
                stop = close[i] - position * stop_atr * atr[i]

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
