from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnTailParams:
    return_window: int = 126
    tail_percentile: float = 0.05
    breakeven_pct: float = 0.015
    trail_atr_mult: float = 2.0
    atr_window: int = 14
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[ReturnTailParams]):
    strategy_id = "gen_local_1790714492"

    @classmethod
    def params_type(cls):
        return ReturnTailParams

    @classmethod
    def warmup_bars(cls, params: ReturnTailParams) -> int:
        return max(params.return_window + 1, params.atr_window)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: ReturnTailParams) -> pd.DataFrame:
        ret = data["close"].pct_change()
        threshold = ret.rolling(params.return_window, min_periods=params.return_window).quantile(params.tail_percentile)
        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [data["high"] - data["low"], (data["high"] - prev_close).abs(), (data["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return pd.DataFrame({"return": ret, "tail_threshold": threshold, "atr": atr}, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReturnTailParams) -> SignalFrame:
        ret = indicators["return"]
        threshold = indicators["tail_threshold"]
        setup = (ret.shift(1) <= threshold.shift(1)) & (ret > 0)
        setup = setup.fillna(False).to_numpy(dtype=bool)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        high_water = 0.0
        stop = 0.0
        breakeven_done = False
        for i in range(len(data)):
            if position == 1:
                high_water = max(high_water, high[i])
                if not breakeven_done and high_water >= entry_price * (1.0 + params.breakeven_pct):
                    breakeven_done = True
                    stop = entry_price
                if breakeven_done and np.isfinite(atr[i]):
                    stop = max(stop, high_water - params.trail_atr_mult * atr[i])
                if low[i] <= stop and breakeven_done:
                    position = 0
                    entry_price = high_water = stop = 0.0
                    breakeven_done = False
            if position == 0 and setup[i] and np.isfinite(atr[i]):
                position = 1
                entry_price = close[i]
                high_water = high[i]
                stop = 0.0
                breakeven_done = False
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
