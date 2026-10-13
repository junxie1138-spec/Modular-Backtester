from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PathAreaParams:
    signature_bars: int = 10
    area_gate: float = 0.08
    atr_bars: int = 14
    trail_atr: float = 2.5
    max_hold_bars: int = 12


class GeneratedStrategy(BaseStrategy[PathAreaParams]):
    strategy_id = "gen_local_1791224711"

    @classmethod
    def params_type(cls) -> type[PathAreaParams]:
        return PathAreaParams

    def warmup_bars(self, params: PathAreaParams) -> int:
        return max(params.signature_bars, params.atr_bars) + 1

    def indicators(self, data: pd.DataFrame, params: PathAreaParams) -> pd.DataFrame:
        x = np.log(data["close"].where(data["close"] > 0))
        y = np.log(data["volume"].where(data["volume"] > 0))
        w = params.signature_bars
        cross = x.shift(1) * y - y.shift(1) * x
        start_x, start_y = x.shift(w - 1), y.shift(w - 1)
        area = 0.5 * (cross.rolling(w - 1, min_periods=w - 1).sum()
                      - start_x * (y - start_y) + start_y * (x - start_x))
        span_x = x.rolling(w, min_periods=w).max() - x.rolling(w, min_periods=w).min()
        span_y = y.rolling(w, min_periods=w).max() - y.rolling(w, min_periods=w).min()
        scale = (span_x * span_y).replace(0, np.nan)
        previous_close = data["close"].shift(1)
        true_range = pd.concat((data["high"] - data["low"],
                                (data["high"] - previous_close).abs(),
                                (data["low"] - previous_close).abs()), axis=1).max(axis=1)
        result = pd.DataFrame(index=data.index)
        result["area"] = (area / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["displacement"] = (x - start_x).fillna(0.0)
        result["atr"] = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean().fillna(0.0)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PathAreaParams) -> SignalFrame:
        area = indicators["area"]
        displacement = indicators["displacement"]
        atr = indicators["atr"].to_numpy()
        close = data["close"].to_numpy()
        long_setup = ((area > params.area_gate) & (displacement > 0)
                      & (indicators["atr"] > 0))
        short_setup = ((area < -params.area_gate) & (displacement < 0)
                       & (indicators["atr"] > 0))
        long_entry = (long_setup & long_setup.shift(1, fill_value=False)).to_numpy()
        short_entry = (short_setup & short_setup.shift(1, fill_value=False)).to_numpy()
        target = np.zeros(len(data), dtype=int)
        position = 0
        extreme = stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if position == 1:
                extreme = max(extreme, close[i])
                stop = max(stop, extreme - params.trail_atr * atr[i])
                if close[i] <= stop or i - entry_bar >= params.max_hold_bars:
                    position = 0
            elif position == -1:
                extreme = min(extreme, close[i])
                stop = min(stop, extreme + params.trail_atr * atr[i])
                if close[i] >= stop or i - entry_bar >= params.max_hold_bars:
                    position = 0
            elif long_entry[i]:
                position, extreme = 1, close[i]
                stop = close[i] - params.trail_atr * atr[i]
                entry_bar = i
            elif short_entry[i]:
                position, extreme = -1, close[i]
                stop = close[i] + params.trail_atr * atr[i]
                entry_bar = i
            target[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
