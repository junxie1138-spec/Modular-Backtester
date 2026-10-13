from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 30
    efficiency_window: int = 5
    z_entry: float = 1.25
    efficiency_min: float = 0.55
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791297772"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.ma_window, params.efficiency_window + 1, params.atr_window)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        mean = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        deviation = close.rolling(params.ma_window, min_periods=params.ma_window).std()
        path = close.diff().abs().rolling(params.efficiency_window, min_periods=params.efficiency_window).sum()
        previous = close.shift(1)
        true_range = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["z"] = ((close - mean) / deviation.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["efficiency"] = (close.diff(params.efficiency_window).abs() / path.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        z = indicators["z"].to_numpy()
        efficiency = indicators["efficiency"].to_numpy()
        atr = indicators["atr"].to_numpy()
        position = 0
        stop = 0.0
        held = 0

        for i in range(len(data)):
            if i and raw[i - 1] != position:
                position = int(raw[i - 1])
                if position:
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]
                    held = 0
            if position:
                held += 1
                stopped = (position == 1 and closes[i] < stop) or (position == -1 and closes[i] > stop)
                raw[i] = 0 if stopped or held >= params.hold_bars else position
            elif atr[i] > 0 and efficiency[i] >= params.efficiency_min:
                raw[i] = 1 if z[i] >= params.z_entry else -1 if z[i] <= -params.z_entry else 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
