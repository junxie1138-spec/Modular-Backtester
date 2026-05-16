from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_window: int = 14
    tension_window: int = 10
    tension_threshold: float = 1.5
    atr_stop_k: float = 1.5
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778905190"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.atr_window, params.tension_window) + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.ewm(span=params.atr_window, min_periods=params.atr_window).mean()

        gap = open_ - prev_close
        neg_gap_sum = gap.clip(upper=0).rolling(params.tension_window, min_periods=params.tension_window).sum()
        ind["tension_mag"] = (-neg_gap_sum / ind["atr"]).clip(lower=0)

        ind["bullish_bar"] = (close > open_).astype(float)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        atr = indicators["atr"].values
        tension_mag = indicators["tension_mag"].values
        bullish_bar = indicators["bullish_bar"].values

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_position = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(2, n):
            if in_position:
                bars_held += 1
                stop_level = entry_price - params.atr_stop_k * entry_atr
                if np.isnan(atr[i]) or close[i] < stop_level or bars_held >= params.max_hold_bars:
                    signal[i] = 0
                    in_position = False
                    bars_held = 0
                else:
                    signal[i] = 1
            else:
                if np.isnan(atr[i]) or np.isnan(tension_mag[i]):
                    continue
                two_bar_confirm = (bullish_bar[i] == 1.0) and (bullish_bar[i - 1] == 1.0)
                if two_bar_confirm and tension_mag[i] >= params.tension_threshold:
                    signal[i] = 1
                    in_position = True
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
