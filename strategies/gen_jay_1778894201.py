from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_window: int = 20
    atr_window: int = 14
    atr_stop_k: float = 2.0
    tide_phase_hi: float = 0.25
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778894201"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.vol_window, params.atr_window) + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        ind["vol_median"] = data["volume"].rolling(params.vol_window).median()
        ind["vol_above_med"] = (data["volume"] > ind["vol_median"]).astype(float)

        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window).mean()

        month_period = data.index.to_period("M")
        monthly_rank = data.groupby(month_period).cumcount()
        monthly_count = data.groupby(month_period)["close"].transform("count")
        denom = (monthly_count - 1).replace(0, 1)
        ind["tide_phase"] = monthly_rank / denom

        ind["in_tide"] = (
            (ind["tide_phase"] <= params.tide_phase_hi) |
            (ind["tide_phase"] >= (1.0 - params.tide_phase_hi))
        ).astype(float)

        up_bar = (data["close"] > data["open"]).astype(float)
        vol_2bar = ind["vol_above_med"] * ind["vol_above_med"].shift(1)
        up_2bar = up_bar * up_bar.shift(1)
        ind["two_bar_confirm"] = vol_2bar * up_2bar

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        close = data["close"].values
        atr = indicators["atr"].values
        in_tide = indicators["in_tide"].values
        two_bar_confirm = indicators["two_bar_confirm"].values
        n = len(close)

        raw_signal = np.zeros(n, dtype=int)
        in_trade = False
        stop_level = np.nan
        hold_bars = 0

        for i in range(n):
            if in_trade:
                hold_bars += 1
                if close[i] < stop_level or hold_bars >= params.max_hold:
                    raw_signal[i] = 0
                    in_trade = False
                else:
                    raw_signal[i] = 1
            else:
                if (
                    in_tide[i] == 1.0
                    and two_bar_confirm[i] == 1.0
                    and not np.isnan(atr[i])
                    and atr[i] > 0.0
                ):
                    raw_signal[i] = 1
                    in_trade = True
                    stop_level = close[i] - params.atr_stop_k * atr[i]
                    hold_bars = 0

        df["signal"] = raw_signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
