from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_window: int = 20
    prey_count: int = 5
    prey_threshold: float = 0.75
    expansion_factor: float = 1.4
    close_eff_entry: float = 0.60
    close_eff_confirm: float = 0.55
    atr_window: int = 14
    atr_multiplier: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778885798"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.range_window + params.prey_count + 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        hl_range = high - low
        median_range = hl_range.rolling(params.range_window).median()

        prev_close = close.shift(1)
        tr = pd.concat([
            hl_range,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        bar_size = hl_range.replace(0, np.nan)
        close_eff = (close - low) / bar_size

        is_prey = (
            hl_range < params.prey_threshold * median_range
        ).fillna(False).astype(float)
        prey_streak = is_prey.rolling(params.prey_count).sum()

        is_trigger = (
            (hl_range > params.expansion_factor * median_range) &
            (close_eff > params.close_eff_entry)
        ).fillna(False).astype(float)

        ind = pd.DataFrame(index=data.index)
        ind["median_range"] = median_range
        ind["atr"] = atr
        ind["close_eff"] = close_eff
        ind["prey_streak"] = prey_streak
        ind["is_trigger"] = is_trigger
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        low_vals = data["low"].values
        n = len(close)

        atr = indicators["atr"].values
        close_eff = indicators["close_eff"].values
        prey_streak = indicators["prey_streak"].values
        is_trigger = indicators["is_trigger"].values

        signal = np.zeros(n, dtype=int)

        in_trade = False
        hwm = 0.0
        stop = 0.0
        bar1_ok = False
        bar1_low = 0.0

        for i in range(n):
            if np.isnan(atr[i]) or np.isnan(prey_streak[i]):
                bar1_ok = False
                continue

            if in_trade:
                if close[i] > hwm:
                    hwm = close[i]
                    stop = hwm - params.atr_multiplier * atr[i]
                if close[i] < stop:
                    signal[i] = 0
                    in_trade = False
                    bar1_ok = False
                else:
                    signal[i] = 1
            else:
                eff = close_eff[i]
                eff_valid = not np.isnan(eff)
                if bar1_ok:
                    if eff_valid and eff >= params.close_eff_confirm and close[i] > bar1_low:
                        signal[i] = 1
                        in_trade = True
                        hwm = close[i]
                        stop = hwm - params.atr_multiplier * atr[i]
                    bar1_ok = False
                else:
                    if (
                        eff_valid
                        and prey_streak[i] >= params.prey_count
                        and is_trigger[i] == 1.0
                    ):
                        bar1_ok = True
                        bar1_low = low_vals[i]

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
