from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_threshold: float = 0.003
    infection_lookback: int = 20
    infection_min_rate: float = 0.25
    growth_lookback: int = 5
    ma_period: int = 200
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778890424"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.ma_period + params.infection_lookback + params.growth_lookback + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        ind["gap"] = (data["open"] - prev_close) / prev_close

        ind["is_infected"] = (ind["gap"] >= params.gap_threshold).astype(float)
        ind["infection_rate"] = ind["is_infected"].rolling(
            params.infection_lookback, min_periods=params.infection_lookback
        ).mean()
        ind["rate_growth"] = (
            ind["infection_rate"] - ind["infection_rate"].shift(params.growth_lookback)
        )

        ind["ma200"] = data["close"].rolling(
            params.ma_period, min_periods=params.ma_period
        ).mean()
        ind["in_regime"] = (data["close"] > ind["ma200"]).astype(int)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0

        in_regime = indicators["in_regime"] == 1
        big_gap = indicators["gap"] >= params.gap_threshold
        epidemic_active = indicators["infection_rate"] >= params.infection_min_rate
        epidemic_growing = indicators["rate_growth"] > 0

        raw_entry = (in_regime & big_gap & epidemic_active & epidemic_growing).fillna(False)

        signal_arr = np.zeros(len(df), dtype=int)
        bars_remaining = 0

        for i in range(len(df)):
            if bars_remaining > 0:
                signal_arr[i] = 1
                bars_remaining -= 1
            elif raw_entry.iloc[i]:
                signal_arr[i] = 1
                bars_remaining = params.hold_bars - 1

        df["signal"] = signal_arr

        roll_max = indicators["infection_rate"].rolling(60, min_periods=20).max()
        rel_rate = (indicators["infection_rate"] / roll_max).clip(0.3, 1.0)
        df["size"] = rel_rate.fillna(1.0)

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
