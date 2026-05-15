from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    efficiency_threshold: float = 0.35
    streak_required: int = 3
    ma_period: int = 200
    atr_period: int = 14
    breakeven_pct: float = 0.015
    trail_atr_mult: float = 2.0
    position_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778885450"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.ma_period + params.atr_period + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        ind["ma200"] = close.rolling(params.ma_period, min_periods=params.ma_period).mean()

        total_range = (high - low).where((high - low) > 0, np.nan)
        ind["bar_efficiency"] = (close - open_) / total_range

        positive_flow = (ind["bar_efficiency"] > params.efficiency_threshold).astype(int)
        streak_groups = positive_flow.ne(positive_flow.shift(1)).cumsum()
        ind["efficiency_streak"] = positive_flow.groupby(streak_groups).cumsum().astype(int)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.ewm(alpha=1.0 / params.atr_period, adjust=False).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        ma200 = indicators["ma200"].values
        streak = indicators["efficiency_streak"].values
        atr = indicators["atr"].values

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, params.position_size, dtype=float)

        in_position = False
        entry_price = 0.0
        stop_price = 0.0
        high_water = 0.0
        breakeven_triggered = False

        for i in range(1, n):
            if np.isnan(ma200[i]) or np.isnan(atr[i]):
                signal[i] = 0
                continue

            if in_position:
                if close[i] > high_water:
                    high_water = close[i]

                if not breakeven_triggered and close[i] >= entry_price * (1.0 + params.breakeven_pct):
                    breakeven_triggered = True
                    if entry_price > stop_price:
                        stop_price = entry_price

                trail_stop = high_water - params.trail_atr_mult * atr[i]
                if trail_stop > stop_price:
                    stop_price = trail_stop

                if close[i] <= stop_price:
                    signal[i] = 0
                    in_position = False
                    entry_price = 0.0
                    stop_price = 0.0
                    high_water = 0.0
                    breakeven_triggered = False
                else:
                    signal[i] = 1

            else:
                above_regime = close[i] > ma200[i]
                streak_ok = streak[i] >= params.streak_required

                if above_regime and streak_ok:
                    signal[i] = 1
                    in_position = True
                    entry_price = close[i]
                    high_water = close[i]
                    stop_price = entry_price - params.trail_atr_mult * atr[i]
                    breakeven_triggered = False
                else:
                    signal[i] = 0

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
