from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_period: int = 14
    range_window: int = 20
    range_pctile: float = 35.0
    min_streak: int = 2
    atr_stop_mult: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778911092"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.atr_period, params.range_window) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.ewm(span=params.atr_period, adjust=False).mean()

        bar_range = high - low
        ind["range_threshold"] = bar_range.rolling(params.range_window).quantile(
            params.range_pctile / 100.0
        )

        # Close position within bar: 0 = at low, 1 = at high; NaN when doji
        bar_range_safe = bar_range.where(bar_range > 0, other=np.nan)
        ind["close_pos"] = (close - low) / bar_range_safe

        # Plastic coil bar: range compressed AND close holds upper half
        compressed = bar_range < ind["range_threshold"]
        bullish_hold = ind["close_pos"] > 0.5
        coil_bar = (compressed & bullish_hold).astype(int)
        ind["coil_bar"] = coil_bar

        # Consecutive streak of coil bars
        groups = (coil_bar != coil_bar.shift()).cumsum()
        ind["streak"] = coil_bar.groupby(groups).cumsum() * coil_bar

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
        streak = indicators["streak"].values.astype(float)

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)

        in_trade = False
        stop_price = np.nan

        for i in range(n):
            if np.isnan(atr[i]):
                raw_signal[i] = 0
                continue

            if in_trade:
                if close[i] <= stop_price:
                    # Fixed volatility stop triggered; not trailing
                    raw_signal[i] = 0
                    in_trade = False
                    stop_price = np.nan
                else:
                    raw_signal[i] = 1
            else:
                # Two-bar confirmation: streak must reach min_streak (default 2)
                if streak[i] >= params.min_streak:
                    raw_signal[i] = 1
                    in_trade = True
                    # Fixed stop anchored to entry-decision close; never adjusts
                    stop_price = close[i] - params.atr_stop_mult * atr[i]
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame(
            {"signal": raw_signal, "size": np.full(n, 0.95)},
            index=data.index,
        )
        # Decide on bar N close; fill executes on bar N+1 open
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
