from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_threshold: int = 3
    atr_short: int = 5
    atr_long: int = 20
    range_expansion_threshold: float = 1.15
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778909554"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.atr_long + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_short = tr.rolling(params.atr_short, min_periods=params.atr_short).mean()
        atr_long = tr.rolling(params.atr_long, min_periods=params.atr_long).mean()

        # Predator-feeding intensity: recent bar energy vs historical baseline
        ind["range_expansion"] = atr_short / atr_long.replace(0.0, np.nan)

        # Consecutive close-direction streak via vectorised cumsum trick
        up = (close > prev_close)
        down = (close < prev_close)

        cumup = up.cumsum()
        last_non_up = cumup.where(~up).ffill().fillna(0)
        up_streak = (cumup - last_non_up).where(up, other=0)

        cumdn = down.cumsum()
        last_non_dn = cumdn.where(~down).ffill().fillna(0)
        dn_streak = (cumdn - last_non_dn).where(down, other=0)

        # Signed streak: positive = consecutive bullish bars, negative = consecutive bearish bars
        ind["streak"] = up_streak - dn_streak

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

        streak = indicators["streak"]
        range_exp = indicators["range_expansion"]

        # Primitive 1: streak count at threshold (prey at critical mass)
        # Primitive 2: ATR expansion confirms energy (predators actively feeding)
        bull = (streak >= params.streak_threshold) & (range_exp > params.range_expansion_threshold)
        bear = (streak <= -params.streak_threshold) & (range_exp > params.range_expansion_threshold)

        raw = pd.Series(0, index=data.index, dtype=int)
        raw[bull] = 1
        raw[bear] = -1

        # Fixed-bar exit: hold for exactly hold_bars bars, then go flat
        hold = params.hold_bars
        raw_arr = raw.values
        final = np.zeros(len(raw_arr), dtype=int)

        i = 0
        while i < len(raw_arr):
            if raw_arr[i] != 0:
                end = min(i + hold, len(raw_arr))
                final[i:end] = int(raw_arr[i])
                i = end
            else:
                i += 1

        df["signal"] = final
        # Mandatory 1-bar shift: decision on bar N fills on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
