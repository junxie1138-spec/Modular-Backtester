from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

_COMPRESSION_WINDOW = 20
_ATR_PERIOD = 14


@dataclass(slots=True)
class GeneratedParams:
    streak_threshold: int = 3
    atr_stop_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778888768"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # rolling median over _COMPRESSION_WINDOW, plus one bar for prev-close TR
        return _COMPRESSION_WINDOW + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(_ATR_PERIOD, min_periods=_ATR_PERIOD).mean()

        bar_range = (high - low).clip(lower=0.0)
        range_median = bar_range.rolling(
            _COMPRESSION_WINDOW, min_periods=_COMPRESSION_WINDOW
        ).median()

        # a bar is 'compressed' when its range is below the rolling median range
        compressed = (bar_range < range_median).fillna(False)

        # consecutive-streak count: length of the current unbroken run of
        # compressed bars (0 on any non-compressed bar)
        group = (~compressed).cumsum()
        streak = compressed.groupby(group).cumcount() + 1
        streak = streak.where(compressed, 0).astype(float)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["streak"] = streak
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        n = len(close)

        raw = np.zeros(n, dtype=np.int64)
        thr = float(params.streak_threshold)
        k = float(params.atr_stop_mult)

        in_pos = False
        hwm = 0.0  # in-trade highest close (water mark)

        for i in range(n):
            a = atr[i]
            if not in_pos:
                # enter once the compression streak crosses the threshold
                if (
                    np.isfinite(streak[i])
                    and streak[i] >= thr
                    and np.isfinite(a)
                    and a > 0.0
                    and np.isfinite(close[i])
                ):
                    in_pos = True
                    hwm = close[i]
                    raw[i] = 1
            else:
                # ratchet the high-water mark up, never down
                if np.isfinite(close[i]) and close[i] > hwm:
                    hwm = close[i]
                # rolling-high ATR trailing stop
                if not np.isfinite(a) or a <= 0.0:
                    in_pos = False
                    raw[i] = 0
                else:
                    stop = hwm - k * a
                    if close[i] < stop:
                        in_pos = False
                        raw[i] = 0
                    else:
                        raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
