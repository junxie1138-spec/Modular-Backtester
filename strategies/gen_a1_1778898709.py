from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeContagionParams:
    range_window: int = 20
    prevalence_threshold: float = 0.30
    hold_bars: int = 4
    ma_window: int = 200
    conviction_scale: bool = True
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[RangeContagionParams]):
    strategy_id = "gen_a1_1778898709"

    @classmethod
    def params_type(cls):
        return RangeContagionParams

    @staticmethod
    def warmup_bars(params: RangeContagionParams) -> int:
        # ma needs ma_window bars; prevalence needs a shift(1) plus a rolling
        # window plus a further shift(1) for the lag series -> range_window + 2.
        return int(max(int(params.ma_window), int(params.range_window) + 2))

    def indicators(self, data: pd.DataFrame, params: RangeContagionParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Range-translation event: the entire high-low interval shifts upward
        # (higher high AND higher low). A pure range-geometry momentum primitive.
        higher_high = high > high.shift(1)
        higher_low = low > low.shift(1)
        translate_up = (higher_high & higher_low).astype(float)

        # Prevalence = epidemic 'infected fraction': share of range-translation
        # bars across the lookback. A rising prevalence is the SI curve climbing.
        w = max(int(params.range_window), 2)
        prevalence = translate_up.rolling(window=w, min_periods=w).mean()

        mw = max(int(params.ma_window), 2)
        ma = close.rolling(window=mw, min_periods=mw).mean()

        ind["prevalence"] = prevalence
        ind["prev_lag"] = prevalence.shift(1)
        ind["regime"] = (close > ma).astype(float)
        return ind

    def generate_signals(self, data, indicators, ctx, params) -> SignalFrame:
        n = len(data)
        thr = float(params.prevalence_threshold)

        prevalence = np.nan_to_num(indicators["prevalence"].to_numpy(dtype=float), nan=0.0)
        prev_lag = np.nan_to_num(indicators["prev_lag"].to_numpy(dtype=float), nan=1.0)
        regime = np.nan_to_num(indicators["regime"].to_numpy(dtype=float), nan=0.0)

        # Fresh upward cross of the critical prevalence threshold, in an uptrend.
        entry = (prevalence >= thr) & (prev_lag < thr) & (regime > 0.5)

        strength = np.clip(prevalence - thr, 0.0, None)

        raw = np.zeros(n, dtype=int)
        size = np.full(n, float(params.base_size), dtype=float)
        hold = max(int(params.hold_bars), 1)

        # Fixed-bar exit: once entered, hold exactly `hold` bars then go flat.
        # No signal-based exit; entries during a held window are ignored.
        i = 0
        while i < n:
            if entry[i]:
                stop = min(i + hold, n)
                if params.conviction_scale:
                    mult = min(1.0 + 5.0 * float(strength[i]), 2.0)
                else:
                    mult = 1.0
                for j in range(i, stop):
                    raw[j] = 1
                    size[j] = float(params.base_size) * mult
                i = stop
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size, index=data.index).astype(float).clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
