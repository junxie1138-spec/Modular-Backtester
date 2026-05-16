from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ext_window: int = 20
    entry_thr: float = 0.35
    atr_window: int = 14
    sma_window: int = 100
    init_stop_atr: float = 2.0
    be_trigger_pct: float = 0.02
    trail_atr_k: float = 2.5
    size_floor: float = 0.5
    size_ceil: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778907385"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.ext_window, params.atr_window, params.sma_window)) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # High-low range dynamics: how far the envelope boundary moved each bar.
        up_ext = (high - prev_high).clip(lower=0.0)
        dn_ext = (prev_low - low).clip(lower=0.0)

        w = max(int(params.ext_window), 1)
        up_sum = up_ext.rolling(w, min_periods=w).sum()
        dn_sum = dn_ext.rolling(w, min_periods=w).sum()
        total = up_sum + dn_sum

        # Net directional extension over total extension: a signal-to-noise ratio in [-1, 1].
        ratio = np.where(total.values > 0.0,
                         (up_sum.values - dn_sum.values) / np.where(total.values > 0.0, total.values, 1.0),
                         0.0)
        ratio = pd.Series(ratio, index=data.index).fillna(0.0)

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        aw = max(int(params.atr_window), 1)
        atr = tr.rolling(aw, min_periods=aw).mean()

        sw = max(int(params.sma_window), 1)
        sma = close.rolling(sw, min_periods=sw).mean()

        out = pd.DataFrame(index=data.index)
        out["ratio"] = ratio
        out["atr"] = atr
        out["sma"] = sma
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        ratio = indicators["ratio"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)

        sig = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        thr = float(params.entry_thr)
        denom = max(1e-9, 1.0 - thr)
        floor = float(params.size_floor)
        ceil = float(params.size_ceil)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        peak = 0.0
        be_armed = False
        entry_size = 1.0

        for i in range(n):
            a = atr[i]
            s = sma[i]
            c = close[i]

            if not in_pos:
                valid = (not np.isnan(a)) and (not np.isnan(s)) and (a > 0.0)
                if valid and ratio[i] >= thr and c > s:
                    in_pos = True
                    entry_price = c
                    peak = c
                    stop = c - float(params.init_stop_atr) * a
                    be_armed = False
                    strength = (ratio[i] - thr) / denom
                    if strength < 0.0:
                        strength = 0.0
                    elif strength > 1.0:
                        strength = 1.0
                    # Signal-scaled position sizing: stronger trend SNR -> larger size.
                    entry_size = floor + (ceil - floor) * strength
                    sig[i] = 1
                    size[i] = entry_size
            else:
                if c > peak:
                    peak = c
                # Breakeven-then-trail: arm at +X%, lock stop to entry, then trail by k*ATR.
                if (not be_armed) and c >= entry_price * (1.0 + float(params.be_trigger_pct)):
                    if entry_price > stop:
                        stop = entry_price
                    be_armed = True
                if be_armed and (not np.isnan(a)):
                    trail = peak - float(params.trail_atr_k) * a
                    if trail > stop:
                        stop = trail
                if c <= stop:
                    in_pos = False
                    sig[i] = 0
                    size[i] = 1.0
                else:
                    sig[i] = 1
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
