from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeWeightedCLVParams:
    channel_window: int = 20
    smooth_window: int = 3
    entry_threshold: float = 0.08
    exit_bars: int = 7
    max_size_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[RangeWeightedCLVParams]):
    strategy_id = "gen_a1_1778904684"

    @classmethod
    def params_type(cls):
        return RangeWeightedCLVParams

    def warmup_bars(self, params):
        return int(max(1, params.channel_window) + max(1, params.smooth_window) + 2)

    def indicators(self, data, params):
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        rng = high - low
        safe_rng = rng.where(rng > 0.0, np.nan)
        # Intrabar close-location value in [0, 1]; neutral 0.5 on zero-range bars.
        clv = (close - low) / safe_rng
        clv = clv.clip(lower=0.0, upper=1.0).fillna(0.5)

        w = max(1, int(params.channel_window))
        sm = max(1, int(params.smooth_window))

        rng_filled = rng.clip(lower=0.0).fillna(0.0)
        weighted = rng_filled * clv
        num = weighted.rolling(w, min_periods=w).sum()
        den = rng_filled.rolling(w, min_periods=w).sum()
        den = den.where(den > 0.0, np.nan)
        # Range-weighted average close-location: wide-range bars dominate.
        rwclv = num / den
        rwclv = rwclv.fillna(0.5)
        rwclv_s = rwclv.rolling(sm, min_periods=sm).mean()

        out = pd.DataFrame(index=data.index)
        out["rng"] = rng_filled
        out["clv"] = clv
        out["rwclv"] = rwclv
        out["rwclv_s"] = rwclv_s
        return out

    def generate_signals(self, data, indicators, ctx, params):
        idx = data.index
        n = len(idx)

        rwclv_s = indicators["rwclv_s"].to_numpy(dtype=float)

        thr = float(params.entry_threshold)
        exit_bars = max(1, int(params.exit_bars))
        max_mult = max(1.0, float(params.max_size_mult))
        span = max(1e-6, 0.5 - thr)

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        pos = 0
        held = 0
        cur_size = 1.0
        for i in range(n):
            if pos != 0:
                held += 1
                if held >= exit_bars:
                    # Fixed-bar exit: flatten exactly exit_bars after entry.
                    pos = 0
                    held = 0
                else:
                    signal[i] = pos
                    size[i] = cur_size
                    continue

            val = rwclv_s[i]
            if np.isnan(val):
                continue
            d = val - 0.5
            if d > thr:
                direction = 1
            elif d < -thr:
                direction = -1
            else:
                continue

            # Signal-scaled position sizing: stronger imbalance -> larger size.
            norm = min(1.0, max(0.0, (abs(d) - thr) / span))
            cur_size = 1.0 + (max_mult - 1.0) * norm
            pos = direction
            held = 0
            signal[i] = direction
            size[i] = cur_size

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size
        # Decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
