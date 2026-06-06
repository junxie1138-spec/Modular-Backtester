from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    """Range-contagion breakout.

    The high-low range is treated as a volatility 'infection' level. Its lag-1
    autocorrelation over a rolling window measures whether infection is
    spreading (positive autocorrelation -> susceptible bars are being infected).
    When the contagion gate is open and the current bar is an outsized-range
    bar, the move is assumed to persist; we trade in the bar's close direction
    and exit a fixed number of bars later.
    """

    strategy_id = "gen_a2_1779146050"

    # Fixed (non-tunable) mechanism constants.
    _WIDE_MULT = 1.3   # range must exceed this multiple of its rolling mean.
    _SIZE_CAP = 3.0    # cap on epidemic-intensity position sizing.

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # rolling(window).corr(shift(1)) consumes window + 1 bars; pad by one.
        return int(params.window) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        w = max(int(params.window), 2)
        out = pd.DataFrame(index=data.index)

        rng = (data["high"] - data["low"]).astype(float).clip(lower=0.0)
        rng_mean = rng.rolling(w).mean()

        # Lag-1 autocorrelation of the high-low range: the contagion measure.
        range_autocorr = rng.rolling(w).corr(rng.shift(1))

        ret = data["close"].astype(float).pct_change()
        intensity = rng / rng_mean.replace(0.0, np.nan)

        out["rng"] = rng
        out["rng_mean"] = rng_mean
        out["range_autocorr"] = range_autocorr
        out["ret"] = ret
        out["intensity"] = intensity
        return out

    def generate_signals(self, data, indicators, ctx, params) -> SignalFrame:
        n = len(data)
        df = pd.DataFrame(index=data.index)

        autocorr = indicators["range_autocorr"].to_numpy(dtype=float)
        rng = indicators["rng"].to_numpy(dtype=float)
        rng_mean = indicators["rng_mean"].to_numpy(dtype=float)
        ret = indicators["ret"].to_numpy(dtype=float)
        intensity = indicators["intensity"].to_numpy(dtype=float)

        hold = max(int(params.hold_bars), 1)
        wide_mult = self._WIDE_MULT
        size_cap = self._SIZE_CAP

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        position = 0
        remaining = 0
        held_size = 1.0

        for i in range(n):
            # Inside an open trade: hold the position for exactly `hold` bars.
            if remaining > 0:
                signal[i] = position
                size[i] = held_size
                remaining -= 1
                if remaining == 0:
                    position = 0
                continue

            ac = autocorr[i]
            rg = rng[i]
            rm = rng_mean[i]
            rt = ret[i]
            it = intensity[i]

            valid = (
                np.isfinite(ac) and np.isfinite(rg) and np.isfinite(rm)
                and np.isfinite(rt) and rm > 0.0
            )
            # Contagion gate open, infection event, and a directional close.
            if valid and ac > 0.0 and rg > wide_mult * rm and rt != 0.0:
                direction = 1 if rt > 0.0 else -1
                sev = it if np.isfinite(it) else 1.0
                held_size = float(min(max(sev, 1.0), size_cap))
                position = direction
                remaining = hold
                signal[i] = position
                size[i] = held_size
                remaining -= 1
                if remaining == 0:
                    position = 0

        df["signal"] = signal
        df["size"] = size

        # MANDATORY one-bar shift: decide on bar N's close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
