from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    autocorr_window: int = 20
    momentum_lookback: int = 10
    regime_threshold: float = 0.15
    mom_reference: float = 0.05
    profit_target: float = 0.04
    max_hold: int = 10
    base_size: float = 0.8
    size_scale: float = 1.4
    min_size: float = 0.5
    max_size: float = 2.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779145033"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(int(params.autocorr_window), int(params.momentum_lookback))) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()
        ret_lag = ret.shift(1)

        w = max(int(params.autocorr_window), 3)
        # Rolling lag-1 autocorrelation of close-to-close returns (the regime variable).
        autocorr = ret.rolling(w).corr(ret_lag)

        m = max(int(params.momentum_lookback), 1)
        momentum = close.pct_change(m)

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["autocorr"] = autocorr
        out["momentum"] = momentum
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        idx = data.index
        n = len(idx)

        close = data["close"].to_numpy(dtype=float)
        autocorr = indicators["autocorr"].to_numpy(dtype=float)
        momentum = indicators["momentum"].to_numpy(dtype=float)

        thr = float(params.regime_threshold)
        pt = float(params.profit_target)
        max_hold = max(int(params.max_hold), 1)
        mom_ref = max(float(params.mom_reference), 1e-6)
        base_size = float(params.base_size)
        size_scale = float(params.size_scale)
        min_size = float(params.min_size)
        max_size = float(params.max_size)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        position = 0
        entry_price = 0.0
        bars_held = 0
        entry_size = 1.0

        for i in range(n):
            ac = autocorr[i]
            mom = momentum[i]

            if position != 0:
                bars_held += 1
                px = close[i]
                if entry_price > 0.0:
                    if position > 0:
                        gain = px / entry_price - 1.0
                    else:
                        gain = 1.0 - px / entry_price
                else:
                    gain = 0.0
                # Exit: profit-target reached OR time-stop, whichever comes first.
                if (gain >= pt) or (bars_held >= max_hold):
                    signal[i] = 0
                    size[i] = 1.0
                    position = 0
                    entry_price = 0.0
                    bars_held = 0
                    entry_size = 1.0
                else:
                    signal[i] = position
                    size[i] = entry_size
                continue

            # Flat: classify regime from rolling return autocorrelation.
            direction = 0
            if np.isfinite(ac) and np.isfinite(mom) and mom != 0.0:
                if ac >= thr:
                    # Persistence regime: ride the prevailing momentum.
                    direction = 1 if mom > 0.0 else -1
                elif ac <= -thr:
                    # Reversal regime: fade the prevailing momentum.
                    direction = -1 if mom > 0.0 else 1

            if direction != 0:
                if thr > 0.0:
                    ac_strength = min(abs(ac) / (2.0 * thr), 1.0)
                else:
                    ac_strength = 1.0
                mom_strength = min(abs(mom) / mom_ref, 1.0)
                conviction = 0.5 * ac_strength + 0.5 * mom_strength
                sz = base_size + size_scale * conviction
                sz = min(max(sz, min_size), max_size)

                signal[i] = direction
                size[i] = sz
                position = direction
                entry_price = close[i]
                bars_held = 0
                entry_size = sz
            else:
                signal[i] = 0
                size[i] = 1.0

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size

        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=1e-6)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
