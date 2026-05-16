from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRegimeParams:
    regime_window: int = 45
    regime_thresh: float = 0.10
    atr_window: int = 14
    trail_k: float = 2.5
    gap_threshold: float = 0.30
    spike_z: float = 2.5
    refractory_bars: int = 3
    max_hold_bars: int = 6
    base_size: float = 0.40
    size_gain: float = 0.60
    max_size: float = 1.00


class GeneratedStrategy(BaseStrategy[GapRegimeParams]):
    strategy_id = "gen_a1_1778897036"

    @classmethod
    def params_type(cls):
        return GapRegimeParams

    @staticmethod
    def warmup_bars(params: GapRegimeParams) -> int:
        return int(max(params.regime_window, params.atr_window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRegimeParams) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        prior_close = close.shift(1)

        # True range and ATR in price units
        tr = pd.concat(
            [
                high - low,
                (high - prior_close).abs(),
                (low - prior_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()
        atr_safe = atr.where(atr > 0.0)

        # Overnight gap: price units, ATR-normalized, and as a return
        gap_px = open_ - prior_close
        gap_norm = gap_px / atr_safe
        gap_ret = open_ / prior_close - 1.0

        # Same-bar intraday follow-through (open -> close)
        intraday = close / open_ - 1.0

        # Regime score: rolling agreement of gap direction vs intraday direction.
        # > 0 => gaps tend to be extended (continuation regime)
        # < 0 => gaps tend to be faded (mean-reversion regime)
        agree = np.sign(gap_ret) * np.sign(intraday)
        regime = agree.rolling(params.regime_window).mean()

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["gap_norm"] = gap_norm
        out["regime"] = regime
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRegimeParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        gap_norm = indicators["gap_norm"].to_numpy(dtype=float)
        regime = indicators["regime"].to_numpy(dtype=float)

        n = len(close)
        pos = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        warm = int(max(params.regime_window, params.atr_window)) + 2

        in_pos = False
        high_water = 0.0
        hold = 0
        entry_size = 1.0
        refractory = 0

        for i in range(n):
            # advance refractory clock before this bar's decisions
            if refractory > 0:
                refractory -= 1

            atr_i = atr[i]
            gn = gap_norm[i]
            rg = regime[i]

            valid = (
                i >= warm
                and np.isfinite(atr_i)
                and atr_i > 0.0
                and np.isfinite(gn)
                and np.isfinite(rg)
            )

            # a large gap is a volatility spike -> open a refractory window
            if valid and abs(gn) > params.spike_z:
                refractory = params.refractory_bars

            if in_pos:
                if close[i] > high_water:
                    high_water = close[i]
                hold += 1
                if np.isfinite(atr_i) and atr_i > 0.0:
                    stop = high_water - params.trail_k * atr_i
                else:
                    stop = high_water
                exit_now = (close[i] < stop) or (hold >= params.max_hold_bars)
                if exit_now:
                    in_pos = False
                    pos[i] = 0
                    size[i] = entry_size
                else:
                    pos[i] = 1
                    size[i] = entry_size
            else:
                entered = False
                if valid and refractory <= 0:
                    continuation = rg > params.regime_thresh
                    fade = rg < -params.regime_thresh
                    long_continuation = continuation and (gn > params.gap_threshold)
                    long_fade = fade and (gn < -params.gap_threshold)
                    if long_continuation or long_fade:
                        # signal-scaled position sizing (the twist):
                        # size grows with regime conviction and gap magnitude
                        regime_conv = min(abs(rg), 1.0)
                        gap_conv = min(abs(gn) / 2.0, 1.0)
                        conv = 0.5 * regime_conv + 0.5 * gap_conv
                        s = params.base_size + params.size_gain * conv
                        s = float(min(max(s, 0.05), params.max_size))
                        in_pos = True
                        high_water = close[i]
                        hold = 0
                        entry_size = s
                        pos[i] = 1
                        size[i] = s
                        entered = True
                if not entered:
                    pos[i] = 0
                    size[i] = 1.0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.05)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
