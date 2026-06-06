from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeUndercutParams:
    lookback: int = 20            # window for the trailing-low channel
    atr_period: int = 14         # ATR window used to normalize probe depth
    min_probe_atr: float = 0.12  # minimum undercut depth, in ATRs, to count as a real probe
    close_pos_thresh: float = 0.60  # close must finish at/above this fraction of the bar range
    prior_weak_thresh: float = 0.45  # prior bar must have closed at/below this fraction (hysteresis arming)
    require_prior_weak: bool = True   # require a weak->strong intrabar character flip
    hold_bars: int = 4           # fixed-bar exit: flat exactly N bars after entry
    size_floor: float = 0.60     # minimum conviction size multiplier
    size_cap: float = 1.00       # maximum conviction size multiplier
    size_slope: float = 0.35     # probe-depth-to-size scaling


class GeneratedStrategy(BaseStrategy[RangeUndercutParams]):
    strategy_id = "gen_a1_1778897470"

    @classmethod
    def params_type(cls):
        return RangeUndercutParams

    def warmup_bars(self, params: RangeUndercutParams) -> int:
        return int(max(params.lookback, params.atr_period)) + 2

    def indicators(self, data: pd.DataFrame, params: RangeUndercutParams) -> pd.DataFrame:
        p = params
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = high - low
        safe_range = bar_range.where(bar_range > 0)
        # Primitive 2 input: where the close sits inside the bar's own high-low range.
        close_pos = ((close - low) / safe_range).fillna(0.5)

        # ATR via classic true range, NaN during warmup.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(p.atr_period).mean()

        # Primitive 1 input: trailing N-day low, excluding the current bar.
        roll_min_low_prior = low.rolling(p.lookback).min().shift(1)
        undercut = (low < roll_min_low_prior).astype(float)
        probe_depth = (roll_min_low_prior - low) / atr.where(atr > 0)

        out = pd.DataFrame(index=data.index)
        out["close_pos"] = close_pos
        out["atr"] = atr
        out["roll_min_low_prior"] = roll_min_low_prior
        out["undercut"] = undercut
        out["probe_depth"] = probe_depth
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeUndercutParams,
    ) -> SignalFrame:
        p = params
        close_pos = indicators["close_pos"]
        probe_depth = indicators["probe_depth"]
        undercut = indicators["undercut"] > 0.5

        # Primitive 1: downside range extension is deep enough to be a genuine probe.
        # NaN probe_depth (warmup) compares False, so no signal fires during warmup.
        meaningful = probe_depth >= p.min_probe_atr
        primitive_one = undercut & meaningful

        # Primitive 2: intrabar rejection - close recovers into the upper part of the range.
        rejection = close_pos >= p.close_pos_thresh
        if p.require_prior_weak:
            # Hysteresis: demand a weak-close bar immediately before the strong-close flip.
            primitive_two = rejection & (close_pos.shift(1) <= p.prior_weak_thresh)
        else:
            primitive_two = rejection

        # Two-primitive AND: both range conditions must agree.
        raw_entry = (primitive_one & primitive_two).fillna(False).to_numpy()

        n = len(data)
        hold = max(1, int(p.hold_bars))
        depth = probe_depth.fillna(0.0).to_numpy()

        sig = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)

        # Fixed-bar exit: on entry, stay long exactly `hold` bars then go flat.
        # No signal-based exit; new entries are ignored while a position is open.
        i = 0
        while i < n:
            if raw_entry[i]:
                end = min(i + hold, n)
                sig[i:end] = 1
                conviction = p.size_floor + depth[i] * p.size_slope
                conviction = float(min(max(conviction, p.size_floor), p.size_cap))
                sizes[i:end] = conviction
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size_series = pd.Series(sizes, index=data.index).shift(1).fillna(1.0)
        df["size"] = size_series.clip(lower=p.size_floor).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
