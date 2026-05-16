from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShockwaveParams:
    atr_len: int = 14
    roc_len: int = 10
    gap_thresh: float = 0.5
    gap_fill_max: float = 0.5
    roc_thresh: float = 0.005
    breakeven_pct: float = 0.03
    stop_mult: float = 2.0
    trail_mult: float = 3.0
    max_hold: int = 12


class GeneratedStrategy(BaseStrategy[GapShockwaveParams]):
    """Runaway (unfilled) gap continuation, AND-gated by multi-week ROC.

    Primitive A: a significant gap whose intraday retracement filled only a
    small fraction of the gap (a propagating shockwave, not a dissipating one).
    Primitive B: the trailing ROC points the same direction as the gap.
    Both must agree to enter. Exit is breakeven-then-trail on an ATR basis.
    """

    strategy_id = "gen_a1_1778898926"

    @classmethod
    def params_type(cls) -> type[GapShockwaveParams]:
        return GapShockwaveParams

    @staticmethod
    def warmup_bars(params: GapShockwaveParams) -> int:
        return int(max(params.atr_len + 1, params.roc_len + 1))

    def indicators(self, data: pd.DataFrame, params: GapShockwaveParams) -> pd.DataFrame:
        p = params
        o = data["open"]
        h = data["high"]
        l = data["low"]
        c = data["close"]

        prev_close = c.shift(1)
        tr = pd.concat(
            [(h - l), (h - prev_close).abs(), (l - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(p.atr_len, min_periods=p.atr_len).mean()

        gap = o - prev_close
        gap_atr = gap / atr
        roc = c.pct_change(p.roc_len)

        up_gap = gap > 0.0
        down_gap = gap < 0.0
        # Fraction of the gap retraced intraday toward the prior close.
        gap_up_safe = gap.where(up_gap)
        gap_dn_safe = (-gap).where(down_gap)
        fill_up = (o - l) / gap_up_safe
        fill_down = (h - o) / gap_dn_safe

        unfilled_up = up_gap & (gap_atr >= p.gap_thresh) & (fill_up <= p.gap_fill_max)
        unfilled_down = down_gap & (gap_atr <= -p.gap_thresh) & (fill_down <= p.gap_fill_max)

        b_up = roc >= p.roc_thresh
        b_down = roc <= -p.roc_thresh

        # Two-primitive AND: runaway gap direction must match ROC direction.
        entry_long = (unfilled_up & b_up).fillna(False)
        entry_short = (unfilled_down & b_down).fillna(False)
        entry_dir = entry_long.astype("int64") - entry_short.astype("int64")

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["gap_atr"] = gap_atr
        ind["roc"] = roc
        ind["entry_dir"] = entry_dir
        return ind

    def generate_signals(self, data, indicators, ctx, params) -> SignalFrame:
        p = params
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_dir = indicators["entry_dir"].to_numpy(dtype=float)

        n = len(close)
        pos = np.zeros(n, dtype=np.int64)

        current = 0
        entry_price = 0.0
        entry_atr = 0.0
        stop = 0.0
        peak = 0.0  # running high for long, running low for short
        bars_held = 0
        breakeven_done = False

        for i in range(n):
            if current == 0:
                d = int(entry_dir[i]) if np.isfinite(entry_dir[i]) else 0
                a = atr[i]
                if d != 0 and np.isfinite(a) and a > 0.0:
                    current = d
                    entry_price = close[i]
                    entry_atr = a
                    bars_held = 0
                    breakeven_done = False
                    peak = close[i]
                    if d == 1:
                        stop = entry_price - p.stop_mult * entry_atr
                    else:
                        stop = entry_price + p.stop_mult * entry_atr
                    pos[i] = d
                else:
                    pos[i] = 0
            else:
                bars_held += 1
                c = close[i]
                exit_now = False
                if current == 1:
                    if c > peak:
                        peak = c
                    if (not breakeven_done) and c >= entry_price * (1.0 + p.breakeven_pct):
                        if entry_price > stop:
                            stop = entry_price
                        breakeven_done = True
                    if breakeven_done:
                        trail = peak - p.trail_mult * entry_atr
                        if trail > stop:
                            stop = trail
                    if c <= stop:
                        exit_now = True
                else:
                    if c < peak:
                        peak = c
                    if (not breakeven_done) and c <= entry_price * (1.0 - p.breakeven_pct):
                        if entry_price < stop:
                            stop = entry_price
                        breakeven_done = True
                    if breakeven_done:
                        trail = peak + p.trail_mult * entry_atr
                        if trail < stop:
                            stop = trail
                    if c >= stop:
                        exit_now = True
                if bars_held >= p.max_hold:
                    exit_now = True
                if exit_now:
                    pos[i] = 0
                    current = 0
                else:
                    pos[i] = current

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
