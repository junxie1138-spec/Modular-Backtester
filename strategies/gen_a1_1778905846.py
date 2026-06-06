from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDefenseParams:
    gap_min_pct: float = 0.0015
    window_w: int = 20
    rank_lookback: int = 120
    rank_threshold: float = 0.75
    confirm_bars: int = 2
    atr_period: int = 14
    breakeven_trigger_pct: float = 0.03
    trail_atr_mult: float = 3.0
    refractory_bars: int = 5
    max_hold_bars: int = 30


class GeneratedStrategy(BaseStrategy[GapDefenseParams]):
    """Long-only gap-defense persistence strategy.

    Mechanism: an up-gap is 'defended' when the bar's low never trades back
    below the prior close (the gap holds all day). The rolling defense rate is
    the fraction of up-gaps in a window that were defended. Its rolling
    percentile rank is the entry primitive: a high rank signals persistent
    demand. Entry requires two consecutive bars of confirmation; after any
    exit a refractory lockout suppresses re-entry. Exit is
    breakeven-then-trail.
    """

    strategy_id = "gen_a1_1778905846"

    @classmethod
    def params_type(cls):
        return GapDefenseParams

    @staticmethod
    def warmup_bars(params):
        return int(params.rank_lookback + params.window_w + params.atr_period + 1)

    def indicators(self, data, params):
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        prior_close = close.shift(1)
        gap = open_ / prior_close - 1.0

        is_up_gap = gap > params.gap_min_pct
        gap_held = is_up_gap & (low > prior_close)

        w = max(int(params.window_w), 1)
        up_count = is_up_gap.astype(float).rolling(w).sum()
        held_count = gap_held.astype(float).rolling(w).sum()
        # NaN-safe: where there are no up-gaps in the window (incl. warmup
        # NaN), the defense rate is defined as 0.0 rather than 0/0.
        defense_rate = (held_count / up_count).where(up_count > 0, 0.0)
        defense_rate = defense_rate.fillna(0.0)

        rl = max(int(params.rank_lookback), 2)
        defense_rank = defense_rate.rolling(rl).rank(pct=True)
        defense_rank = defense_rank.fillna(0.0)

        tr = pd.concat(
            [
                (high - low),
                (high - prior_close).abs(),
                (low - prior_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(max(int(params.atr_period), 1)).mean()
        atr = atr.bfill().fillna(tr.expanding().mean()).fillna(0.0)

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap.fillna(0.0)
        out["defense_rate"] = defense_rate
        out["defense_rank"] = defense_rank
        out["atr"] = atr
        return out

    def generate_signals(self, data, indicators, ctx, params):
        idx = data.index
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        rank = indicators["defense_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        cond = rank >= float(params.rank_threshold)
        cb = max(int(params.confirm_bars), 1)

        # Two-bar confirmation: entry condition must hold cb consecutive bars.
        confirm = np.zeros(n, dtype=bool)
        run = 0
        for i in range(n):
            run = run + 1 if bool(cond[i]) else 0
            confirm[i] = run >= cb

        warmup = self.warmup_bars(params)
        signal = np.zeros(n, dtype=int)

        position = 0
        entry_price = 0.0
        stop = 0.0
        breakeven_armed = False
        bars_held = 0
        refractory_until = -1

        be_trigger = float(params.breakeven_trigger_pct)
        k = float(params.trail_atr_mult)
        max_hold = max(int(params.max_hold_bars), 1)
        refr = max(int(params.refractory_bars), 0)

        for i in range(n):
            if i < warmup:
                signal[i] = 0
                continue

            price = close[i]
            atr_i = atr[i]
            if not np.isfinite(atr_i) or atr_i <= 0:
                atr_i = 0.0

            if position == 0:
                if i >= refractory_until and confirm[i] and atr_i > 0.0:
                    position = 1
                    entry_price = price
                    stop = price - k * atr_i
                    breakeven_armed = False
                    bars_held = 0
                signal[i] = position
            else:
                bars_held += 1

                # Breakeven: once price reaches +X%, lift stop to entry.
                if (not breakeven_armed) and price >= entry_price * (1.0 + be_trigger):
                    breakeven_armed = True
                    stop = max(stop, entry_price)

                # Trail: after breakeven, ratchet stop up by k*ATR; never down.
                if breakeven_armed and atr_i > 0.0:
                    stop = max(stop, price - k * atr_i)

                exit_now = (price <= stop) or (bars_held >= max_hold)
                if exit_now:
                    position = 0
                    refractory_until = i + 1 + refr
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = 1.0
        # MANDATORY one-bar shift: decide on bar N close, fill on N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
