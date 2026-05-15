from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    comp_slow: int = 60
    comp_lookback: int = 5
    comp_pct: float = 0.30
    z_win: int = 20
    z_enter: float = 0.5
    atr_win: int = 14
    init_atr_mult: float = 2.5
    breakeven_pct: float = 0.03
    trail_atr_mult: float = 3.0
    base_size: float = 1.0
    size_gain: float = 0.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778887242"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.comp_slow + params.comp_lookback,
                       params.z_win, params.atr_win) + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # True range and ATR (NaN-safe via min_periods)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_win, min_periods=params.atr_win).mean()

        # Primitive 1: range-compression coil.
        # Percentile rank of current true range within the slow window;
        # a low rank means a tight bar. We require that compression
        # (rank below comp_pct) occurred somewhere in the recent lookback.
        rank = tr.rolling(params.comp_slow,
                          min_periods=params.comp_slow).rank(pct=True)
        comp_recent = rank.rolling(params.comp_lookback,
                                   min_periods=params.comp_lookback).min()
        prim1 = comp_recent < params.comp_pct

        # Primitive 2: fresh distance-from-MA z-score reclaim.
        ma = close.rolling(params.z_win, min_periods=params.z_win).mean()
        sd = close.rolling(params.z_win, min_periods=params.z_win).std()
        sd = sd.where(sd > 0.0)
        z = (close - ma) / sd
        z_prev = z.shift(1)
        prim2 = (z >= params.z_enter) & (z_prev < params.z_enter)

        # Two-primitive AND gate.
        entry_cond = (prim1 & prim2)

        # Epidemic-style susceptible reservoir: more compressed -> larger pool.
        suscept = (1.0 - rank).clip(lower=0.0, upper=1.0)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["z"] = z
        out["entry_cond"] = entry_cond.fillna(False).astype(bool)
        out["suscept"] = suscept
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: GeneratedParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_cond = indicators["entry_cond"].to_numpy(dtype=bool)
        suscept = indicators["suscept"].to_numpy(dtype=float)
        n = len(close)

        raw = np.zeros(n, dtype=int)
        size = np.full(n, params.base_size, dtype=float)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        peak = 0.0
        breakeven_done = False
        pos_size = params.base_size

        # Path-dependent breakeven-then-trail exit: bar-indexed loop.
        for i in range(n):
            if not in_pos:
                if entry_cond[i] and not np.isnan(atr[i]) and atr[i] > 0.0:
                    in_pos = True
                    entry_price = close[i]
                    peak = close[i]
                    stop = entry_price - params.init_atr_mult * atr[i]
                    breakeven_done = False
                    s = suscept[i] if not np.isnan(suscept[i]) else 0.0
                    pos_size = params.base_size * (1.0 + params.size_gain * s)
                    raw[i] = 1
                    size[i] = pos_size
            else:
                if close[i] > peak:
                    peak = close[i]
                # Breakeven: once +breakeven_pct is reached, lift stop to entry.
                if (not breakeven_done) and \
                        close[i] >= entry_price * (1.0 + params.breakeven_pct):
                    if entry_price > stop:
                        stop = entry_price
                    breakeven_done = True
                # Trail: after breakeven, trail by trail_atr_mult*ATR; up only.
                if breakeven_done and not np.isnan(atr[i]):
                    trail = peak - params.trail_atr_mult * atr[i]
                    if trail > stop:
                        stop = trail
                # Exit when close breaches the (monotonically rising) stop.
                if close[i] <= stop:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1
                    size[i] = pos_size

        # Mandatory one-bar shift: decision on bar N close, fill on N+1.
        sig = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        sz = pd.Series(size, index=data.index).shift(1).fillna(params.base_size)
        sz = sz.clip(lower=1e-6)
        df["signal"] = sig
        df["size"] = sz
        return SignalFrame(data=df, signal_column="signal", size_column="size")
