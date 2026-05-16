from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


ATR_PERIOD = 14
RANK_LOOKBACK = 252
ENTRY_PCTILE = 0.55
TROUGH_PCTILE = 0.25
TROUGH_WINDOW = 20
SMA_PERIOD = 200
BREAKEVEN_TRIGGER = 0.04
INITIAL_STOP_ATR = 3.0
MAX_HOLD = 20


@dataclass(slots=True)
class GapPredatorPreyParams:
    gap_window: int = 40
    trail_atr_mult: float = 4.0


class GeneratedStrategy(BaseStrategy[GapPredatorPreyParams]):
    strategy_id = "gen_a1_1778912064"

    @classmethod
    def params_type(cls):
        return GapPredatorPreyParams

    def warmup_bars(self, params):
        return int(params.gap_window) + RANK_LOOKBACK + 5

    def indicators(self, data, params):
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        prev_close = close.shift(1)
        gap = open_ / prev_close - 1.0

        up_gap = (gap > 0.0).astype(float)
        dn_gap = (gap < 0.0).astype(float)
        w = max(int(params.gap_window), 2)
        prey = up_gap.rolling(w, min_periods=w).sum()
        predator = dn_gap.rolling(w, min_periods=w).sum()
        net = prey - predator

        net_rank = net.rolling(RANK_LOOKBACK, min_periods=RANK_LOOKBACK).rank(pct=True)
        rank_prev = net_rank.shift(1)

        cross_up = (net_rank > ENTRY_PCTILE) & (rank_prev <= ENTRY_PCTILE)
        trough = (net_rank <= TROUGH_PCTILE).astype(float)
        trough_recent = trough.rolling(TROUGH_WINDOW, min_periods=1).max() > 0.0
        entry = (cross_up & trough_recent).astype(float)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

        sma = close.rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
        regime = (close > sma).astype(float)

        out = pd.DataFrame(index=data.index)
        out["net"] = net
        out["net_rank"] = net_rank
        out["atr"] = atr
        out["sma200"] = sma
        out["entry"] = entry.fillna(0.0)
        out["regime"] = regime.fillna(0.0)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close_v = data["close"].to_numpy(dtype=float)
        atr_v = indicators["atr"].to_numpy(dtype=float)
        entry_v = indicators["entry"].to_numpy(dtype=float)
        regime_v = indicators["regime"].to_numpy(dtype=float)
        trail_mult = float(params.trail_atr_mult)

        n = len(close_v)
        raw = np.zeros(n, dtype=np.int64)
        position = 0
        entry_price = 0.0
        stop = 0.0
        bars_held = 0
        armed = False

        for i in range(n):
            a = atr_v[i]
            if position == 0:
                if (
                    entry_v[i] > 0.5
                    and regime_v[i] > 0.5
                    and np.isfinite(a)
                    and a > 0.0
                ):
                    position = 1
                    entry_price = close_v[i]
                    stop = entry_price - INITIAL_STOP_ATR * a
                    bars_held = 0
                    armed = False
                    raw[i] = 1
            else:
                bars_held += 1
                gain = close_v[i] / entry_price - 1.0
                if not armed and gain >= BREAKEVEN_TRIGGER:
                    armed = True
                    if entry_price > stop:
                        stop = entry_price
                if armed and np.isfinite(a) and a > 0.0:
                    trail = close_v[i] - trail_mult * a
                    if trail > stop:
                        stop = trail
                if close_v[i] <= stop or bars_held >= MAX_HOLD:
                    position = 0
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
