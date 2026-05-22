from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBreakoutParams:
    gap_threshold: float = 0.003
    breakout_lookback: int = 30
    atr_window: int = 14
    atr_init_mult: float = 2.0
    atr_trail_mult: float = 3.0
    be_trigger_pct: float = 0.04
    max_hold: int = 20
    target_vol: float = 0.011


class GeneratedStrategy(BaseStrategy[GapBreakoutParams]):
    strategy_id = "gen_a2_1779150118"

    @classmethod
    def params_type(cls):
        return GapBreakoutParams

    def warmup_bars(self, params: GapBreakoutParams) -> int:
        return int(max(params.breakout_lookback, params.atr_window)) + 2

    def indicators(self, data: pd.DataFrame, params: GapBreakoutParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]
        prev_close = close.shift(1)

        # Primitive 1: overnight gap (open vs prior close).
        gap_pct = open_ / prev_close - 1.0

        # Primitive 2: rolling N-day resistance from prior bars only.
        resistance = high.rolling(params.breakout_lookback).max().shift(1)

        # ATR for stop placement and sizing.
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()
        atr_pct = atr / close

        # Two-primitive AND: gap up AND same-bar close clears resistance.
        raw_entry = (gap_pct > params.gap_threshold) & (close > resistance)
        raw_entry = raw_entry.fillna(False)

        out = pd.DataFrame(index=data.index)
        out["gap_pct"] = gap_pct
        out["resistance"] = resistance
        out["atr"] = atr
        out["atr_pct"] = atr_pct
        out["raw_entry"] = raw_entry
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapBreakoutParams,
    ) -> SignalFrame:
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        atr_pct = indicators["atr_pct"].to_numpy(dtype=float)
        raw = indicators["raw_entry"].to_numpy()
        n = len(close)

        signal = np.zeros(n, dtype=int)

        position = 0
        entry_price = 0.0
        stop = 0.0
        run_high = 0.0
        breakeven = False
        held = 0

        for i in range(n):
            if position == 0:
                if bool(raw[i]) and not np.isnan(atr[i]) and atr[i] > 0.0:
                    position = 1
                    entry_price = close[i]
                    run_high = high[i]
                    stop = entry_price - params.atr_init_mult * atr[i]
                    breakeven = False
                    held = 0
                    signal[i] = 1
            else:
                held += 1
                if high[i] > run_high:
                    run_high = high[i]

                # Breakeven arm: once price reaches +X%, lift stop to entry.
                if not breakeven and high[i] >= entry_price * (1.0 + params.be_trigger_pct):
                    breakeven = True
                    if entry_price > stop:
                        stop = entry_price

                # Trail by k*ATR below the running high; stop only moves up.
                if breakeven and not np.isnan(atr[i]) and atr[i] > 0.0:
                    trail = run_high - params.atr_trail_mult * atr[i]
                    if trail > stop:
                        stop = trail

                if low[i] <= stop or held >= params.max_hold:
                    position = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        size = np.where(
            np.isnan(atr_pct) | (atr_pct <= 0.0),
            params.target_vol,
            atr_pct,
        )
        size = np.clip(params.target_vol / size, 0.5, 1.5)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
