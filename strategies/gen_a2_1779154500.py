from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ScarredBreakoutParams:
    peak_window: int = 60
    scar_window: int = 40
    dd_threshold: float = 0.07
    breakout_window: int = 20
    atr_window: int = 14
    atr_mult: float = 2.5
    be_trigger: float = 0.03
    trail_mult: float = 3.0
    refractory: int = 5
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[ScarredBreakoutParams]):
    strategy_id = "gen_a2_1779154500"

    @classmethod
    def params_type(cls):
        return ScarredBreakoutParams

    @staticmethod
    def warmup_bars(params: ScarredBreakoutParams) -> int:
        a = params.peak_window + params.scar_window + 5
        b = params.breakout_window + 2
        c = params.atr_window + 2
        return int(max(a, b, c))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ScarredBreakoutParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Primitive 1: drawdown depth. Rolling peak -> running drawdown -> the
        # worst (deepest) drawdown seen within the recent scar window.
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        drawdown = close / peak - 1.0
        scar_dd = drawdown.rolling(
            params.scar_window, min_periods=params.scar_window
        ).min()

        # Primitive 2: breakout. Prior N-bar Donchian high (shifted to exclude
        # the current bar so a fresh close > high is a genuine new high).
        donchian_high_prev = high.rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max().shift(1)

        # ATR for the initial stop and the trailing stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        # Two-primitive AND: a recent deep drawdown scar AND a fresh breakout.
        scar_ok = scar_dd <= -params.dd_threshold
        breakout_ok = close > donchian_high_prev
        entry_raw = (scar_ok & breakout_ok).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["scar_dd"] = scar_dd
        out["donchian_high_prev"] = donchian_high_prev
        out["atr"] = atr
        out["entry_raw"] = entry_raw.astype(bool)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ScarredBreakoutParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_raw = indicators["entry_raw"].to_numpy()

        pos = np.zeros(n, dtype=int)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        be_armed = False
        last_entry = -(10 ** 9)

        for i in range(n):
            if not in_pos:
                can_enter = (
                    bool(entry_raw[i])
                    and (i - last_entry >= params.refractory)
                    and np.isfinite(atr[i])
                    and atr[i] > 0.0
                )
                if can_enter:
                    in_pos = True
                    entry_price = close[i]
                    stop = entry_price - params.atr_mult * atr[i]
                    be_armed = False
                    last_entry = i
                    pos[i] = 1
                else:
                    pos[i] = 0
            else:
                # Breakeven-then-trail: once price reaches +be_trigger, lock the
                # stop at entry, then trail by trail_mult*ATR. Stop only rises.
                if (not be_armed) and high[i] >= entry_price * (1.0 + params.be_trigger):
                    be_armed = True
                    if entry_price > stop:
                        stop = entry_price
                if be_armed and np.isfinite(atr[i]):
                    trail = close[i] - params.trail_mult * atr[i]
                    if trail > stop:
                        stop = trail
                if low[i] <= stop:
                    in_pos = False
                    be_armed = False
                    pos[i] = 0
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.base_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
