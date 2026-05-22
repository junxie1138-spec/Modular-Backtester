from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# --- fixed (non-tunable) structural constants -----------------------------
_HIGH_WINDOW = 20   # Donchian breakout lookback
_ATR_WINDOW = 14    # ATR lookback for the volatility stop
_MAX_HOLD = 18      # ~3-4 week holding-horizon cap (trading bars)


@dataclass(slots=True)
class ShockwaveBreakoutParams:
    streak_threshold: int = 3
    atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[ShockwaveBreakoutParams]):
    strategy_id = "gen_a1_1778885961"

    @classmethod
    def params_type(cls):
        return ShockwaveBreakoutParams

    @staticmethod
    def warmup_bars(params: ShockwaveBreakoutParams) -> int:
        return _HIGH_WINDOW + _ATR_WINDOW + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveBreakoutParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # True Range -> ATR (NaN during warmup, handled downstream)
        prior_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prior_close).abs(),
                (low - prior_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(_ATR_WINDOW, min_periods=_ATR_WINDOW).mean()

        # A bar 'breaks out' if its close exceeds the highest high of the
        # prior _HIGH_WINDOW bars (classic Donchian breakout).
        prior_high = (
            high.rolling(_HIGH_WINDOW, min_periods=_HIGH_WINDOW).max().shift(1)
        )
        new_high = (close > prior_high).fillna(False)

        # Consecutive-streak count of breakout bars: cumulative count that
        # resets to zero whenever a non-breakout bar appears.
        nh = new_high.astype(int)
        reset = (nh == 0).cumsum()
        streak = nh.groupby(reset).cumsum()

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["streak"] = streak.astype(float)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        n = len(close)

        threshold = max(1, int(params.streak_threshold))
        k = float(params.atr_mult)

        sig = np.zeros(n, dtype=int)
        position = 0
        entry_stop = 0.0
        bars_held = 0

        for i in range(n):
            exited = False
            if position == 1:
                bars_held += 1
                # Fixed volatility-stop: stop level is frozen at entry
                # (entry_price - k*ATR_at_entry), never trailed.
                if close[i] <= entry_stop or bars_held >= _MAX_HOLD:
                    position = 0
                    bars_held = 0
                    exited = True
                else:
                    sig[i] = 1

            if position == 0 and not exited:
                a = atr[i]
                s = streak[i]
                if not np.isnan(a) and not np.isnan(s) and a > 0.0:
                    if s >= threshold:
                        position = 1
                        entry_stop = close[i] - k * a
                        bars_held = 0
                        sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        # MANDATORY one-bar shift: decide on bar N close, fill on N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
