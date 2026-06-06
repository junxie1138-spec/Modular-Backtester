from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveFrontParams:
    """Two tunable params only (hard twist: <=2)."""

    window: int = 20
    speed_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[ShockwaveFrontParams]):
    """Trend-strength as a kinematic shockwave front speed.

    The front speed is the net directional displacement over ``window`` bars
    expressed in ATRs traversed per bar:  (close - close[-window]) / (window * ATR).
    A high positive speed means an uncongested upward front; a low negative speed
    means the front has reversed. Trading is a pure long/short stop-and-reverse:
    the only exit is the opposite entry condition firing (signal-reversal exit).
    """

    strategy_id = "gen_a1_1778894719"

    @classmethod
    def params_type(cls) -> type[ShockwaveFrontParams]:
        return ShockwaveFrontParams

    @staticmethod
    def warmup_bars(params: ShockwaveFrontParams) -> int:
        # close.shift(window) needs `window` bars; ATR rolling-mean of true range
        # needs `window` TR values and TR needs one prior close. +2 for safety.
        return int(max(2, params.window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveFrontParams) -> pd.DataFrame:
        n = int(max(2, params.window))
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.rolling(n, min_periods=n).mean()
        # Guard against zero/non-positive ATR before dividing.
        atr = atr.where(atr > 0.0)

        net_move = close - close.shift(n)
        # Shockwave front speed: ATRs of net directional travel per bar.
        front_speed = net_move / (atr * float(n))

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["net_move"] = net_move
        out["front_speed"] = front_speed
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveFrontParams,
    ) -> SignalFrame:
        thr = abs(float(params.speed_threshold))
        fs = indicators["front_speed"].to_numpy(dtype=float)
        n = len(fs)

        raw = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        # Bar-indexed loop: path-dependent stop-and-reverse state machine.
        # A position is held until the OPPOSITE entry band is crossed; this is
        # the signal-reversal exit (the only exit is the inverted entry).
        pos = 0
        for i in range(n):
            v = fs[i]
            if not np.isfinite(v):
                raw[i] = pos  # hold current position through NaN/warmup bars
                continue
            if v >= thr:
                pos = 1
            elif v <= -thr:
                pos = -1
            # else: inside the deadzone -> hold whatever pos already is
            raw[i] = pos
            if pos != 0 and thr > 0.0:
                conviction = abs(v) / thr
                size[i] = float(min(2.0, max(0.5, conviction)))

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index)
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size, index=data.index).clip(lower=0.1)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
