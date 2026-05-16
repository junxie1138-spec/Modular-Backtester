from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    """Parameters for the drawdown escape-velocity strategy."""

    dd_window: int = 60          # rolling window defining the prior peak / drawdown
    recovery_window: int = 8     # bars over which recovery thrust is accumulated
    dd_min: float = 0.04         # minimum drawdown depth (R bars ago) to be 'in a hole'
    ratio_long: float = 0.35     # escape-velocity threshold to go long
    ratio_short: float = -0.25   # escape-velocity threshold (negative) to go short
    dd_floor: float = 0.01       # floor on hole depth to keep the ratio finite
    refractory_bars: int = 5     # lockout bars after any exit before re-entry


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    """Trade the recovery-thrust-to-drawdown-depth ratio as an escape-velocity signal.

    Drawdown depth is close vs its rolling high. Recovery thrust is the summed
    close-to-close return over a short window. The ratio thrust / |depth R bars
    ago| measures how fast lost ground is being regained per unit of hole. A high
    positive ratio (fast V-recovery from a real hole) goes long; a deeply negative
    ratio (relapse while still underwater) goes short. Entry needs the condition
    confirmed on two consecutive bars; the position is held until the entry
    condition flips false (signal-reversal exit), then a refractory lockout runs.
    """

    strategy_id = "gen_a1_1778914526"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        w = max(int(params.dd_window), 2)
        r = max(int(params.recovery_window), 1)
        return int(w + r + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        w = max(int(params.dd_window), 2)
        r = max(int(params.recovery_window), 1)
        floor = float(params.dd_floor)

        # Drawdown relative to the rolling peak; <= 0 when underwater.
        roll_high = close.rolling(w, min_periods=w).max()
        dd = close / roll_high - 1.0

        # Recovery thrust: summed close-to-close returns over the recovery window.
        ret = close.pct_change()
        thrust = ret.rolling(r, min_periods=r).sum()

        # Hole depth as it stood R bars ago, so the thrust is not part of it.
        dd_lag = dd.shift(r)
        depth = dd_lag.abs()
        denom = depth.clip(lower=floor)
        recovery_ratio = thrust / denom

        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["dd_lag"] = dd_lag
        out["depth"] = depth
        out["thrust"] = thrust
        out["recovery_ratio"] = recovery_ratio
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        dd_min = float(params.dd_min)
        ratio_long = float(params.ratio_long)
        ratio_short = float(params.ratio_short)
        refr = max(int(params.refractory_bars), 0)
        warmup = GeneratedStrategy.warmup_bars(params)

        depth_s = indicators["depth"]
        ratio_s = indicators["recovery_ratio"]

        # Raw regime conditions. NaN comparisons yield False, so warmup is safe.
        underwater = (depth_s >= dd_min).fillna(False)
        long_raw = (underwater & (ratio_s >= ratio_long)).fillna(False).to_numpy()
        short_raw = (underwater & (ratio_s <= ratio_short)).fillna(False).to_numpy()

        signal = np.zeros(n, dtype=int)
        pos = 0
        cooldown = 0

        for i in range(n):
            if i < warmup or i < 1:
                signal[i] = 0
                continue

            if pos == 0:
                if cooldown > 0:
                    cooldown -= 1
                else:
                    # Two-bar confirmation before any entry.
                    long_conf = bool(long_raw[i]) and bool(long_raw[i - 1])
                    short_conf = bool(short_raw[i]) and bool(short_raw[i - 1])
                    if long_conf:
                        pos = 1
                    elif short_conf:
                        pos = -1
            elif pos == 1:
                # Signal-reversal exit: hold until the long entry condition flips.
                if not bool(long_raw[i]):
                    pos = 0
                    cooldown = refr
            else:  # pos == -1
                if not bool(short_raw[i]):
                    pos = 0
                    cooldown = refr

            signal[i] = pos

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        # Decide on bar N's close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
