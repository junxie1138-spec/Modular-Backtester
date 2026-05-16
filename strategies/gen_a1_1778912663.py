from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveStreakParams:
    entry_streak: int = 3
    exit_streak: int = 2
    max_streak_scale: int = 7
    min_size: float = 0.35
    max_size: float = 1.0
    require_deceleration: bool = True


class GeneratedStrategy(BaseStrategy[ShockwaveStreakParams]):
    """Mean-reversion on consecutive down-close streaks.

    A streak of down closes is treated as a traffic-style congestion
    shockwave. Entry fires when the down-streak reaches a threshold and
    (optionally) the wave is dissipating - each successive drop shallower
    than the previous one. Position size scales with streak depth. The
    only exit is a signal reversal: an up-close streak reaching its own
    threshold, i.e. the entry condition flipping to its opposite.
    """

    strategy_id = "gen_a1_1778912663"

    @classmethod
    def params_type(cls) -> type[ShockwaveStreakParams]:
        return ShockwaveStreakParams

    @staticmethod
    def warmup_bars(params: ShockwaveStreakParams) -> int:
        return int(max(params.max_streak_scale, params.entry_streak,
                       params.exit_streak, 2)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveStreakParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        down = (close < close.shift(1)).fillna(False)
        up = (close > close.shift(1)).fillna(False)

        # consecutive-streak counts: cumsum within each unbroken block
        down_blk = (~down).cumsum()
        up_blk = (~up).cumsum()
        down_streak = down.groupby(down_blk).cumsum().astype(float)
        up_streak = up.groupby(up_blk).cumsum().astype(float)

        # shockwave dissipation: today's drop shallower than yesterday's
        # (ret rising means the loss magnitude is shrinking)
        decel = (ret > ret.shift(1)).fillna(False).astype(float)

        # signal-scaled size: deeper down-streak -> larger position
        span = max(int(params.max_streak_scale) - int(params.entry_streak), 1)
        frac = ((down_streak - float(params.entry_streak)) / float(span)).clip(0.0, 1.0)
        lo = float(params.min_size)
        hi = float(params.max_size)
        size_raw = (lo + (hi - lo) * frac).clip(lower=0.01)

        out = pd.DataFrame(index=data.index)
        out["down_streak"] = down_streak.fillna(0.0)
        out["up_streak"] = up_streak.fillna(0.0)
        out["decel"] = decel.fillna(0.0)
        out["size_raw"] = size_raw.fillna(lo)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: ShockwaveStreakParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        down_streak = indicators["down_streak"].to_numpy(dtype=float)
        up_streak = indicators["up_streak"].to_numpy(dtype=float)
        decel = indicators["decel"].to_numpy(dtype=float) > 0.5
        size_raw = indicators["size_raw"].to_numpy(dtype=float)

        entry_streak = int(params.entry_streak)
        exit_streak = int(params.exit_streak)
        require_decel = bool(params.require_deceleration)
        min_size = float(params.min_size)

        n = len(df)
        sig = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_size = 1.0
        for i in range(n):
            entry_ok = (down_streak[i] >= entry_streak
                        and (decel[i] or not require_decel))
            if not in_pos:
                if entry_ok:
                    in_pos = True
                    sig[i] = 1
                    es = float(size_raw[i])
                    if not np.isfinite(es) or es <= 0.0:
                        es = min_size
                    entry_size = es
                    size[i] = entry_size
            else:
                # signal-reversal exit: the opposite streak condition fires
                if up_streak[i] >= exit_streak:
                    in_pos = False
                    sig[i] = 0
                    size[i] = 1.0
                else:
                    sig[i] = 1
                    size[i] = entry_size

        df["signal"] = sig
        df["size"] = size

        # mandatory one-bar shift: decide on bar N close, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.01)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
