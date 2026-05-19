from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_len: int = 3
    min_streak_return: float = 0.005
    size_scale: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    """Mean-reversion: fade the net direction of a consecutive range-expansion streak.

    A 'prey boom' is a run of bars each posting a wider high-low range than the
    bar before it - volatility energy feeding on itself. When the run reaches
    ``streak_len`` bars, the accumulated move is treated as exhausted and faded.
    The position is held until the opposite-direction entry fires (signal-
    reversal exit): a completed expansion streak in the other direction.
    """

    strategy_id = "gen_a2_1779148080"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # Longest lookback: pct_change over streak_len bars, plus the 1-bar
        # shift used to detect range expansion, plus a small NaN cushion.
        return int(max(2, params.streak_len)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        k = int(max(1, params.streak_len))
        thr = float(abs(params.min_streak_return))
        scale = float(params.size_scale)

        # Range-expansion flag: this bar's range strictly wider than the prior.
        rng = (data["high"].astype(float) - data["low"].astype(float))
        expanding = (rng > rng.shift(1)).fillna(False)
        exp_int = expanding.astype(int)

        # Length of the current run of consecutive expanding bars. Each non-
        # expanding bar opens a fresh group, so the cumulative sum within a
        # group counts only the unbroken expansion run ending at that bar.
        grp = (exp_int == 0).cumsum()
        streak = exp_int.groupby(grp).cumsum().astype(float)
        out["streak"] = streak

        # Net close-to-close move accumulated across the streak window.
        net_ret = data["close"].astype(float).pct_change(k)
        net_ret = net_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["net_ret"] = net_ret

        # Streak is complete once it reaches the threshold length. Fade the
        # net direction of the move that built it.
        trigger = streak >= float(k)
        out["long_entry"] = (trigger & (net_ret < -thr)).astype(float)
        out["short_entry"] = (trigger & (net_ret > thr)).astype(float)

        # Size grows with how overheated the streak is, capped at 3x.
        ratio = (streak / float(k)).clip(lower=1.0, upper=3.0)
        out["size_raw"] = (max(0.01, scale) * ratio).clip(lower=0.01)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        long_e = (indicators["long_entry"].to_numpy() > 0.5)
        short_e = (indicators["short_entry"].to_numpy() > 0.5)

        # Signal-reversal exit: a position is held until the OPPOSITE entry
        # condition fires; only then does it close (and reverse).
        pos = np.zeros(n, dtype=int)
        cur = 0
        for i in range(n):
            if cur == 0:
                if long_e[i]:
                    cur = 1
                elif short_e[i]:
                    cur = -1
            elif cur == 1:
                if short_e[i]:
                    cur = -1
            else:  # cur == -1
                if long_e[i]:
                    cur = 1
            pos[i] = cur

        df = pd.DataFrame(index=data.index)
        # Decide on bar N's close, fill on bar N+1.
        df["signal"] = pd.Series(pos, index=data.index).shift(1).fillna(0).astype(int)

        size = indicators["size_raw"].astype(float)
        size = size.replace([np.inf, -np.inf], np.nan)
        size = size.fillna(float(max(0.01, params.size_scale))).clip(lower=0.01)
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
