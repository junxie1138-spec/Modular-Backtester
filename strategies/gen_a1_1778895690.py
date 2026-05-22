from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StaircaseBreakoutParams:
    streak_k: int = 4
    tol: float = 0.002
    profit_target: float = 0.02
    max_hold_bars: int = 2
    base_size: float = 0.6
    streak_size_scale: float = 0.15
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[StaircaseBreakoutParams]):
    strategy_id = "gen_a1_1778895690"

    @classmethod
    def params_type(cls) -> type[StaircaseBreakoutParams]:
        return StaircaseBreakoutParams

    @staticmethod
    def warmup_bars(params: StaircaseBreakoutParams) -> int:
        return int(params.streak_k) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: StaircaseBreakoutParams) -> pd.DataFrame:
        k = max(1, int(params.streak_k))
        tol = float(params.tol)

        low = data["low"]
        high = data["high"]
        prev_low = low.shift(1)

        # Hysteresis: a small tolerance band keeps a micro-dip from resetting
        # the higher-low staircase streak.
        cond = (low >= prev_low * (1.0 - tol)).fillna(False)

        # Vectorised consecutive-True streak count.
        grp = (~cond).cumsum()
        streak = cond.groupby(grp).cumsum().astype(float)
        streak = streak.reindex(data.index).fillna(0.0)

        # Ceiling of the staircase: highest high over the prior k bars.
        prior_high = high.rolling(k).max().shift(1)

        out = pd.DataFrame(index=data.index)
        out["streak"] = streak
        out["prior_high"] = prior_high
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StaircaseBreakoutParams,
    ) -> SignalFrame:
        k = max(1, int(params.streak_k))
        pt = float(params.profit_target)
        max_hold = max(1, int(params.max_hold_bars))
        base_size = float(params.base_size)
        size_scale = float(params.streak_size_scale)
        max_size = float(params.max_size)

        close = data["close"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        prior_high = indicators["prior_high"].to_numpy(dtype=float)

        n = len(close)
        streak = np.nan_to_num(streak, nan=0.0)

        valid_high = np.isfinite(prior_high)
        entry_cond = valid_high & (streak >= k) & (close > prior_high)
        excess = np.clip(streak - k, 0.0, None)

        signal = np.zeros(n, dtype=int)
        size = np.full(n, 1.0, dtype=float)

        in_pos = False
        armed = True
        entry_price = 0.0
        bars_held = 0
        pos_size = base_size

        for i in range(n):
            if in_pos:
                bars_held += 1
                gain = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if gain >= pt or bars_held >= max_hold:
                    signal[i] = 0
                    in_pos = False
                else:
                    signal[i] = 1
                    size[i] = pos_size
            else:
                # Hysteresis re-arm: streak must collapse below k before a new
                # breakout from a fresh staircase can be taken.
                if streak[i] < k:
                    armed = True
                if armed and entry_cond[i]:
                    sz = base_size * (1.0 + size_scale * excess[i])
                    if sz > max_size:
                        sz = max_size
                    if sz <= 0.0:
                        sz = base_size
                    pos_size = sz
                    entry_price = close[i]
                    bars_held = 0
                    in_pos = True
                    armed = False
                    signal[i] = 1
                    size[i] = sz
                else:
                    signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = (
            pd.Series(size, index=data.index).shift(1).fillna(1.0).astype(float)
        )
        df["size"] = df["size"].clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
