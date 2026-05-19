from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    dd_window: int = 60
    rank_window: int = 120
    recover_lag: int = 4
    dd_rank_max: float = 0.45
    thrust_rank_min: float = 0.80
    spike_thresh: float = 0.04
    refractory_bars: int = 3


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779152104"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.dd_window + params.rank_window + params.recover_lag + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)

        dd_win = max(2, int(params.dd_window))
        rank_win = max(2, int(params.rank_window))
        lag = max(1, int(params.recover_lag))

        # Drawdown from the trailing rolling high (<= 0).
        roll_max = close.rolling(dd_win, min_periods=dd_win).max()
        drawdown = close / roll_max - 1.0

        # Primitive A: percentile rank of drawdown depth within the rank window.
        # Low rank => among the deepest drawdowns observed recently.
        dd_rank = drawdown.rolling(rank_win, min_periods=rank_win).rank(pct=True)

        # Recovery velocity: improvement of the drawdown curve over `lag` bars.
        # Positive => the underwater gap is closing.
        improvement = drawdown - drawdown.shift(lag)

        # Primitive B: percentile rank of recovery velocity within the rank window.
        # High rank => drawdown is healing unusually fast.
        thrust_rank = improvement.rolling(rank_win, min_periods=rank_win).rank(pct=True)

        ret = close.pct_change()

        out = pd.DataFrame(index=data.index)
        out["drawdown"] = drawdown
        out["dd_rank"] = dd_rank
        out["improvement"] = improvement
        out["thrust_rank"] = thrust_rank
        out["ret"] = ret
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)

        dd_rank = indicators["dd_rank"].to_numpy(dtype=float)
        thrust_rank = indicators["thrust_rank"].to_numpy(dtype=float)
        ret = indicators["ret"].to_numpy(dtype=float)

        # Two-primitive AND: deep-drawdown rank AND top-band recovery-velocity rank.
        # NaN comparisons yield False, so warmup bars never trigger an entry.
        deep = np.nan_to_num(dd_rank, nan=1.0) <= float(params.dd_rank_max)
        healing = np.nan_to_num(thrust_rank, nan=0.0) >= float(params.thrust_rank_min)
        raw_entry = deep & healing

        spike = np.abs(np.nan_to_num(ret, nan=0.0)) > float(params.spike_thresh)
        refractory = max(0, int(params.refractory_bars))

        sig = np.zeros(n, dtype=np.int64)
        cooldown = 0
        for i in range(n):
            if cooldown > 0:
                allowed = False
                cooldown -= 1
            else:
                allowed = True
            # Signal-reversal exit: signal is 1 only while the entry condition
            # holds; it drops to 0 the moment the condition flips off.
            if allowed and raw_entry[i]:
                sig[i] = 1
            # Refractory period: a large single-bar return spike suppresses
            # entries for the next `refractory` bars.
            if spike[i]:
                cooldown = refractory

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
