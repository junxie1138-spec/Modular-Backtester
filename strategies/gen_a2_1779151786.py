from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 60
    rank_window: int = 60
    roc_window: int = 10
    arm_low_pct: float = 0.15
    disarm_high_pct: float = 0.60
    recovery_thr: float = 0.55
    hold_bars: int = 17


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779151786"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        dd_path = params.peak_window + params.rank_window
        roc_path = params.roc_window + params.rank_window
        return int(max(dd_path, roc_path)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        dd = close / peak - 1.0
        dd_rank = dd.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        roc = close.pct_change(params.roc_window)
        roc_rank = roc.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["dd_rank"] = dd_rank
        out["roc"] = roc
        out["roc_rank"] = roc_rank
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        n = len(data)
        sig = np.zeros(n, dtype=int)
        dd_rank = indicators["dd_rank"].to_numpy(dtype=float)
        roc_rank = indicators["roc_rank"].to_numpy(dtype=float)

        warmup = GeneratedStrategy.warmup_bars(params)
        hold = max(1, int(params.hold_bars))
        arm_low = float(params.arm_low_pct)
        disarm_high = float(params.disarm_high_pct)
        rec_thr = float(params.recovery_thr)

        armed = False
        i = max(warmup, 1)
        while i < n:
            dr = dd_rank[i]
            if not np.isnan(dr):
                if dr <= arm_low:
                    armed = True
                elif dr >= disarm_high:
                    armed = False

            rr = roc_rank[i]
            rr_prev = roc_rank[i - 1]
            cross_up = (
                not np.isnan(rr)
                and not np.isnan(rr_prev)
                and rr > rec_thr
                and rr_prev <= rec_thr
            )

            if armed and cross_up:
                end = min(i + hold, n)
                sig[i:end] = 1
                armed = False
                i = end
                continue
            i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
