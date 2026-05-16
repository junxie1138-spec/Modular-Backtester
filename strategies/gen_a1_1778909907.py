from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    corr_window: int = 20
    rank_window: int = 120
    gap_window: int = 5
    enter_pctl: float = 0.80
    exit_pctl: float = 0.50


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778909907"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.rank_window + params.corr_window + params.gap_window + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        prior_close = data["close"].shift(1)
        # Overnight gap return: today's open vs prior close.
        gap = data["open"] / prior_close - 1.0
        out["gap"] = gap

        # Rolling lag-1 autocorrelation of the gap series itself.
        cw = max(int(params.corr_window), 3)
        ac = gap.rolling(cw, min_periods=cw).corr(gap.shift(1))
        out["gap_autocorr"] = ac

        # Twist: percentile rank of that autocorrelation within its own history.
        rw = max(int(params.rank_window), 5)
        pctl = ac.rolling(rw, min_periods=rw).rank(pct=True)
        out["gap_pctl"] = pctl

        # Recent gap-direction context.
        gw = max(int(params.gap_window), 1)
        out["gap_mean"] = gap.rolling(gw, min_periods=gw).mean()

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        df = pd.DataFrame(index=data.index)

        pctl = indicators["gap_pctl"].to_numpy(dtype=float)
        gap_mean = indicators["gap_mean"].to_numpy(dtype=float)
        gap_ok = gap_mean > 0.0  # NaN > 0 -> False, NaN-safe

        enter = float(params.enter_pctl)
        exit_ = float(params.exit_pctl)
        if exit_ >= enter:
            exit_ = enter - 0.05

        sig = np.zeros(n, dtype=int)
        regime_on = False
        position = 0

        for i in range(n):
            p = pctl[i]
            valid = p == p  # False when NaN

            # Schmitt-trigger hysteresis on the gap-autocorrelation percentile.
            if not regime_on:
                if valid and p >= enter:
                    regime_on = True
            else:
                if valid and p < exit_:
                    regime_on = False

            entry_cond = regime_on and bool(gap_ok[i])

            # Signal-reversal exit: hold until the entry condition flips off.
            if position == 0:
                if entry_cond:
                    position = 1
            else:
                if not entry_cond:
                    position = 0

            sig[i] = position

        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
