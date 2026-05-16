from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 10
    runs_ratio_min: float = 1.10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778897056"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # pct_change (1) + rolling(window) → window + 1; +1 extra safety
        return params.window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        w = params.window

        ret = close.pct_change()
        sign = np.sign(ret)

        # 1 where sign flips bar-over-bar, 0 otherwise
        is_change = (sign != sign.shift(1)).astype(float)
        # Runs = sign-change count + 1 (the opening run needs no preceding change)
        rolling_runs = is_change.rolling(w, min_periods=w).sum() + 1.0

        rolling_pos = (sign > 0).astype(float).rolling(w, min_periods=w).sum()
        rolling_neg = (sign < 0).astype(float).rolling(w, min_periods=w).sum()
        total = (rolling_pos + rolling_neg).clip(lower=1.0)
        # Wald-Wolfowitz expected runs under the null hypothesis of a random walk
        expected_runs = 2.0 * rolling_pos * rolling_neg / total + 1.0

        # Ratio > 1 → more alternating than random (elastic regime)
        runs_ratio = rolling_runs / expected_runs.clip(lower=1e-9)
        cum_ret = ret.rolling(w, min_periods=w).sum()

        ind = pd.DataFrame(index=data.index)
        ind["runs_ratio"] = runs_ratio
        ind["cum_ret"] = cum_ret
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        runs_ratio = indicators["runs_ratio"]
        cum_ret = indicators["cum_ret"]

        # Elastic regime: sign alternation exceeds random-walk expectation
        elastic = runs_ratio > params.runs_ratio_min
        # Spring stretched: rolling cumulative return is negative (below equilibrium)
        stretched = cum_ret < 0.0

        entry = (elastic & stretched).astype(int)

        df = pd.DataFrame(index=data.index)
        # Shift by 1: decision on bar N fills on bar N+1
        df["signal"] = entry.shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
