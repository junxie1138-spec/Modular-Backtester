from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    lookback: int = 63
    percentile: float = 0.75


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1791298931"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        change = data["close"].diff()
        up = change.gt(0)
        down = change.lt(0)
        out = pd.DataFrame(index=data.index)
        out["up_run"] = up.groupby((~up).cumsum()).cumsum()
        out["down_run"] = down.groupby((~down).cumsum()).cumsum()
        out["up_cut"] = out["up_run"].shift(1).rolling(params.lookback, min_periods=params.lookback).quantile(params.percentile).fillna(np.inf)
        out["down_cut"] = out["down_run"].shift(1).rolling(params.lookback, min_periods=params.lookback).quantile(params.percentile).fillna(np.inf)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        entry = indicators["up_run"].gt(0) & indicators["up_run"].ge(indicators["up_cut"])
        exit_signal = indicators["down_run"].gt(0) & indicators["down_run"].ge(indicators["down_cut"])
        decisions = pd.Series(np.nan, index=data.index)
        decisions.loc[entry] = 1
        decisions.loc[exit_signal] = 0
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
