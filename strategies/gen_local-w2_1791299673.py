from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    vol_window: int = 5
    baseline_window: int = 40
    vol_ratio: float = 0.9


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791299673"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return params.vol_window + params.baseline_window + 1

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change(fill_method=None)
        out["vol"] = returns.rolling(params.vol_window, min_periods=params.vol_window).std()
        out["baseline"] = out["vol"].rolling(params.baseline_window, min_periods=params.baseline_window).median().shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        month_turn = pd.Series((data.index.day >= 26) | (data.index.day <= 7), index=data.index)
        active = (month_turn & indicators["vol"].gt(0) & indicators["vol"].le(indicators["baseline"] * params.vol_ratio)).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = (active & active.shift(1, fill_value=False)).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
