from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlasticTrendParams:
    window: int = 10
    persistence: float = 0.25


class GeneratedStrategy(BaseStrategy[PlasticTrendParams]):
    strategy_id = "gen_local-w2_1791296988"

    @classmethod
    def params_type(cls) -> type[PlasticTrendParams]:
        return PlasticTrendParams

    @classmethod
    def warmup_bars(cls, params: PlasticTrendParams) -> int:
        return params.window + 2

    def indicators(self, data: pd.DataFrame, params: PlasticTrendParams) -> pd.DataFrame:
        level = np.log(data["close"]).ewm(span=params.window, adjust=False).mean()
        slope = level.diff()
        innovation = slope.diff()
        net = innovation.rolling(params.window, min_periods=params.window).sum()
        gross = innovation.abs().rolling(params.window, min_periods=params.window).sum()
        persistence = (net / gross.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "slope": slope.fillna(0.0),
            "persistence": persistence.fillna(0.0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PlasticTrendParams,
    ) -> SignalFrame:
        entry_condition = (indicators["slope"] > 0) & (indicators["persistence"] > params.persistence)
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
