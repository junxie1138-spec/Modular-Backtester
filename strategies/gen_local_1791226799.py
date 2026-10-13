from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    path_bars: int = 8
    area_threshold: float = 0.045


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791226799"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(200, 20 + params.path_bars)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        volume = data["volume"].div(data["volume"].rolling(20, min_periods=20).mean().replace(0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        x, y = returns.cumsum(), volume.cumsum()
        accumulated_area = (0.5 * (x.shift(1).fillna(0.0) * volume - y.shift(1).fillna(0.0) * returns)).cumsum()
        x0 = x.shift(params.path_bars).fillna(0.0)
        y0 = y.shift(params.path_bars).fillna(0.0)
        a0 = accumulated_area.shift(params.path_bars).fillna(0.0)
        out["area"] = (accumulated_area - a0 - 0.5 * (x0 * (y - y0) - y0 * (x - x0))).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean().fillna(np.inf)
        out["pullback"] = data["close"].lt(data["close"].shift(params.path_bars)).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        entry = (data["close"] > indicators["ma200"]) & indicators["pullback"] & (indicators["area"] < -params.area_threshold)
        exit_signal = indicators["area"] > params.area_threshold
        df = pd.DataFrame(index=data.index)
        signals = []
        holding = False
        for enter, leave in zip(entry, exit_signal):
            if holding and leave:
                holding = False
            elif not holding and enter:
                holding = True
            signals.append(int(holding))
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
