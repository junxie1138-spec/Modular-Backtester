from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    coil_bars: int = 8
    baseline_bars: int = 40
    compression_ratio: float = 0.70
    close_position_min: float = 0.60
    profit_target: float = 0.05
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791298435"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return params.coil_bars + params.baseline_bars

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        high, low, close = data["high"], data["low"], data["close"]
        bar_range = high - low
        recent = bar_range.rolling(params.coil_bars, min_periods=params.coil_bars).mean()
        baseline = bar_range.shift(params.coil_bars).rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean()
        ceiling = high.shift(2).rolling(params.coil_bars, min_periods=params.coil_bars).max()
        close_position = (close - low).div(bar_range.replace(0, np.nan)).fillna(0.0)

        out = pd.DataFrame(index=data.index)
        out["range_ratio"] = recent.div(baseline.replace(0, np.nan)).fillna(1.0)
        out["floor_confirmed"] = (
            (low > low.shift(1))
            & (low.shift(1) > low.shift(2))
            & (high <= ceiling)
            & (high.shift(1) <= ceiling)
            & (close_position >= params.close_position_min)
            & (close_position.shift(1) >= params.close_position_min)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opening = data["open"].to_numpy()
        closing = data["close"].to_numpy()
        compressed = (indicators["range_ratio"] <= params.compression_ratio).to_numpy()
        confirmed = indicators["floor_confirmed"].to_numpy()
        entry_bar = -1
        entry_price = 0.0

        for i in range(len(data)):
            active = bool(decisions[i - 1]) if i else False
            if active and entry_bar < 0:
                entry_bar = i
                entry_price = float(opening[i])

            if active:
                if closing[i] >= entry_price * (1.0 + params.profit_target) or i - entry_bar + 1 >= params.hold_bars:
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif i >= self.warmup_bars(params) and compressed[i] and confirmed[i]:
                decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
