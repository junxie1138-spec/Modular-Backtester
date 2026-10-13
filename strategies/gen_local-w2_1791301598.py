from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMigrationParams:
    range_lookback: int = 12
    move_threshold: float = 0.7
    edge_threshold: float = 0.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RangeMigrationParams]):
    strategy_id = "gen_local-w2_1791301598"

    @classmethod
    def params_type(cls) -> type[RangeMigrationParams]:
        return RangeMigrationParams

    @classmethod
    def warmup_bars(cls, params: RangeMigrationParams) -> int:
        return max(200, params.range_lookback + 1)

    def indicators(self, data: pd.DataFrame, params: RangeMigrationParams) -> pd.DataFrame:
        daily_range = data["high"] - data["low"]
        midpoint = (data["high"] + data["low"]) / 2
        prior_range = daily_range.rolling(params.range_lookback, min_periods=params.range_lookback).mean().shift(1)
        ma200 = data["close"].rolling(200, min_periods=200).mean()

        result = pd.DataFrame(index=data.index)
        result["range_move"] = ((midpoint - midpoint.shift(1)) / prior_range.where(prior_range.gt(0))).fillna(0.0)
        result["close_position"] = ((data["close"] - data["low"]) / daily_range.where(daily_range.gt(0))).fillna(0.5)
        result["ma200"] = ma200.fillna(0.0)
        result["ready"] = ma200.notna() & prior_range.gt(0) & daily_range.gt(0)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeMigrationParams,
    ) -> SignalFrame:
        ready = indicators["ready"]
        long_entry = ready & data["close"].gt(indicators["ma200"]) & indicators["range_move"].le(-params.move_threshold) & indicators["close_position"].ge(params.edge_threshold)
        short_entry = ready & data["close"].lt(indicators["ma200"]) & indicators["range_move"].ge(params.move_threshold) & indicators["close_position"].le(1.0 - params.edge_threshold)

        desired = np.zeros(len(data), dtype=int)
        long_values = long_entry.to_numpy(dtype=bool)
        short_values = short_entry.to_numpy(dtype=bool)
        position = 0
        exit_at = -1
        for i in range(len(data)):
            if position and i >= exit_at:
                position = 0
            elif position == 0:
                if long_values[i]:
                    position = 1
                    exit_at = i + params.hold_bars
                elif short_values[i]:
                    position = -1
                    exit_at = i + params.hold_bars
            desired[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
