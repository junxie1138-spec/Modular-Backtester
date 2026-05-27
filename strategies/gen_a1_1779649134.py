from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PredatorStrikeParams:
    range_lookback: int = 10
    range_expansion_mult: float = 1.6
    close_position_max: float = 0.25
    hold_bars: int = 4
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[PredatorStrikeParams]):
    strategy_id = "gen_a1_1779649134"

    @classmethod
    def params_type(cls) -> type[PredatorStrikeParams]:
        return PredatorStrikeParams

    @classmethod
    def warmup_bars(cls, params: PredatorStrikeParams) -> int:
        return int(max(params.range_lookback, 1))

    def indicators(self, data: pd.DataFrame, params: PredatorStrikeParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        bar_range = (high - low)
        range_safe = bar_range.where(bar_range > 0.0)
        close_pos = (close - low) / range_safe
        close_pos = close_pos.fillna(0.5).clip(0.0, 1.0)

        lookback = int(max(params.range_lookback, 1))
        avg_range = bar_range.rolling(lookback, min_periods=lookback).mean()
        avg_range_safe = avg_range.where(avg_range > 0.0)
        range_ratio = bar_range / avg_range_safe

        ind = pd.DataFrame(index=data.index)
        ind["bar_range"] = bar_range
        ind["avg_range"] = avg_range
        ind["range_ratio"] = range_ratio
        ind["close_pos"] = close_pos
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PredatorStrikeParams,
    ) -> SignalFrame:
        range_ratio = indicators["range_ratio"].to_numpy(dtype=float, copy=False)
        close_pos = indicators["close_pos"].to_numpy(dtype=float, copy=False)

        expansion = float(params.range_expansion_mult)
        pos_max = float(params.close_position_max)
        hold = int(max(params.hold_bars, 1))

        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        held_until = -1

        for i in range(n):
            if i <= held_until:
                raw_signal[i] = 1
                continue

            rr = range_ratio[i]
            cp = close_pos[i]
            if np.isnan(rr) or np.isnan(cp):
                continue

            if rr >= expansion and cp <= pos_max:
                raw_signal[i] = 1
                held_until = i + hold - 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = float(max(params.base_size, 0.0))
        return SignalFrame(data=df, signal_column="signal", size_column="size")
