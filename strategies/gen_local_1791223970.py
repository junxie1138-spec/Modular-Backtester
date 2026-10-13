from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuarterTurnParams:
    tail_bars: int = 10
    max_tail_return: float = -0.01
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[QuarterTurnParams]):
    strategy_id = "gen_local_1791223970"

    @classmethod
    def params_type(cls) -> type[QuarterTurnParams]:
        return QuarterTurnParams

    def warmup_bars(self, params: QuarterTurnParams) -> int:
        return int(max(200, params.tail_bars + 1))

    def indicators(self, data: pd.DataFrame, params: QuarterTurnParams) -> pd.DataFrame:
        close = data["close"]
        quarter = pd.Series(data.index.to_period("Q"), index=data.index)
        ma200 = close.rolling(200, min_periods=200).mean()
        result = pd.DataFrame(index=data.index)
        result["quarter_start"] = quarter.ne(quarter.shift(1)).fillna(False)
        result["daily_return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["prior_tail_return"] = close.pct_change(periods=params.tail_bars, fill_method=None).shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["above_ma200"] = close.gt(ma200).fillna(False)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QuarterTurnParams,
    ) -> SignalFrame:
        entries = (
            indicators["quarter_start"]
            & indicators["above_ma200"]
            & indicators["daily_return"].gt(0.0)
            & indicators["prior_tail_return"].le(params.max_tail_return)
        ).fillna(False).to_numpy(dtype=bool)

        raw = np.zeros(len(data), dtype=int)
        active = False
        exit_at = -1
        hold_bars = max(1, int(params.hold_bars))
        for i, wants_entry in enumerate(entries):
            if active and i >= exit_at:
                active = False
            elif not active and wants_entry:
                active = True
                exit_at = i + hold_bars
            raw[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
