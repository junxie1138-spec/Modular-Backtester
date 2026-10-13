from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StaircaseParams:
    signal_window: int = 5
    rank_window: int = 63
    entry_percentile: float = 0.8
    profit_target: float = 0.012
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StaircaseParams]):
    strategy_id = "gen_local-w2_1791227407"

    @classmethod
    def params_type(cls) -> type[StaircaseParams]:
        return StaircaseParams

    def warmup_bars(self, params: StaircaseParams) -> int:
        return params.signal_window + params.rank_window

    def indicators(self, data: pd.DataFrame, params: StaircaseParams) -> pd.DataFrame:
        daily_range = (data["high"] - data["low"]).replace(0, np.nan)
        body = ((data["close"] - data["open"]) / daily_range).clip(-1, 1).fillna(0.0)
        staircase = (np.sign(data["high"].diff()).fillna(0.0) + np.sign(data["low"].diff()).fillna(0.0)) / 2.0
        body_score = body.rolling(params.signal_window, min_periods=params.signal_window).mean()
        staircase_score = staircase.rolling(params.signal_window, min_periods=params.signal_window).mean()
        result = pd.DataFrame(index=data.index)
        result["body_rank"] = body_score.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True).fillna(0.5)
        result["staircase_rank"] = staircase_score.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True).fillna(0.5)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StaircaseParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        body_ranks = indicators["body_rank"].to_numpy()
        staircase_ranks = indicators["staircase_rank"].to_numpy()
        position = 0
        target = 0
        entry_price = 0.0
        held = 0

        for i in range(len(data)):
            if target != position:
                position = target
                if position:
                    entry_price = float(opens[i])
                    held = 0
            if position:
                held += 1
                gain = position * (float(closes[i]) / entry_price - 1.0)
                if gain >= params.profit_target or held >= params.max_hold_bars:
                    target = 0
            elif body_ranks[i] >= params.entry_percentile and staircase_ranks[i] >= params.entry_percentile:
                target = 1
            elif body_ranks[i] <= 1.0 - params.entry_percentile and staircase_ranks[i] <= 1.0 - params.entry_percentile:
                target = -1
            else:
                target = 0
            raw[i] = target

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
