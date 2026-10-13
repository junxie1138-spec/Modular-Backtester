from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphSeasonalityParams:
    graph_window: int = 8
    degree_gap: int = 2
    profit_target: float = 0.03
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GraphSeasonalityParams]):
    strategy_id = "gen_local-w2_1791227318"

    @classmethod
    def params_type(cls) -> type[GraphSeasonalityParams]:
        return GraphSeasonalityParams

    @staticmethod
    def warmup_bars(params: GraphSeasonalityParams) -> int:
        return params.graph_window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GraphSeasonalityParams) -> pd.DataFrame:
        price = np.log(data["close"].where(data["close"] > 0))
        peak_degree = pd.Series(1, index=data.index, dtype=int)
        trough_degree = pd.Series(1, index=data.index, dtype=int)

        for lag in range(2, params.graph_window + 1):
            peak_visible = price.notna() & price.shift(lag).notna()
            trough_visible = peak_visible.copy()
            for between in range(1, lag):
                line = price + (price.shift(lag) - price) * (between / lag)
                middle = price.shift(between)
                peak_visible &= middle < line
                trough_visible &= middle > line
            peak_degree += peak_visible.fillna(False).astype(int)
            trough_degree += trough_visible.fillna(False).astype(int)

        day = data.index.day
        out = pd.DataFrame(index=data.index)
        out["trough_excess"] = trough_degree - peak_degree
        out["month_turn"] = (day >= 26) | (day <= 3)
        out["midmonth"] = (day >= 10) & (day <= 19)
        out["valid"] = price.shift(params.graph_window).notna()
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GraphSeasonalityParams,
    ) -> SignalFrame:
        long_entry = (
            indicators["valid"]
            & indicators["month_turn"]
            & (indicators["trough_excess"] >= params.degree_gap)
        )
        short_entry = (
            indicators["valid"]
            & indicators["midmonth"]
            & (indicators["trough_excess"] <= -params.degree_gap)
        )

        decisions = np.zeros(len(data), dtype=int)
        position = 0
        previous_decision = 0
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if previous_decision != position:
                position = previous_decision
                if position:
                    entry_price = float(data["open"].iat[i])
                    entry_bar = i

            decision = position
            if position:
                gain = position * (float(data["close"].iat[i]) / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    decision = 0
            elif bool(long_entry.iat[i]):
                decision = 1
            elif bool(short_entry.iat[i]):
                decision = -1

            decisions[i] = decision
            previous_decision = decision

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
