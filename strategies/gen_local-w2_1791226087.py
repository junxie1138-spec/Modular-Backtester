from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RelativeSlopeParams:
    trend_span: int = 12
    innovation_span: int = 20
    range_window: int = 30
    lower_position: float = 0.55
    upper_position: float = 0.85
    profit_target: float = 0.03
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[RelativeSlopeParams]):
    strategy_id = "gen_local-w2_1791226087"

    @classmethod
    def params_type(cls) -> type[RelativeSlopeParams]:
        return RelativeSlopeParams

    def warmup_bars(self, params: RelativeSlopeParams) -> int:
        return max(params.trend_span + params.innovation_span + 1, params.range_window)

    def indicators(self, data: pd.DataFrame, params: RelativeSlopeParams) -> pd.DataFrame:
        log_close = np.log(data["close"].where(data["close"] > 0))
        trend = log_close.ewm(span=params.trend_span, min_periods=params.trend_span).mean()
        slope = trend.diff()
        expected_slope = slope.ewm(span=params.innovation_span, min_periods=params.innovation_span).mean()
        range_low = data["low"].rolling(params.range_window, min_periods=params.range_window).min()
        range_high = data["high"].rolling(params.range_window, min_periods=params.range_window).max()
        result = pd.DataFrame(index=data.index)
        result["slope_innovation"] = (slope - expected_slope).fillna(0.0)
        result["range_position"] = ((data["close"] - range_low) / (range_high - range_low).replace(0, np.nan)).clip(0, 1).fillna(0.0)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RelativeSlopeParams,
    ) -> SignalFrame:
        eligible = (
            (indicators["slope_innovation"] > 0)
            & indicators["range_position"].between(params.lower_position, params.upper_position)
        )
        decisions = np.zeros(len(data), dtype=int)
        pending_entry = False
        in_position = False
        entry_price = 0.0
        entry_bar = 0

        for i in range(len(data)):
            if pending_entry:
                entry_price = float(data["open"].iloc[i])
                entry_bar = i
                in_position = True
                pending_entry = False

            if in_position:
                if (
                    float(data["close"].iloc[i]) >= entry_price * (1.0 + params.profit_target)
                    or i - entry_bar + 1 >= params.max_hold
                ):
                    in_position = False
                else:
                    decisions[i] = 1
            elif bool(eligible.iloc[i]) and i + 1 < len(data):
                decisions[i] = 1
                pending_entry = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
